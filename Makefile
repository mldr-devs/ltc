# SR distillation pipeline for the v2 simulator, one chain per config file (mirrors main's Makefile):
#
#   $(CFG_DIR)/<exp>.txt -> $(OUT)/data/<exp>.pkl.lz4 -> $(OUT)/<exp>.csv -> $(OUT)/<exp>.split.json
#     -> $(OUT)/<exp>.split_sr.pkl -> $(OUT)/<exp>.split_sr.eq.json -> $(OUT)/<exp>.srrun.pkl.lz4
#     -> $(OUT)/compare_sr_<exp>/summary.csv

OUT       ?= out
DATA_DIR  := $(OUT)/data
RUN_DIR   := $(OUT)/runs

CFG_DIR   ?= cfg
CONFIGS   := $(wildcard $(CFG_DIR)/*.txt)
EXPS      := $(notdir $(basename $(CONFIGS)))

HISTORIES     := $(addprefix $(DATA_DIR)/, $(addsuffix .pkl.lz4, $(EXPS)))
CSV_FILES     := $(addprefix $(OUT)/, $(addsuffix .csv, $(EXPS)))
SPLIT_FILES   := $(addprefix $(OUT)/, $(addsuffix .split.json, $(EXPS)))
SR_MODELS     := $(addprefix $(OUT)/, $(addsuffix .split_sr.pkl, $(EXPS)))
SR_PICKS      := $(addprefix $(OUT)/, $(addsuffix .split_sr.eq.json, $(EXPS)))
SR_RUNS       := $(addprefix $(OUT)/, $(addsuffix .srrun.pkl.lz4, $(EXPS)))
SR_COMPARES   := $(addprefix $(OUT)/compare_sr_, $(addsuffix /summary.csv, $(EXPS)))

RUN_FLAGS ?=

# Set to a lock file to run one training at a time when the GPU is small; distillation stays parallel.
TRAIN_LOCK ?=

# A replay is a single rollout of 100k mini-slots (5000 frame-times); the metrics read its last 20%.
REPLAY_EPOCHS ?= 20
REPLAY_STEPS  ?= 5000
SELECT_EPOCHS ?= $(REPLAY_EPOCHS)
SELECT_STEPS  ?= $(REPLAY_STEPS)
SR_EQ         ?=
REPLAY_FLAGS  ?= --stochastic_policy

CSV_EPOCHS ?= 1

SR_ITERATIONS    ?= 100
SR_POPULATIONS   ?= 10
SR_BALANCED      ?=

export XLA_PYTHON_CLIENT_PREALLOCATE ?= false

# PySR's Julia needs a newer libstdc++ than the system one numpy/jax load first.
JULIA_LIBSTDCXX := $(wildcard $(shell python -c 'import sys; print(sys.prefix)')/julia_env/pyjuliapkg/install/lib/julia/libstdc++.so.6)
$(OUT)/%.split_sr.pkl $(OUT)/%.split_sr.eq.json $(OUT)/%.srrun.pkl.lz4: export LD_PRELOAD := $(JULIA_LIBSTDCXX)

define run_ltc
	rm -rf $(RUN_DIR)/$(1).$(2)
	mkdir -p $(RUN_DIR)/$(1).$(2)
	cd $(RUN_DIR)/$(1).$(2) && $(if $(and $(TRAIN_LOCK),$(filter train,$(2))),flock $(TRAIN_LOCK),) env PYTHONPATH=$(CURDIR) python -m ltc.run \
		@$(abspath $(CFG_DIR))/$(1).txt $(3) $(RUN_FLAGS)
	mv $(RUN_DIR)/$(1).$(2)/history_*.pkl.lz4 "$@"
endef

define check_replay
	python -m ltc.utils.check_replay --file "$@"
endef

.PHONY: all train csv split sr sr-select sr-run compare-sr compare-forest clean cleansrrun
.PRECIOUS: $(HISTORIES) $(CSV_FILES) $(SPLIT_FILES) $(SR_MODELS) $(SR_PICKS) $(OUT)/%.split_forest.pkl $(OUT)/%.forestrun.pkl.lz4

all: compare-sr

train: $(HISTORIES)

csv: $(CSV_FILES)

split: $(SPLIT_FILES)

sr: $(SR_MODELS)

sr-select: $(SR_PICKS)

sr-run: $(SR_RUNS)

compare-sr: $(SR_COMPARES)

$(OUT) $(DATA_DIR) $(RUN_DIR):
	mkdir -p $@

$(DATA_DIR)/%.pkl.lz4: $(CFG_DIR)/%.txt | $(DATA_DIR) $(RUN_DIR)
	$(call run_ltc,$*,train,)

$(OUT)/%.csv: $(DATA_DIR)/%.pkl.lz4 ltc/symbolic/history2csv.py | $(OUT)
	python -m ltc.symbolic.history2csv --file "$<" --output "$@" --epochs $(CSV_EPOCHS)

$(OUT)/%.split.json: $(OUT)/%.csv ltc/symbolic/split.py | $(OUT)
	python -m ltc.symbolic.split --file "$<" --output "$@"

$(OUT)/%.split_sr.pkl: $(OUT)/%.csv $(OUT)/%.split.json ltc/symbolic/sr_split.py ltc/symbolic/sr.py
	for i in 1 2 3; do python -m ltc.symbolic.sr_split --file "$(OUT)/$*.csv" --split "$(OUT)/$*.split.json" \
		--output "$(OUT)/$*" --pysr_output_dir $(OUT)/output_split \
		--n_iterations $(SR_ITERATIONS) --n_populations $(SR_POPULATIONS) $(SR_BALANCED) && break; [ $$i = 3 ] && exit 1; done

$(OUT)/%.split_sr.eq.json: $(OUT)/%.split_sr.pkl $(CFG_DIR)/%.txt ltc/symbolic/sr_select.py
ifeq ($(strip $(SR_EQ)),)
	for i in 1 2 3; do python -m ltc.symbolic.sr_select --sr_pkl "$(OUT)/$*.split_sr.pkl" --cfg "$(CFG_DIR)/$*.txt" \
		--output "$@" --n_epochs $(SELECT_EPOCHS) --n_steps $(SELECT_STEPS) \
		--replay_flags="$(REPLAY_FLAGS) $(RUN_FLAGS)" --work_dir "$(RUN_DIR)/sr_select.$*" && break; [ $$i = 3 ] && exit 1; done
else
	@echo "SR_EQ=$(SR_EQ) pins the complexity cap; skipping the front replay."
	@printf '{"index": %s, "pinned": true}\n' "$(SR_EQ)" > "$@"
endif

$(OUT)/%.srrun.pkl.lz4: $(OUT)/%.split_sr.pkl $(OUT)/%.split_sr.eq.json | $(RUN_DIR)
	$(call run_ltc,$*,srrun,--agent_type sr-jax --sr_pkl $(abspath $(OUT))/$*.split_sr.pkl $(if $(SR_EQ),--sr_eq $(SR_EQ),) \
		--n_epochs $(REPLAY_EPOCHS) --n_steps $(REPLAY_STEPS) $(REPLAY_FLAGS))
	$(call check_replay)

$(OUT)/compare_sr_%/summary.csv: $(DATA_DIR)/%.pkl.lz4 $(OUT)/%.srrun.pkl.lz4 plots_compare_distilled.py | $(OUT)
	python plots_compare_distilled.py --trained "$(DATA_DIR)/$*.pkl.lz4" --sr "$(OUT)/$*.srrun.pkl.lz4" \
		--output_dir "$(OUT)/compare_sr_$*"

FOREST_ESTIMATORS ?= 50
FOREST_COMPARES := $(addprefix $(OUT)/compare_forest_, $(addsuffix /summary.csv, $(EXPS)))

compare-forest: $(FOREST_COMPARES)

$(OUT)/%.split_forest.pkl: $(OUT)/%.csv $(OUT)/%.split.json ltc/symbolic/forest_split.py
	python -m ltc.symbolic.forest_split --file "$(OUT)/$*.csv" --split "$(OUT)/$*.split.json" \
		--output "$(OUT)/$*" --n_estimators $(FOREST_ESTIMATORS)

$(OUT)/%.forestrun.pkl.lz4: $(OUT)/%.split_forest.pkl | $(RUN_DIR)
	$(call run_ltc,$*,forestrun,--agent_type forester --forest_pkl $(abspath $(OUT))/$*.split_forest.pkl \
		--n_epochs $(REPLAY_EPOCHS) --n_steps $(REPLAY_STEPS) $(REPLAY_FLAGS))
	$(call check_replay)

$(OUT)/compare_forest_%/summary.csv: $(DATA_DIR)/%.pkl.lz4 $(OUT)/%.forestrun.pkl.lz4 plots_compare_distilled.py | $(OUT)
	python plots_compare_distilled.py --trained "$(DATA_DIR)/$*.pkl.lz4" --forester "$(OUT)/$*.forestrun.pkl.lz4" \
		$(if $(wildcard $(OUT)/$*.srrun.pkl.lz4),--sr "$(OUT)/$*.srrun.pkl.lz4",) --output_dir "$(OUT)/compare_forest_$*"

clean:
	rm -rf $(OUT)

cleansrrun:
	rm -rf $(OUT)/*srrun*
