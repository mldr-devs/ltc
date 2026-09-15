DATA_DIR       := data/nosvi

# Training runs that produce the histories the distillation pipeline consumes.
# ltc.run stamps the current commit into the filename, so it has to be known here too.
COMMIT         := $(shell git rev-parse --short HEAD)
TRAIN_N        ?= 10
TRAIN_EPOCHS   ?= 50
TRAIN_STEPS    ?= 2000
TRAIN_WINDOW   ?= 10
# The two traffic variants only differ by seed in the filename, since ltc.run encodes
# nothing else; keep them distinct or the second run would overwrite the first.
TRAIN_SAT_SEED    ?= 42
TRAIN_NONSAT_SEED ?= 43
TRAIN_NONSAT_TRAFFIC ?= bursty
# Add --skip_git_check here when running from a dirty worktree.
TRAIN_FLAGS    ?=

# Reuse a history already trained for this (n, seed) whatever commit it carries, and
# fall back to a HEAD-stamped name only when there is none. Pinning the target to
# HEAD instead would retrain everything on every new commit.
train_history = $(or $(lastword $(sort $(wildcard $(DATA_DIR)/history_$(TRAIN_N)_$(TRAIN_N)_$(1)_*.pkl.lz4))),$(DATA_DIR)/history_$(TRAIN_N)_$(TRAIN_N)_$(1)_$(COMMIT).pkl.lz4)

TRAIN_SAT      := $(call train_history,$(TRAIN_SAT_SEED))
TRAIN_NONSAT   := $(call train_history,$(TRAIN_NONSAT_SEED))
TRAIN_FILES    := $(TRAIN_SAT) $(TRAIN_NONSAT)

PKL_FILES      := $(sort $(wildcard $(DATA_DIR)/*.pkl.lz4) $(TRAIN_FILES))
BASENAMES      := $(notdir $(PKL_FILES:.pkl.lz4=))
CSV_FILES      := $(addprefix out/, $(addsuffix .csv, $(BASENAMES)))
FOREST_FILES   := $(addprefix out/, $(addsuffix .forest.pkl, $(BASENAMES)))
SR_STAMPS      := $(addprefix out/, $(addsuffix .sr.done, $(BASENAMES)))
SR_SPLIT_STAMP := $(addprefix out/, $(addsuffix .split.done, $(BASENAMES)))
FOREST_RUN_STAMPS := $(addprefix out/, $(addsuffix .forestrun.done, $(BASENAMES)))

# Simulation replaying a distilled random forest through the Forester agent.
FOREST_RUN_EPOCHS ?= 1
FOREST_RUN_STEPS  ?= 2000
FOREST_RUN_FLAGS  ?=

SR_RUN_STAMPS  := $(addprefix out/, $(addsuffix .srrun.done, $(BASENAMES)))
COMPARE_STAMPS := $(addprefix out/, $(addsuffix .compare.done, $(BASENAMES)))

# Simulation replaying a distilled expression through SRJaxAgent.
SR_EQ          ?= 2
# The `all_*` plots draw one point per epoch, so a single epoch renders as blank axes.
SR_RUN_EPOCHS  ?= 10
SR_RUN_STEPS   ?= 2000
# Must match the window the model was distilled from, else the expression reads wrong columns.
SR_WINDOW      ?= 10
SR_RUN_FLAGS   ?=

.PHONY: all train distill sr report split report-split forest-run sr-run compare clean

all: train distill  split compare

train: $(TRAIN_FILES)

distill: $(FOREST_FILES)

sr: $(SR_STAMPS)

split: $(SR_SPLIT_STAMP)

report: out/report.html

report-split: out/report_split.html

forest-run: $(FOREST_RUN_STAMPS)

sr-run: $(SR_RUN_STAMPS)

compare: $(COMPARE_STAMPS)

.PRECIOUS: $(CSV_FILES) $(TRAIN_FILES)

out:
	mkdir -p out

$(DATA_DIR):
	mkdir -p $(DATA_DIR)

# Training runs feeding the distillation pipeline. ltc.run always writes its history
# into the repo root under a name it builds itself, so move it into $(DATA_DIR) after.
$(TRAIN_SAT): | $(DATA_DIR)
	python -m ltc.run --traffic_type saturated \
		--n $(TRAIN_N) --seed $(TRAIN_SAT_SEED) \
		--n_epochs $(TRAIN_EPOCHS) --n_steps $(TRAIN_STEPS) --window_size $(TRAIN_WINDOW) \
		$(TRAIN_FLAGS)
	mv "$(notdir $@)" "$@"

$(TRAIN_NONSAT): | $(DATA_DIR)
	python -m ltc.run --traffic_type $(TRAIN_NONSAT_TRAFFIC) \
		--n $(TRAIN_N) --seed $(TRAIN_NONSAT_SEED) \
		--n_epochs $(TRAIN_EPOCHS) --n_steps $(TRAIN_STEPS) --window_size $(TRAIN_WINDOW) \
		$(TRAIN_FLAGS)
	mv "$(notdir $@)" "$@"

out/%.csv: $(DATA_DIR)/%.pkl.lz4 | out
	python -m ltc.symbolic.history2csv --file "$<" --output "$@"

out/%.forest.pkl: out/%.csv
	python -m ltc.symbolic.tree --file "$<" --output "$@"

out/%.sr.done: out/%.csv ltc/symbolic/sr.py
	python -m ltc.symbolic.sr --file "$<" --output "out/$*" --pysr_output_dir out/output
	touch "$@"

out/report.html: $(SR_STAMPS) $(FOREST_FILES)
	marimo export html ltc/symbolic/report.py -o "$@" -f

out/%.split.done: out/%.csv ltc/symbolic/sr_split.py
	python -m ltc.symbolic.sr_split --file "$<" --output "out/$*" --pysr_output_dir out/output_split
	touch "$@"

# Run the simulator with the distilled split forest as the agent policy.
# The stem is history_<n>_<n_final>_<seed>_<commit>, so n and seed come back out of it.
# ltc.run writes its own history_<n>_<n>_<seed>_<COMMIT>.pkl.lz4 in the repo root; move it
# under a stable name so it does not collide with the sr-run history and `compare` can find it.
out/%.forestrun.done: out/%.split.done
	python -m ltc.run --agent_type forester \
		--forest_pkl "out/$*.split_forest.pkl" \
		--n $(word 2,$(subst _, ,$*)) --seed $(word 4,$(subst _, ,$*)) \
		--n_epochs $(FOREST_RUN_EPOCHS) --n_steps $(FOREST_RUN_STEPS) $(FOREST_RUN_FLAGS) --save_plots
	mv "history_$(word 2,$(subst _, ,$*))_$(word 2,$(subst _, ,$*))_$(word 4,$(subst _, ,$*))_$(COMMIT).pkl.lz4" "out/$*.forestrun.pkl.lz4"
	touch "$@"

# Run the simulator with the distilled expression as the agent policy.
# The stem is history_<n>_<n_final>_<seed>_<commit>, so n and seed come back out of it.
# ltc.run writes its own history_<n>_<n>_<seed>_<COMMIT>.pkl.lz4 in the repo root; move it
# under a stable name so it does not collide with the forest-run history and `compare` can find it.
out/%.srrun.done: out/%.split.done
	python -m ltc.run --agent_type sr-jax \
		--sr_pkl "out/$*.split_sr.pkl" --sr_eq $(SR_EQ) \
		--n $(word 2,$(subst _, ,$*)) --seed $(word 4,$(subst _, ,$*)) \
		--n_epochs $(SR_RUN_EPOCHS) --n_steps $(SR_RUN_STEPS) --window_size $(SR_WINDOW) \
		$(SR_RUN_FLAGS) --save_plots
	mv "history_$(word 2,$(subst _, ,$*))_$(word 2,$(subst _, ,$*))_$(word 4,$(subst _, ,$*))_$(COMMIT).pkl.lz4" "out/$*.srrun.pkl.lz4"
	touch "$@"

# Replay the non-saturated history under the traffic it was trained on.
$(patsubst $(DATA_DIR)/%.pkl.lz4,out/%.srrun.done,$(TRAIN_NONSAT)): SR_RUN_FLAGS += --traffic_type $(TRAIN_NONSAT_TRAFFIC)
$(patsubst $(DATA_DIR)/%.pkl.lz4,out/%.forestrun.done,$(TRAIN_NONSAT)): FOREST_RUN_FLAGS += --traffic_type $(TRAIN_NONSAT_TRAFFIC)

# Overlay the trained teacher against both distillates (random forest + symbolic
# regression): throughput and fairness over time, plus steady-state values side by side.
out/%.compare.done: out/%.forestrun.done out/%.srrun.done plots_compare_distilled.py
	python plots_compare_distilled.py \
		--trained "$(DATA_DIR)/$*.pkl.lz4" \
		--forester "out/$*.forestrun.pkl.lz4" --sr "out/$*.srrun.pkl.lz4" \
		--output_dir "out/compare_$*"
	touch "$@"

out/report_split.html: $(SR_SPLIT_STAMP)
	marimo export html ltc/symbolic/report_split.py -o "$@" -f

clean:
	rm -rf out
