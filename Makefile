# Experiment pipeline, one chain per config file:
#
#   cfg/<exp>.txt            experiment definition (ltc.run flags)
#     -> out/data/<exp>.pkl.lz4        training history
#     -> out/<exp>.csv                 (observation, action) dataset
#     -> out/<exp>.split.done          half/half distillation: forest + SR
#          -> out/<exp>.forestrun.pkl.lz4   forest agent replayed in the simulator
#          -> out/<exp>.srrun.pkl.lz4       SR agent replayed in the simulator
#
# Adding an experiment means adding a cfg/<name>.txt; nothing here needs editing.

DATA_DIR  := out/data
RUN_DIR   := out/runs

CONFIGS   := $(wildcard cfg/*.txt)
EXPS      := $(notdir $(basename $(CONFIGS)))

HISTORIES     := $(addprefix $(DATA_DIR)/, $(addsuffix .pkl.lz4, $(EXPS)))
CSV_FILES     := $(addprefix out/, $(addsuffix .csv, $(EXPS)))
SPLIT_STAMPS  := $(addprefix out/, $(addsuffix .split.done, $(EXPS)))
FOREST_RUNS   := $(addprefix out/, $(addsuffix .forestrun.pkl.lz4, $(EXPS)))
SR_RUNS       := $(addprefix out/, $(addsuffix .srrun.pkl.lz4, $(EXPS)))

# Add --skip_git_check here when running from a dirty worktree; it is passed to
# every ltc.run invocation, training and replay alike.
RUN_FLAGS ?=

# Replay of the distilled policies. Epoch/step counts are overridden rather than
# taken from the cfg: no learning happens, and the `all_*` plots draw one point per
# epoch, so a single epoch would render as blank axes.
REPLAY_EPOCHS ?= 10
REPLAY_STEPS  ?= 2000
SR_EQ         ?= 2

# Distillation size knobs, forwarded to ltc.symbolic.sr_split.
SR_ITERATIONS    ?= 100
SR_POPULATIONS   ?= 10
FOREST_ESTIMATORS ?= 1500

# The flags of one experiment, expanded by the shell at recipe time.
# The '\#' is escaped because make would otherwise read it as a comment.
cfg_flags = $$(sed -e 's/\#.*//' $(CURDIR)/cfg/$(1).txt | tr '\n' ' ')

# ltc.run names its history itself (history_<n>_<n_final>_<seed>_<commit>.pkl.lz4) and
# writes it, plus any --save_plots figures, into the current directory. Every stage
# therefore gets its own scratch directory -- $(RUN_DIR)/<exp>.<stage>, where the plots
# stay -- and the single history produced there is moved to the target.
# $(1) is the experiment, $(2) the stage name, $(3) the extra ltc.run flags.
define run_ltc
	rm -rf $(RUN_DIR)/$(1).$(2)
	mkdir -p $(RUN_DIR)/$(1).$(2)
	cd $(RUN_DIR)/$(1).$(2) && PYTHONPATH=$(CURDIR) python -m ltc.run \
		$(call cfg_flags,$(1)) $(3) $(RUN_FLAGS)
	mv $(RUN_DIR)/$(1).$(2)/history_*.pkl.lz4 "$@"
endef

.PHONY: all train csv split forest-run sr-run report-split clean
.PRECIOUS: $(HISTORIES) $(CSV_FILES)

all: forest-run sr-run

train: $(HISTORIES)

csv: $(CSV_FILES)

split: $(SPLIT_STAMPS)

forest-run: $(FOREST_RUNS)

sr-run: $(SR_RUNS)

report-split: out/report_split.html

out $(DATA_DIR) $(RUN_DIR):
	mkdir -p $@

# 1. Training run: one history per config file.
$(DATA_DIR)/%.pkl.lz4: cfg/%.txt | $(DATA_DIR) $(RUN_DIR)
	$(call run_ltc,$*,train,)

# 2. Flatten the history into the distillation dataset.
out/%.csv: $(DATA_DIR)/%.pkl.lz4 | out
	python -m ltc.symbolic.history2csv --file "$<" --output "$@"

# 3. Distillation on the half/half agent split: the train half fits both a random
# forest and a symbolic expression, the test half is held out. One PySR fit produces
# <exp>.split_forest.pkl, <exp>.split_sr.pkl and <exp>.split.json together, hence the
# single stamp guarding all three.
out/%.split.done: out/%.csv ltc/symbolic/sr_split.py
	python -m ltc.symbolic.sr_split --file "$<" --output "out/$*" --pysr_output_dir out/output_split \
		--n_iterations $(SR_ITERATIONS) --n_populations $(SR_POPULATIONS) --n_estimators $(FOREST_ESTIMATORS)
	touch "$@"

# 4a. Replay the distilled forest as the station policy, under the experiment's own
# traffic and topology flags.
out/%.forestrun.pkl.lz4: out/%.split.done | $(RUN_DIR)
	$(call run_ltc,$*,forestrun,--agent_type forester --forest_pkl $(CURDIR)/out/$*.split_forest.pkl \
		--n_epochs $(REPLAY_EPOCHS) --n_steps $(REPLAY_STEPS) --save_plots)

# 4b. Same for the distilled symbolic expression.
out/%.srrun.pkl.lz4: out/%.split.done | $(RUN_DIR)
	$(call run_ltc,$*,srrun,--agent_type sr-jax --sr_pkl $(CURDIR)/out/$*.split_sr.pkl --sr_eq $(SR_EQ) \
		--n_epochs $(REPLAY_EPOCHS) --n_steps $(REPLAY_STEPS) --save_plots)

out/report_split.html: $(SPLIT_STAMPS)
	marimo export html ltc/symbolic/report_split.py -o "$@" -f

clean:
	rm -rf out
