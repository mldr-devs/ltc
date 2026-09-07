# Experiment pipeline, one chain per config file:
#
#   cfg/<exp>.txt            experiment definition (ltc.run flags)
#     -> out/data/<exp>.pkl.lz4        training history
#     -> out/<exp>.csv                 (observation, action) dataset
#     -> out/<exp>.split.json           half/half agent split, shared by both paths
#          -> out/<exp>.split_forest.pkl    distilled random forest
#               -> out/<exp>.forestrun.pkl.lz4   forest agent replayed in the simulator
#          -> out/<exp>.split_sr.pkl        distilled symbolic model
#             (+ out/<exp>.split_sr.scale.json, the decoder scale ltc.run reads back)
#            -> out/<exp>.split_sr.eq.json    the Pareto-front equation that replays best
#               -> out/<exp>.srrun.pkl.lz4       SR agent replayed in the simulator
#
# Adding an experiment means adding a cfg/<name>.txt; nothing here needs editing.

DATA_DIR  := out/data
RUN_DIR   := out/runs

CONFIGS   := $(wildcard cfg/*.txt)
EXPS      := $(notdir $(basename $(CONFIGS)))

HISTORIES     := $(addprefix $(DATA_DIR)/, $(addsuffix .pkl.lz4, $(EXPS)))
CSV_FILES     := $(addprefix out/, $(addsuffix .csv, $(EXPS)))
SPLIT_FILES   := $(addprefix out/, $(addsuffix .split.json, $(EXPS)))
FOREST_MODELS := $(addprefix out/, $(addsuffix .split_forest.pkl, $(EXPS)))
SR_MODELS     := $(addprefix out/, $(addsuffix .split_sr.pkl, $(EXPS)))
SR_PICKS      := $(addprefix out/, $(addsuffix .split_sr.eq.json, $(EXPS)))
FOREST_RUNS   := $(addprefix out/, $(addsuffix .forestrun.pkl.lz4, $(EXPS)))
SR_RUNS       := $(addprefix out/, $(addsuffix .srrun.pkl.lz4, $(EXPS)))
# Trained teacher vs both distillates, overlaid on shared axes.
COMPARES      := $(addprefix out/compare_, $(addsuffix /summary.csv, $(EXPS)))

# One A4 summary page per ltc.run rollout: the training run and both replays.
PAGES         := $(addprefix out/, $(foreach s,train forestrun srrun, $(addsuffix .$(s).page.pdf, $(EXPS))))

# Add --skip_git_check here when running from a dirty worktree; it is passed to
# every ltc.run invocation, training and replay alike.
RUN_FLAGS ?=

# Replay of the distilled policies. Epoch/step counts are overridden rather than
# taken from the cfg: no learning happens, and the `all_*` plots draw one point per
# epoch, so a single epoch would render as blank axes.
REPLAY_EPOCHS ?= 10
REPLAY_STEPS  ?= 2000
# Empty lets ltc.symbolic.sr_select choose, by replaying the whole front; set an
# index to pin one and skip that. PySR's own ranking is not an option worth
# offering here -- it ranks by fit, and on the bursty run its pick is the one
# equation on the front that deadlocks the network.
SR_EQ         ?=
# Extra flags for both replays. Sampling the distilled action is the default: one
# shared deterministic policy puts every station in lockstep, and with argmax the
# replays reach zero throughput however the models are labelled or fit.
#
# --sr_scale 1 disables the calibration: ltc.symbolic.sr_split still fits the scale
# and writes the sidecar, but the replay ignores it. The calibrator maximizes the
# likelihood of hard labels, so it sharpens an already-saturated decoder rather than
# softening it -- on the nonsaturated run it returned a=4.12 with every row clipped
# onto a vertex. Drop the flag to let ltc.run read the sidecar back.
REPLAY_FLAGS  ?= --stochastic_policy --sr_scale 1

# Summary page. The raster panel shows one epoch, so by default it lands on the
# last one (the trained policy) and on the first steps of it.
PAGE_EPOCH      ?= -1
PAGE_ZOOM_STEPS ?= 200
PAGE_ZOOM_START ?= 0
PAGE_SMOOTH     ?= 1
PAGE_FLAGS      ?=

# Distillation size knobs, forwarded to ltc.symbolic.sr_split.
SR_ITERATIONS    ?= 100
SR_POPULATIONS   ?= 10
FOREST_ESTIMATORS ?= 1500
# Set to --balanced to class-balance the symbolic fit. Empty by default: the squared
# loss on simplex-coded labels has E[y|x] = 2p(x)-1 as its minimizer, which is exactly
# what the decoder turns back into a sampling probability, and balancing replaces it
# with the decision boundary. See ltc.symbolic.sr.fit_sr.
SR_BALANCED      ?=

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

define render_page
	python -m ltc.utils.history_page --file "$<" --output "$@" \
		--epoch $(PAGE_EPOCH) --zoom_steps $(PAGE_ZOOM_STEPS) --zoom_start $(PAGE_ZOOM_START) \
		--smooth $(PAGE_SMOOTH) $(PAGE_FLAGS)
endef

.PHONY: all train csv split forest sr sr-select distill forest-run sr-run pages compare report-split clean cleanforestrun cleansrrun
.PRECIOUS: $(HISTORIES) $(CSV_FILES) $(SPLIT_FILES) $(FOREST_MODELS) $(SR_MODELS) $(SR_PICKS)

all: forest-run sr-run pages compare

train: $(HISTORIES)

csv: $(CSV_FILES)

split: $(SPLIT_FILES)

forest: $(FOREST_MODELS)

sr: $(SR_MODELS)

sr-select: $(SR_PICKS)

distill: forest sr

forest-run: $(FOREST_RUNS)

sr-run: $(SR_RUNS)

pages: $(PAGES)

compare: $(COMPARES)

report-split: out/report_split.html

out $(DATA_DIR) $(RUN_DIR):
	mkdir -p $@

# 1. Training run: one history per config file.
$(DATA_DIR)/%.pkl.lz4: cfg/%.txt | $(DATA_DIR) $(RUN_DIR)
	$(call run_ltc,$*,train,)

# 2. Flatten the history into the distillation dataset. The labels are the actions
# the agent actually took: the argmax of its trained Q-network disagrees with them
# on 72% of the steps, and distilling that argmax yields a policy that collides
# permanently.
CSV_LABELS ?= actions
# How many trailing epochs the dataset pools. The last epoch alone is the converged
# policy but barely visits the congested states -- 207 buffer-full steps on the bursty
# run, 26 of them after a collision -- which is too few to fit the teacher's backoff.
CSV_EPOCHS ?= 10

out/%.csv: $(DATA_DIR)/%.pkl.lz4 ltc/symbolic/history2csv.py | out
	python -m ltc.symbolic.history2csv --file "$<" --output "$@" --labels $(CSV_LABELS) \
		--epochs $(CSV_EPOCHS)

# 3. The half/half agent split, written once so both distillations train on the
# same agents and hold out the same ones.
out/%.split.json: out/%.csv ltc/symbolic/split.py | out
	python -m ltc.symbolic.split --file "$<" --output "$@"

# 3a/3b. The two distillations. They share the split and nothing else, so either
# can be refit without disturbing the other.
out/%.split_forest.pkl: out/%.csv out/%.split.json ltc/symbolic/forest_split.py
	python -m ltc.symbolic.forest_split --file "out/$*.csv" --split "out/$*.split.json" \
		--output "out/$*" --n_estimators $(FOREST_ESTIMATORS)

out/%.split_sr.pkl: out/%.csv out/%.split.json ltc/symbolic/sr_split.py ltc/symbolic/sr.py
	python -m ltc.symbolic.sr_split --file "out/$*.csv" --split "out/$*.split.json" \
		--output "out/$*" --pysr_output_dir out/output_split \
		--n_iterations $(SR_ITERATIONS) --n_populations $(SR_POPULATIONS) $(SR_BALANCED)

# 4a. Replay the distilled forest as the station policy, under the experiment's own
# traffic and topology flags.
out/%.forestrun.pkl.lz4: out/%.split_forest.pkl | $(RUN_DIR)
	$(call run_ltc,$*,forestrun,--agent_type forester --forest_pkl $(CURDIR)/out/$*.split_forest.pkl \
		--n_epochs $(REPLAY_EPOCHS) --n_steps $(REPLAY_STEPS) --save_plots $(REPLAY_FLAGS))

# 3c. Pick the equation off the front by replaying all of them. PySR ranks the
# front by fit, which says nothing about whether the decoded expression is a
# working policy: see the table in ltc.symbolic.sr_select. Skipped when SR_EQ pins
# an index, since then there is nothing to choose.
out/%.split_sr.eq.json: out/%.split_sr.pkl cfg/%.txt ltc/symbolic/sr_select.py
ifeq ($(strip $(SR_EQ)),)
	python -m ltc.symbolic.sr_select --sr_pkl "out/$*.split_sr.pkl" --cfg "cfg/$*.txt" \
		--output "$@" --n_epochs $(REPLAY_EPOCHS) --n_steps $(REPLAY_STEPS) \
		--replay_flags "$(REPLAY_FLAGS)"
else
	@echo "SR_EQ=$(SR_EQ) pins the equation; skipping the front replay."
	@printf '{"index": %s, "pinned": true}\n' "$(SR_EQ)" > "$@"
endif

# 4b. Same for the distilled symbolic expression. ltc.run reads the selected index
# out of the .eq.json sidecar unless SR_EQ overrides it.
out/%.srrun.pkl.lz4: out/%.split_sr.pkl out/%.split_sr.eq.json | $(RUN_DIR)
	$(call run_ltc,$*,srrun,--agent_type sr-jax --sr_pkl $(CURDIR)/out/$*.split_sr.pkl $(if $(SR_EQ),--sr_eq $(SR_EQ),) \
		--n_epochs $(REPLAY_EPOCHS) --n_steps $(REPLAY_STEPS) --save_plots $(REPLAY_FLAGS))

# 5. One page per rollout. Each stage keeps its own history path, hence one rule
# per stage rather than a single out/%.page.pdf pattern.
out/%.train.page.pdf: $(DATA_DIR)/%.pkl.lz4 ltc/utils/history_page.py | out
	$(render_page)

out/%.forestrun.page.pdf: out/%.forestrun.pkl.lz4 ltc/utils/history_page.py | out
	$(render_page)

out/%.srrun.page.pdf: out/%.srrun.pkl.lz4 ltc/utils/history_page.py | out
	$(render_page)

# 6. Overlay the trained teacher against both distillates: aggregate throughput
# and Jain's fairness over time, plus the steady-state values side by side.
out/compare_%/summary.csv: $(DATA_DIR)/%.pkl.lz4 out/%.forestrun.pkl.lz4 out/%.srrun.pkl.lz4 plots_compare_distilled.py | out
	python plots_compare_distilled.py \
		--trained "$(DATA_DIR)/$*.pkl.lz4" \
		--forester "out/$*.forestrun.pkl.lz4" --sr "out/$*.srrun.pkl.lz4" \
		--output_dir "out/compare_$*"

out/report_split.html: $(SPLIT_FILES) $(FOREST_MODELS) $(SR_MODELS)
	marimo export html ltc/symbolic/report_split.py -o "$@" -f

clean:
	rm -rf out

cleansrrun:
	rm -rf out/*srun*

cleansforestrun:
	rm -rf out/*forestrun*
