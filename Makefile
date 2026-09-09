# Experiment pipeline, one chain per config file:
#
#   cfg/<exp>.txt            experiment definition (ltc.run flags)
#     -> $(OUT)/data/<exp>.pkl.lz4        training history
#     -> $(OUT)/<exp>.csv                 (observation, action) dataset
#     -> $(OUT)/<exp>.split.json           half/half agent split, shared by both paths
#          -> $(OUT)/<exp>.split_forest.pkl    distilled random forest
#               -> $(OUT)/<exp>.forestrun.pkl.lz4   forest agent replayed in the simulator
#          -> $(OUT)/<exp>.split_sr.pkl        distilled symbolic model
#             (+ $(OUT)/<exp>.split_sr.scale.json, the decoder scale ltc.run reads back)
#            -> $(OUT)/<exp>.split_sr.eq.json    the Pareto-front equation that replays best
#               -> $(OUT)/<exp>.srrun.pkl.lz4       SR agent replayed in the simulator
#
# Adding an experiment means adding a cfg/<name>.txt; nothing here needs editing.

# Root of every generated artifact. Override to keep a run's outputs apart, e.g.
# `make all OUT=out/sweep-b`; nothing below writes outside it.
OUT       ?= out
DATA_DIR  := $(OUT)/data
RUN_DIR   := $(OUT)/runs

CONFIGS   := $(wildcard cfg/*.txt)
EXPS      := $(notdir $(basename $(CONFIGS)))

HISTORIES     := $(addprefix $(DATA_DIR)/, $(addsuffix .pkl.lz4, $(EXPS)))
CSV_FILES     := $(addprefix $(OUT)/, $(addsuffix .csv, $(EXPS)))
SPLIT_FILES   := $(addprefix $(OUT)/, $(addsuffix .split.json, $(EXPS)))
FOREST_MODELS := $(addprefix $(OUT)/, $(addsuffix .split_forest.pkl, $(EXPS)))
SR_MODELS     := $(addprefix $(OUT)/, $(addsuffix .split_sr.pkl, $(EXPS)))
SR_PICKS      := $(addprefix $(OUT)/, $(addsuffix .split_sr.eq.json, $(EXPS)))
FOREST_RUNS   := $(addprefix $(OUT)/, $(addsuffix .forestrun.pkl.lz4, $(EXPS)))
SR_RUNS       := $(addprefix $(OUT)/, $(addsuffix .srrun.pkl.lz4, $(EXPS)))
# Trained teacher vs both distillates, overlaid on shared axes.
COMPARES      := $(addprefix $(OUT)/compare_, $(addsuffix /summary.csv, $(EXPS)))

# One A4 summary page per ltc.run rollout: the training run and both replays.
PAGES         := $(addprefix $(OUT)/, $(foreach s,train forestrun srrun, $(addsuffix .$(s).page.pdf, $(EXPS))))

# Add --skip_git_check here when running from a dirty worktree; it is passed to
# every ltc.run invocation, training and replay alike.
RUN_FLAGS ?=

# Replay of the distilled policies. Nothing is learned, so a replay is one rollout
# and one epoch; the summary page bins that rollout into step windows rather than
# drawing a point per epoch, so there is nothing left for extra epochs to add.
# Lengthen REPLAY_STEPS, not REPLAY_EPOCHS, when a replay needs to run longer.
# 3000 steps against ltc.utils.metrics.DEFAULT_LAST_PERCENT of 0.5: the first 1500
# are dropped as warm-up, comfortably past the few hundred a replay needs to settle,
# and the remaining 1500 are enough for Jain's index to stop being biased by how few
# successes each station happened to get.
REPLAY_EPOCHS ?= 1
REPLAY_STEPS  ?= 3000
# Candidate replays in ltc.symbolic.sr_select must match the production replay, or
# the winner is picked on behaviour it never shows: a metastable bursty policy that
# escapes mutual collision late scored 0.0689 over 30000 steps and 0.002 over 3000.
# The price is a ranking decided by 1500 steps, which still separates working from
# deadlocked -- the point of the selection -- but not near-ties.
SELECT_EPOCHS ?= $(REPLAY_EPOCHS)
SELECT_STEPS  ?= $(REPLAY_STEPS)
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

# Summary page. The whole page is one rollout: a replay has only the one, and a
# training run defaults to its last epoch, i.e. the converged policy. The raster
# zooms into a stretch of that same rollout, shaded on every curve.
PAGE_EPOCH      ?= -1
PAGE_ZOOM_STEPS ?= 200
PAGE_ZOOM_START ?= 2000
# Steps summarised by one point of every curve. Empty lets the page pick
# n_steps // 100.
PAGE_WINDOW     ?=
PAGE_SMOOTH     ?= 1
PAGE_FLAGS      ?=

# Distillation size knobs, forwarded to ltc.symbolic.sr_split.
SR_ITERATIONS    ?= 100
SR_POPULATIONS   ?= 10
FOREST_ESTIMATORS ?= 50
# Set to --balanced to class-balance the forest. Empty by default: balancing
# deadlocked the replayed forest in every cell of a {50, 1500} trees x {pooled, last
# epoch} grid, while every unweighted fit reached full throughput. See
# ltc.symbolic.forest_split.fit_forest_split for the table.
FOREST_BALANCED  ?=
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
		--smooth $(PAGE_SMOOTH) $(if $(PAGE_WINDOW),--window $(PAGE_WINDOW),) $(PAGE_FLAGS)
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

report-split: $(OUT)/report_split.html

$(OUT) $(DATA_DIR) $(RUN_DIR):
	mkdir -p $@

# 1. Training run: one history per config file.
$(DATA_DIR)/%.pkl.lz4: cfg/%.txt | $(DATA_DIR) $(RUN_DIR)
	$(call run_ltc,$*,train,)

# 2. Flatten the history into the distillation dataset. The labels are the actions
# the agent actually took: the argmax of its trained Q-network disagrees with them
# on 72% of the steps, and distilling that argmax yields a policy that collides
# permanently.
CSV_LABELS ?= actions
# Trailing epochs pooled into the dataset. 1 is the converged policy alone. Pooling
# 10 measured no better once class_weight went (0.0727 against 0.0747 replayed) and
# mixes in epochs the teacher had not converged in.
CSV_EPOCHS ?= 1

$(OUT)/%.csv: $(DATA_DIR)/%.pkl.lz4 ltc/symbolic/history2csv.py | $(OUT)
	python -m ltc.symbolic.history2csv --file "$<" --output "$@" --labels $(CSV_LABELS) \
		--epochs $(CSV_EPOCHS)

# 3. The half/half agent split, written once so both distillations train on the
# same agents and hold out the same ones.
$(OUT)/%.split.json: $(OUT)/%.csv ltc/symbolic/split.py | $(OUT)
	python -m ltc.symbolic.split --file "$<" --output "$@"

# 3a/3b. The two distillations. They share the split and nothing else, so either
# can be refit without disturbing the other.
$(OUT)/%.split_forest.pkl: $(OUT)/%.csv $(OUT)/%.split.json ltc/symbolic/forest_split.py
	python -m ltc.symbolic.forest_split --file "$(OUT)/$*.csv" --split "$(OUT)/$*.split.json" \
		--output "$(OUT)/$*" --n_estimators $(FOREST_ESTIMATORS) $(FOREST_BALANCED)

$(OUT)/%.split_sr.pkl: $(OUT)/%.csv $(OUT)/%.split.json ltc/symbolic/sr_split.py ltc/symbolic/sr.py
	python -m ltc.symbolic.sr_split --file "$(OUT)/$*.csv" --split "$(OUT)/$*.split.json" \
		--output "$(OUT)/$*" --pysr_output_dir $(OUT)/output_split \
		--n_iterations $(SR_ITERATIONS) --n_populations $(SR_POPULATIONS) $(SR_BALANCED)

# 4a. Replay the distilled forest as the station policy, under the experiment's own
# traffic and topology flags.
$(OUT)/%.forestrun.pkl.lz4: $(OUT)/%.split_forest.pkl | $(RUN_DIR)
	$(call run_ltc,$*,forestrun,--agent_type forester --forest_pkl $(abspath $(OUT))/$*.split_forest.pkl \
		--n_epochs $(REPLAY_EPOCHS) --n_steps $(REPLAY_STEPS) --save_plots $(REPLAY_FLAGS))

# 3c. Pick the equation off the front by replaying all of them. PySR ranks the
# front by fit, which says nothing about whether the decoded expression is a
# working policy: see the table in ltc.symbolic.sr_select. Skipped when SR_EQ pins
# an index, since then there is nothing to choose.
$(OUT)/%.split_sr.eq.json: $(OUT)/%.split_sr.pkl cfg/%.txt ltc/symbolic/sr_select.py
ifeq ($(strip $(SR_EQ)),)
	python -m ltc.symbolic.sr_select --sr_pkl "$(OUT)/$*.split_sr.pkl" --cfg "cfg/$*.txt" \
		--output "$@" --n_epochs $(SELECT_EPOCHS) --n_steps $(SELECT_STEPS) \
		--replay_flags "$(REPLAY_FLAGS)" --work_dir "$(RUN_DIR)/sr_select.$*"
else
	@echo "SR_EQ=$(SR_EQ) pins the equation; skipping the front replay."
	@printf '{"index": %s, "pinned": true}\n' "$(SR_EQ)" > "$@"
endif

# 4b. Same for the distilled symbolic expression. ltc.run reads the selected index
# out of the .eq.json sidecar unless SR_EQ overrides it.
$(OUT)/%.srrun.pkl.lz4: $(OUT)/%.split_sr.pkl $(OUT)/%.split_sr.eq.json | $(RUN_DIR)
	$(call run_ltc,$*,srrun,--agent_type sr-jax --sr_pkl $(abspath $(OUT))/$*.split_sr.pkl $(if $(SR_EQ),--sr_eq $(SR_EQ),) \
		--n_epochs $(REPLAY_EPOCHS) --n_steps $(REPLAY_STEPS) --save_plots $(REPLAY_FLAGS))

# 5. One page per rollout. Each stage keeps its own history path, hence one rule
# per stage rather than a single $(OUT)/%.page.pdf pattern.
$(OUT)/%.train.page.pdf: $(DATA_DIR)/%.pkl.lz4 ltc/utils/history_page.py | $(OUT)
	$(render_page)

$(OUT)/%.forestrun.page.pdf: $(OUT)/%.forestrun.pkl.lz4 ltc/utils/history_page.py | $(OUT)
	$(render_page)

$(OUT)/%.srrun.page.pdf: $(OUT)/%.srrun.pkl.lz4 ltc/utils/history_page.py | $(OUT)
	$(render_page)

# 6. Overlay the trained teacher against both distillates: aggregate throughput
# and Jain's fairness over time, plus the steady-state values side by side.
$(OUT)/compare_%/summary.csv: $(DATA_DIR)/%.pkl.lz4 $(OUT)/%.forestrun.pkl.lz4 $(OUT)/%.srrun.pkl.lz4 plots_compare_distilled.py | $(OUT)
	python plots_compare_distilled.py \
		--trained "$(DATA_DIR)/$*.pkl.lz4" \
		--forester "$(OUT)/$*.forestrun.pkl.lz4" --sr "$(OUT)/$*.srrun.pkl.lz4" \
		--output_dir "$(OUT)/compare_$*"

$(OUT)/report_split.html: $(SPLIT_FILES) $(FOREST_MODELS) $(SR_MODELS)
	marimo export html ltc/symbolic/report_split.py -o "$@" -f

clean:
	rm -rf $(OUT)

cleansrrun:
	rm -rf $(OUT)/*srrun*

cleanforestrun:
	rm -rf $(OUT)/*forestrun*
