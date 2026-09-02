PKL_FILES      := $(wildcard data/nosvi/*.pkl.lz4)
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

.PHONY: all distill sr report split report-split forest-run clean

all: distill sr report split report-split

distill: $(FOREST_FILES)

sr: $(SR_STAMPS)

split: $(SR_SPLIT_STAMP)

report: out/report.html

report-split: out/report_split.html

forest-run: $(FOREST_RUN_STAMPS)

.PRECIOUS: $(CSV_FILES)

out:
	mkdir -p out

out/%.csv: data/nosvi/%.pkl.lz4 | out
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
# ltc.run writes its own history_*.pkl.lz4 in the repo root, hence the stamp file.
out/%.forestrun.done: out/%.split.done
	python -m ltc.run --agent_type forester \
		--forest_pkl "out/$*.split_forest.pkl" \
		--n $(word 2,$(subst _, ,$*)) --seed $(word 4,$(subst _, ,$*)) \
		--n_epochs $(FOREST_RUN_EPOCHS) --n_steps $(FOREST_RUN_STEPS) $(FOREST_RUN_FLAGS) --save_plots
	touch "$@"

out/report_split.html: $(SR_SPLIT_STAMP)
	marimo export html ltc/symbolic/report_split.py -o "$@" -f

clean:
	rm -rf out
