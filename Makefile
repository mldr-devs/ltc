PKL_FILES      := $(wildcard data/nosvi/*.pkl.lz4)
BASENAMES      := $(notdir $(PKL_FILES:.pkl.lz4=))
CSV_FILES      := $(addprefix out/, $(addsuffix .csv, $(BASENAMES)))
FOREST_FILES   := $(addprefix out/, $(addsuffix .forest.pkl, $(BASENAMES)))
SR_STAMPS      := $(addprefix out/, $(addsuffix .sr.done, $(BASENAMES)))
SR_SPLIT_STAMP := $(addprefix out/, $(addsuffix .split.done, $(BASENAMES)))

.PHONY: all distill sr report split report-split clean

all: distill sr report split report-split

distill: $(FOREST_FILES)

sr: $(SR_STAMPS)

split: $(SR_SPLIT_STAMP)

report: out/report.html

report-split: out/report_split.html

.PRECIOUS: $(CSV_FILES)

out:
	mkdir -p out

out/%.csv: explain/nosvi/%.pkl.lz4 | out
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

out/report_split.html: $(SR_SPLIT_STAMP)
	marimo export html ltc/symbolic/report_split.py -o "$@" -f

clean:
	rm -rf out
