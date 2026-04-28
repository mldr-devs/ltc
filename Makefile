PKL_FILES    := $(wildcard explain/nosvi/*.pkl.lz4)
BASENAMES    := $(notdir $(PKL_FILES:.pkl.lz4=))
CSV_FILES    := $(addprefix out/, $(addsuffix .csv, $(BASENAMES)))
FOREST_FILES := $(addprefix out/, $(addsuffix .forest.pkl, $(BASENAMES)))
SR_STAMPS    := $(addprefix out/, $(addsuffix .sr.done, $(BASENAMES)))

.PHONY: all distill sr clean

all: distill sr

distill: $(FOREST_FILES)

sr: $(SR_STAMPS)

.PRECIOUS: $(CSV_FILES)

out:
	mkdir -p out

out/%.csv: explain/nosvi/%.pkl.lz4 | out
	python -m ltc.symbolic.history2csv --file "$<" --output "$@"

out/%.forest.pkl: out/%.csv
	python -m ltc.symbolic.tree --file "$<" --output "$@"

out/%.sr.done: out/%.csv
	python -m ltc.symbolic.sr --file "$<" --output "out/$*" --pysr_output_dir out/output
	touch "$@"

clean:
	rm -rf out
