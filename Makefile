PKL_FILES    := $(wildcard explain/nosvi/*.pkl.lz4)
BASENAMES    := $(notdir $(PKL_FILES:.pkl.lz4=))
CSV_FILES    := $(addprefix out/, $(addsuffix .csv, $(BASENAMES)))
FOREST_FILES := $(addprefix out/, $(addsuffix .forest.pkl, $(BASENAMES)))

.PHONY: all distill clean

all: distill

distill: $(FOREST_FILES)

out:
	mkdir -p out

out/%.csv: explain/nosvi/%.pkl.lz4 | out
	python -m ltc.symbolic.history2csv --file "$<" --output "$@"

out/%.forest.pkl: out/%.csv
	python -m ltc.symbolic.tree --file "$<" --output "$@"

clean:
	rm -rf out
