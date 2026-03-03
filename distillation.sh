#!/usr/bin/env bash

pushd explain

# Case 1: randomize=False, shuffle-agents=False
python -m ltc.symbolic.nn2sym -n 8
python -m ltc.forest  "qnet_history_5_5_42.pkl 2.lz4sFalserFalseN8.nc"

# Case 2: randomize=False, shuffle-agents=True
python -m ltc.symbolic.nn2sym -n 8 --shuffle-agents
python -m ltc.forest  "qnet_history_5_5_42.pkl 2.lz4sTruerFalseN8.nc"

# Case 3: randomize=True, shuffle-agents=False
python -m ltc.symbolic.nn2sym -n 8 --randomize
python -m ltc.forest  "qnet_history_5_5_42.pkl 2.lz4sFalserTrueN8.nc"

# Case 4: randomize=True, shuffle-agents=True
python -m ltc.symbolic.nn2sym -n 8 --randomize --shuffle-agents
python -m ltc.forest  "qnet_history_5_5_42.pkl 2.lz4sTruerTrueN8.nc"

popd