# Learning to communicate

A JAX-based slotted-channel simulator for multi-agent medium access control, together with the
deep-RL agent it was built for and a set of reference baselines from the literature.

The proposed method is a DDQN agent over an MLP Q-network, trained fully online and without
coordination between stations. Everything else in `ltc/baselines/` is a comparison point.

## Install

```bash
git clone https://github.com/mldr-devs/ltc.git
cd ltc                     # requires Python >= 3.12
pip install -e .
```

Optional extras: `.[gpu]` for CUDA, `.[symbolic]` for the distillation pipeline, `.[analysis]`
for the sensitivity analysis.

## Run

```bash
# the method: 10 DDQN stations, 50 x 2000 slots
python -m ltc.run --n 10 --n_epochs 50 --n_steps 2000 --seed 42
```

Each run writes a compressed history file (`history_<n>_<n_final>_<seed>_<commit>.pkl.lz4`)
holding the final agent states, the per-slot outputs, and the invocation arguments. The commit
hash is recorded automatically, and the run refuses to start with uncommitted tracked changes
unless you pass `--skip_git_check`.

## Cite

```bibtex
@article{ltc,
  title={{Learning to Communicate}},
  author={Szczech, Kamil and Wojnar, Maksymilian and Rusek, Krzysztof and Kosek-Szott, Katarzyna and Szott, Szymon},
  year={2026}
}
```
