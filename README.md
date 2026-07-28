# LTC - MAB branch

Code for *"Can Bandits Learn to Share a Channel? Distributed Multi-Armed Bandits for Slotted ALOHA"* 
(K. Szczech, M. Wojnar, K. Rusek, K. Kosek-Szott, S. Szott, AGH University of Krakow).

A JAX-based slotted-ALOHA simulator with fully distributed, fully online multi-armed bandit
agents that learn a channel access policy without coordination, signaling, pre-training, or
knowledge of the network size.

## Install

```bash
git clone -b mab https://github.com/mldr-devs/ltc.git
cd ltc                     # requires Python >= 3.12
pip install -e .
```

## Run

```bash
# SALSA (Softmax), N = 20 nodes, T = 30 000 slots, one seed
python -m ltc.run --mab_type softmax --n 20 --n_epochs 600 --n_steps 50 --seed 42
```

| `--mab_type`  | Agent                                                        |
| ------------- | ------------------------------------------------------------ |
| `softmax`     | Softmax / gradient bandit — **SALSA**, the proposed method   |
| `exp3`        | EXP3                                                         |
| `ts`          | Continuous (discounted) Thompson sampling                    |
| `discrete_ts` | Discrete (Dirichlet) Thompson sampling                       |
| `mtoa_l`      | MTOA-L baseline, local reward; null actions set by `--mtoa_L` |
| `mtoa_g`      | MTOA-G baseline, global reward; uses L = N − 1               |

Key options:

- `--n` — number of nodes (default 10).
- `--n_epochs`, `--n_steps` — simulation length is their product, in slots (default 200 × 50 = 10 000).
- `--traffic_type` — `saturated` (full buffer, the default and the setting used in the paper),
  `constant`, `bursty`, or `custom`.
- `--phy_error_prob` — per-slot channel error probability (default 0). Errors are
  indistinguishable from collisions at the agent.
- `--seed` — random seed; the paper reports the mean of five runs, so repeat with `--seed 0..4`.
- `--save_plots` — render figures alongside the saved history.
- `--skip_git_check` — required if you have uncommitted local changes.

More examples:

```bash
# lossy channel
python -m ltc.run --mab_type softmax --n 20 --n_epochs 600 --n_steps 50 --phy_error_prob 0.1

# MTOA-L baseline with L = 19
python -m ltc.run --mab_type mtoa_l --mtoa_L 19 --n 20 --n_epochs 600 --n_steps 50
```

Each run writes a compressed history file (`*.pkl.lz4`) holding the final agent states, the
per-slot outputs, and the invocation arguments.

Agent hyperparameters are currently set in `ltc/run.py`; edit them there to reproduce a
different configuration.

## Hyperparameter tuning

Multi-objective (throughput and fairness) tuning with Optuna:

```bash
pip install -e ".[analysis]"
python -m analysis.tune_mab --mab_type softmax --n_values 5,20
```

## Tests

```bash
python -m unittest
```

## Cite

```bibtex
@inproceedings{szczech2026salsa,
  title={{Can Bandits Learn to Share a Channel? Distributed Multi-Armed Bandits for Slotted ALOHA}},
  author={Szczech, Kamil and Wojnar, Maksymilian and Rusek, Krzysztof and Kosek-Szott, Katarzyna and Szott, Szymon},
  booktitle={TODO},
  year={2026}
}
```

## License

See [LICENSE](LICENSE).
