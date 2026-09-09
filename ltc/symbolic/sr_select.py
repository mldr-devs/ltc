"""Pick an equation off the PySR Pareto front by replaying each one.

PySR ranks its front by fit: ``model_selection='best'`` maximizes the score (loss
drop per unit of complexity), ``'accuracy'`` takes the lowest loss. Neither
predicts whether the decoded expression is a usable channel-access policy, and on
the bursty run the two disagree with the simulator in opposite directions:

    idx  complexity     loss   score   throughput
      1           3  0.008921   0.622       0.0660
      2           4  0.004046   0.791       0.0000   <- 'best' picks this
      3           5  0.003022   0.292       0.0665
      4          12  0.002995   0.001       0.0005
      6          17  0.002973   0.002       0.0665   <- 'accuracy' picks this

The equation with the *worst* loss on that list runs fine; the second- and
fourth-best collapse to zero. The reason is distribution shift. The squared loss
is averaged over the teacher's own states, and the teacher almost never sits in a
congested one -- 0.9% of the bursty rows have a full buffer. Whether the policy
deadlocks is decided entirely inside that 0.9%: a station whose decoded
persistence stays near 1.0 when its frame fails collides with every other station
forever, buffers stop draining, and the run flatlines. A 0.001 difference in loss
is the difference between 0.0 and full throughput, so the loss cannot rank these.

So rank them by what they are for. Every candidate is replayed through
``ltc.run`` under the experiment's own traffic and topology flags, and the one
with the highest steady-state throughput wins, ties going to the simpler
expression. The choice lands in ``<prefix>.split_sr.eq.json``, which ``ltc.run``
reads back the way it reads the scale sidecar.

This makes the selection on-policy: the symbolic model is chosen by how it
behaves in the environment, not by how closely it fits the teacher. That is a
deliberate departure from pure distillation and worth stating as such wherever
these results are reported.
"""

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys

import cloudpickle
import lz4.frame

from ltc.utils.history import unpack_history
from ltc.utils.metrics import steady_state_metrics


def front_indices(sr_model) -> list[int]:
    """Every equation index on the front, simplest first."""
    equations = sr_model.equations_
    if isinstance(equations, list):  # multi-output; this pipeline fits one
        equations = equations[0]
    return [int(i) for i in equations.index]


def replay(sr_pkl: str, index: int, cfg_flags: list[str], replay_flags: list[str],
           work_dir: str) -> str:
    """Run one equation through ltc.run and return the history it wrote.

    ltc.run names its own history and drops it in the working directory, so each
    candidate gets a scratch directory of its own. That directory has to sit inside
    the repository: ltc.run stamps the commit hash into the filename and shells out
    to git for it, which fails anywhere else -- hence out/runs rather than a temp
    directory.
    """
    run_dir = os.path.join(work_dir, f"eq{index}")
    os.makedirs(run_dir, exist_ok=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "ltc.run",
         *cfg_flags,
         "--agent_type", "sr-jax",
         "--sr_pkl", os.path.abspath(sr_pkl),
         "--sr_eq", str(index),
         *replay_flags],
        cwd=run_dir, env=env, capture_output=True, text=True,
    )
    if result.returncode != 0:
        # The replay output is otherwise swallowed, and a candidate that crashes
        # looks the same as one that simply ran badly.
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise RuntimeError(f"Replay of equation {index} failed (exit {result.returncode}).")

    histories = [f for f in os.listdir(run_dir) if f.endswith(".pkl.lz4")]
    if len(histories) != 1:
        raise RuntimeError(f"Expected one history in {run_dir}, found {histories}")
    return os.path.join(run_dir, histories[0])


def score_equation(history_path: str) -> tuple[float, float]:
    with lz4.frame.open(history_path, "rb") as f:
        payload = cloudpickle.load(f)
    _, history, _ = unpack_history(payload)
    return steady_state_metrics(history)


def read_cfg_flags(path: str) -> list[str]:
    """The ltc.run flags of a cfg file: one per line, '#' starts a comment."""
    flags = []
    with open(path) as f:
        for line in f:
            flags.extend(line.split("#", 1)[0].split())
    return flags


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Replay every equation on the PySR front and record the best one."
    )
    parser.add_argument("--sr_pkl", type=str, required=True,
                        help="Fitted model from ltc.symbolic.sr_split.")
    parser.add_argument("--cfg", type=str, required=True,
                        help="Experiment cfg file whose ltc.run flags each replay inherits.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .json. Defaults to <sr_pkl without .pkl>.eq.json, "
                             "which is where ltc.run looks for it.")
    parser.add_argument("--n_epochs", type=int, default=1,
                        help="Epochs per candidate replay. Match the production replay: a "
                             "longer run ranks more steadily but can score a candidate in a "
                             "regime the real replay never reaches.")
    parser.add_argument("--n_steps", type=int, default=3000, help="Steps per epoch.")
    parser.add_argument("--replay_flags", type=str, default="--stochastic_policy --sr_scale 1",
                        help="Extra ltc.run flags shared by every candidate. Must match the "
                             "flags the final replay uses, or the winner is chosen under "
                             "conditions it will not run in.")
    parser.add_argument("--tolerance", type=float, default=0.1,
                        help="Relative throughput window inside which equations count as "
                             "tied and the simplest wins. The default is the measured "
                             "run-to-run spread: over five seeds, complexity 3 and "
                             "complexity 28 both averaged 0.0700 with a 10% standard "
                             "deviation, so a single replay cannot rank them.")
    parser.add_argument("--work_dir", type=str, default=None,
                        help="Scratch directory for the candidate replays. Defaults to "
                             "out/runs/sr_select.<model>, and must stay inside the "
                             "repository because ltc.run reads the commit hash from git.")
    parser.add_argument("--keep_runs", action="store_true", default=False,
                        help="Keep the per-candidate scratch directories for inspection.")
    args = parser.parse_args()

    with open(args.sr_pkl, "rb") as f:
        sr_model = pickle.load(f)

    equations = sr_model.equations_
    if isinstance(equations, list):
        equations = equations[0]

    cfg_flags = read_cfg_flags(args.cfg)
    replay_flags = [
        *args.replay_flags.split(),
        "--n_epochs", str(args.n_epochs),
        "--n_steps", str(args.n_steps),
        "--skip_git_check",
    ]

    stem = os.path.basename(args.sr_pkl).removesuffix(".pkl")
    work_dir = args.work_dir or os.path.join("out", "runs", f"sr_select.{stem}")
    shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir, exist_ok=True)

    candidates = []
    try:
        for index in front_indices(sr_model):
            history_path = replay(args.sr_pkl, index, cfg_flags, replay_flags, work_dir)
            throughput, fairness = score_equation(history_path)
            row = equations.loc[index]
            candidates.append({
                "index": index,
                "complexity": int(row["complexity"]),
                "loss": float(row["loss"]),
                "equation": str(row["equation"]),
                "throughput": throughput,
                "fairness": fairness,
            })
            print(f"  eq {index:2d} (complexity {int(row['complexity']):2d}, "
                  f"loss {float(row['loss']):.6g}): throughput {throughput:.4f}, "
                  f"fairness {fairness:.3f}")
    finally:
        if not args.keep_runs:
            shutil.rmtree(work_dir, ignore_errors=True)
        else:
            print(f"Candidate runs kept in {work_dir}")

    if not candidates:
        raise RuntimeError("The model has no equations to choose from.")

    # Simplest equation whose throughput is within --tolerance of the best. Ranking
    # on throughput alone picks the top of the noise: one replay has a ~10% spread,
    # so it chose complexity 28 over complexity 3 for a 1.7% gap that vanished over
    # five seeds.
    top = max(c["throughput"] for c in candidates)
    tied = [c for c in candidates if c["throughput"] >= top * (1 - args.tolerance)]
    best = min(tied, key=lambda c: (c["complexity"], -c["throughput"]))
    pysr_pick = int(sr_model.get_best().name)

    output = args.output or f"{args.sr_pkl.removesuffix('.pkl')}.eq.json"
    with open(output, "w") as f:
        json.dump({"index": best["index"], "throughput": best["throughput"],
                   "fairness": best["fairness"], "equation": best["equation"],
                   "complexity": best["complexity"], "pysr_pick": pysr_pick,
                   "tolerance": args.tolerance, "tied": [c["index"] for c in tied],
                   "candidates": candidates}, f, indent=2)

    print(f"\nSelected eq {best['index']} (complexity {best['complexity']}, "
          f"loss {best['loss']:.6g}): {best['equation']}")
    print(f"  throughput {best['throughput']:.4f}, fairness {best['fairness']:.3f}")
    if len(tied) > 1:
        print(f"  simplest of {len(tied)} within {args.tolerance:.0%} of the best "
              f"({top:.4f}): eq {[c['index'] for c in tied]}")
    if best["index"] != pysr_pick:
        chosen = next(c for c in candidates if c["index"] == pysr_pick)
        print(f"  PySR's own pick was eq {pysr_pick}, which replays at "
              f"throughput {chosen['throughput']:.4f}.")
    print(f"Saved: {output}")
