"""Morris sensitivity analysis for reward function constants.

Varies 9 reward/penalty constants using SALib Morris screening (N*(k+1) evaluations).
Metric: fraction of timesteps with exactly one transmitter (channel_state == 1),
measured over the last few epochs of each training run to capture converged behavior.

Usage:
    python analysis/sensitivity_analysis.py [options]
    python analysis/sensitivity_analysis.py --n_epochs 30 --n_trajectories 10
"""
import argparse
import json
import lz4.frame
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import cloudpickle
import matplotlib.pyplot as plt
import numpy as np
from SALib.analyze.morris import analyze
from SALib.sample.morris import sample

REPO_ROOT = pathlib.Path(__file__).parent.parent.resolve()
VENV_PYTHON = str(REPO_ROOT / '.venv' / 'bin' / 'python')
BOOTSTRAP = str(pathlib.Path(__file__).parent / 'sa_bootstrap.py')

EVAL_EPOCHS = 3  # last N epochs used for metric

PARAM_NAMES = [
    'TX_REWARD',
    'EMPTY_BUFFER_REWARD',
    'NO_TX_REWARD',
    'NO_TX_PENALTY',
    'EMPTY_TX_PENALTY',
    'COLLISION_PENALTY',
    'MAX_RETRANSMISSION_PENALTY',
    'SAFE_IDLE_PERIOD',
    'PENALIZED_IDLE_PERIOD',
]

PROBLEM = {
    'num_vars': len(PARAM_NAMES),
    'names': PARAM_NAMES,
    'bounds': [
        [0.5, 2.0],    # TX_REWARD
        [0.0, 1.0],    # EMPTY_BUFFER_REWARD
        [-0.2, 0.2],   # NO_TX_REWARD
        [-2.0, 0.0],   # NO_TX_PENALTY
        [-1.0, 0.0],   # EMPTY_TX_PENALTY
        [-2.0, 0.0],   # COLLISION_PENALTY
        [-2.0, 0.0],   # MAX_RETRANSMISSION_PENALTY
        [10, 50],      # SAFE_IDLE_PERIOD (int)
        [10, 50],      # PENALIZED_IDLE_PERIOD (int)
    ],
}


def run_one(param_dict: dict, seed: int, n_epochs: int, n_steps: int, run_index: int) -> float:
    run_dir = pathlib.Path(tempfile.mkdtemp(prefix=f'sa_run_{run_index}_'))
    try:
        env = {
            **os.environ,
            'GIT_DIR': str(REPO_ROOT / '.git'),
            'GIT_WORK_TREE': str(REPO_ROOT),
            'SA_N_EPOCHS': str(n_epochs),
            'SA_N_STEPS': str(n_steps),
            'SA_SEED': str(seed),
            **{f'SA_{k}': str(v) for k, v in param_dict.items()},
        }
        subprocess.run(
            [VENV_PYTHON, BOOTSTRAP],
            env=env,
            cwd=run_dir,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        history_file = next(run_dir.glob('*.pkl.lz4'))
        with lz4.frame.open(history_file, 'rb') as f:
            _, all_outputs, _ = cloudpickle.load(f)
        n = min(EVAL_EPOCHS, n_epochs)
        last_epochs = np.array(all_outputs.channel_state[-n:])
        return float(np.mean(last_epochs == 1))
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description='Morris sensitivity analysis for reward constants.')
    parser.add_argument('--n_epochs', type=int, default=30, help='Training epochs per run.')
    parser.add_argument('--n_steps', type=int, default=2000, help='Steps per epoch.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--n_trajectories', type=int, default=10, help='Morris trajectories (r).')
    parser.add_argument('--num_levels', type=int, default=6, help='Morris grid levels.')
    parser.add_argument('--output', type=str, default='sa_results.json', help='Output JSON path.')
    args = parser.parse_args()

    X = sample(PROBLEM, N=args.n_trajectories, num_levels=args.num_levels, seed=args.seed)
    n_runs = len(X)
    print(f"SA: {n_runs} runs, {args.n_epochs} epochs × {args.n_steps} steps each", flush=True)

    Y = np.zeros(n_runs)
    for i, row in enumerate(X):
        param_dict = dict(zip(PARAM_NAMES, row))
        print(f"[{i+1}/{n_runs}] {param_dict}", flush=True)
        try:
            Y[i] = run_one(
                param_dict, seed=args.seed, n_epochs=args.n_epochs,
                n_steps=args.n_steps, run_index=i,
            )
        except Exception as e:
            print(f"  FAILED: {e} — using NaN", flush=True)
            Y[i] = float('nan')
        print(f"  metric = {Y[i]:.4f}", flush=True)

    valid = ~np.isnan(Y)
    if not np.all(valid):
        print(f"WARNING: {(~valid).sum()} runs failed, substituting mean for NaN", flush=True)
        Y[~valid] = np.nanmean(Y)

    Si = analyze(PROBLEM, X, Y, print_to_console=True)

    results = {
        'names': PARAM_NAMES,
        'mu_star': Si['mu_star'].tolist(),
        'mu_star_conf': Si['mu_star_conf'].tolist(),
        'sigma': Si['sigma'].tolist(),
        'n_runs': n_runs,
        'n_epochs': args.n_epochs,
        'n_steps': args.n_steps,
        'seed': args.seed,
    }
    out_path = pathlib.Path(args.output)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}", flush=True)

    _plot_results(results, str(out_path.with_suffix('.png')))


def _plot_results(results: dict, output_path: str) -> None:
    names = results['names']
    mu_star = np.array(results['mu_star'])
    mu_star_conf = np.array(results['mu_star_conf'])
    sigma = np.array(results['sigma'])

    order = np.argsort(mu_star)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        np.array(names)[order], mu_star[order],
        xerr=mu_star_conf[order], color='steelblue', capsize=4,
    )
    ax.set_xlabel('μ* (mean absolute elementary effect)')
    ax.set_title('Morris Sensitivity Analysis — Reward Constants')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}", flush=True)

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.scatter(mu_star, sigma, zorder=3)
    for name, mx, sx in zip(names, mu_star, sigma):
        ax2.annotate(name, (mx, sx), textcoords='offset points', xytext=(4, 4), fontsize=8)
    ax2.set_xlabel('μ*')
    ax2.set_ylabel('σ')
    ax2.set_title('Morris: importance vs interactions')
    plt.tight_layout()
    scatter_path = output_path.replace('.png', '_scatter.png')
    plt.savefig(scatter_path, dpi=150)
    print(f"Scatter plot saved to {scatter_path}", flush=True)


if __name__ == '__main__':
    main()
