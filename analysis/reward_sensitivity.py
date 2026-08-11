import importlib
import json
import math
import os
import pathlib
import runpy
import shutil
import subprocess
import sys
import tempfile
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Callable

import cloudpickle
import lz4.frame
import matplotlib.pyplot as plt
import numpy as np
from SALib.analyze.morris import analyze
from SALib.sample.morris import sample

from ltc.sim.constants import Actions


@dataclass(frozen=True)
class Param:
    name: str
    low: float
    high: float
    cast: Callable[[float], float]


def _positive_int(value: float) -> int:
    return max(1, round(float(value)))


REPO_ROOT = pathlib.Path(__file__).parent.parent.resolve()

PARAMS = (
    Param('TX_REWARD', 0.5, 2.0, float),
    Param('EMPTY_BUFFER_REWARD', 0.0, 1.0, float),
    Param('NO_TX_REWARD', -0.2, 0.2, float),
    Param('NO_TX_PENALTY', -2.0, 0.0, float),
    Param('EMPTY_TX_PENALTY', -1.0, 0.0, float),
    Param('COLLISION_PENALTY', -2.0, 0.0, float),
    Param('MAX_RETRANSMISSION_PENALTY', -2.0, 0.0, float),
    Param('SAFE_IDLE_PERIOD', 10, 50, _positive_int),
    Param('PENALIZED_IDLE_PERIOD', 10, 50, _positive_int),
)

NAMES = [param.name for param in PARAMS]

PROBLEM = {
    'num_vars': len(PARAMS),
    'names': NAMES,
    'bounds': [[param.low, param.high] for param in PARAMS],
}


def traffic_args(traffic: str, n: int) -> list[str]:
    if traffic == 'saturated':
        return ['--traffic_type', 'saturated']
    elif traffic == 'mid':
        rate = 0.3 / n
    elif traffic == 'low':
        rate = 0.1 / n
    else:
        raise ValueError(f'Unknown traffic type: {traffic}')

    loc = math.log(-math.log(1.0 - rate))
    return ['--traffic_type', 'custom', '--f3dB', '1.0', '--scale', '0.0', '--loc', f'{loc:.4f}']


def patch_reward_constants(values: dict) -> dict:
    module = importlib.import_module('ltc.sim.process_output')
    applied = {param.name: param.cast(values[param.name]) for param in PARAMS}

    for name, value in applied.items():
        setattr(module, name, value)

    return applied


def run_child(payload_path: str) -> None:
    payload = json.loads(pathlib.Path(payload_path).read_text())
    patch_reward_constants(payload['params'])

    sys.argv = ['run.py'] + payload['run_args']
    runpy.run_path(str(REPO_ROOT / 'ltc' / 'run.py'), run_name='__main__')


def score(all_outputs, eval_epochs: int) -> float:
    channel_state = np.asarray(all_outputs.channel_state)[-eval_epochs:].reshape(-1)
    actions = np.asarray(all_outputs.actions)[-eval_epochs:]
    actions = actions.reshape(-1, actions.shape[-1])

    successes = channel_state == 1
    per_station = ((actions == Actions.TX.value) & successes[:, None]).sum(0)
    total = per_station.sum()

    if total == 0:
        return 0.0

    fairness = total ** 2 / (len(per_station) * (per_station ** 2).sum())
    return float(successes.mean() * fairness)


def evaluate(params: dict, run_args: list[str], index: int, eval_epochs: int) -> float:
    run_dir = pathlib.Path(tempfile.mkdtemp(prefix=f'sa_run_{index}_'))
    try:
        payload = run_dir / 'params.json'
        payload.write_text(json.dumps({'params': params, 'run_args': run_args}))

        process = subprocess.run(
            [sys.executable, str(pathlib.Path(__file__).resolve()), '--child', str(payload)],
            env={**os.environ, 'GIT_DIR': str(REPO_ROOT / '.git'), 'GIT_WORK_TREE': str(REPO_ROOT)},
            cwd=run_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if process.returncode != 0:
            tail = '\n'.join(process.stderr.strip().splitlines()[-15:])
            raise RuntimeError(f'run {index} exited {process.returncode}:\n{tail}')

        histories = list(run_dir.glob('*.pkl.lz4'))
        if len(histories) != 1:
            raise RuntimeError(f'run {index} wrote {len(histories)} history files, expected 1')

        with lz4.frame.open(histories[0], 'rb') as history_file:
            _, all_outputs, _ = cloudpickle.load(history_file)

        return score(all_outputs, eval_epochs)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def morris_screening(args: Namespace) -> dict:
    run_args = [
        '--n', str(args.n),
        '--n_epochs', str(args.n_epochs),
        '--n_steps', str(args.n_steps),
        '--noise_dims', str(args.noise_dims),
        '--seed', str(args.seed),
        '--skip_git_check',
    ] + traffic_args(args.traffic, args.n)

    samples = sample(PROBLEM, N=args.n_trajectories, num_levels=args.num_levels, seed=args.seed)
    print(f'SA: {len(samples)} runs, {args.n_epochs} epochs x {args.n_steps} steps each')

    log_path = args.output.with_suffix('.jsonl')
    log_path.write_text('')
    print(f'Per-run metrics appended to {log_path}')

    metrics = np.zeros(len(samples))
    for index, row in enumerate(samples):
        params = dict(zip(NAMES, row))
        print(f'[{index + 1}/{len(samples)}] {params}')

        try:
            metrics[index] = evaluate(params, run_args, index, args.eval_epochs)
        except RuntimeError as error:
            print(f'  retrying after: {error}')
            metrics[index] = evaluate(params, run_args, index, args.eval_epochs)

        with log_path.open('a') as log:
            log.write(json.dumps({'index': index, 'metric': metrics[index], 'params': params}) + '\n')

        print(f'  metric = {metrics[index]:.4f}')

    indices = analyze(PROBLEM, samples, metrics, print_to_console=True)

    return {
        'names': NAMES,
        'mu_star': indices['mu_star'].tolist(),
        'mu_star_conf': indices['mu_star_conf'].tolist(),
        'sigma': indices['sigma'].tolist(),
        'n_runs': len(samples),
        'n': args.n,
        'n_epochs': args.n_epochs,
        'n_steps': args.n_steps,
        'seed': args.seed,
        'noise_dims': args.noise_dims,
        'eval_epochs': args.eval_epochs,
        'traffic': args.traffic,
    }


def plot_results(results: dict, output_path: pathlib.Path) -> None:
    names = np.array(results['names'])
    mu_star = np.array(results['mu_star'])
    mu_star_conf = np.array(results['mu_star_conf'])
    sigma = np.array(results['sigma'])

    order = np.argsort(mu_star)[::-1]
    _, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names[order], mu_star[order], xerr=mu_star_conf[order], color='steelblue', capsize=4)
    ax.set_xlabel('mu* (mean absolute elementary effect)')
    ax.set_title('Morris sensitivity analysis, reward constants')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'Plot saved to {output_path}')

    _, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(mu_star, sigma, zorder=3)
    for name, x, y in zip(names, mu_star, sigma):
        ax.annotate(name, (x, y), textcoords='offset points', xytext=(4, 4), fontsize=8)
    ax.set_xlabel('mu*')
    ax.set_ylabel('sigma')
    ax.set_title('Morris: importance vs interactions')
    plt.tight_layout()
    scatter_path = output_path.with_name(f'{output_path.stem}_scatter{output_path.suffix}')
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    print(f'Scatter plot saved to {scatter_path}')


if __name__ == '__main__':
    args = ArgumentParser(description='Morris sensitivity analysis for reward constants.')
    args.add_argument('--seed', type=int, default=42, help='Random seed.')
    args.add_argument('--n', type=int, default=10, help='Stations in the simulation, all running the learning agent.')
    args.add_argument('--n_epochs', type=int, default=50, help='Training epochs per run.')
    args.add_argument('--n_steps', type=int, default=1000, help='Steps per epoch.')
    args.add_argument('--eval_epochs', type=int, default=10, help='Trailing epochs the metric is measured over.')
    args.add_argument('--noise_dims', type=int, default=10, help='Gaussian noise dimensions, matching the method.')
    args.add_argument('--traffic', type=str, default='mid', choices=['saturated', 'mid', 'low'], help='Offered load. Saturated never empties the buffer, so EMPTY_BUFFER_REWARD and EMPTY_TX_PENALTY cannot fire.')
    args.add_argument('--n_trajectories', type=int, default=20, help='Morris trajectories (r).')
    args.add_argument('--num_levels', type=int, default=6, help='Morris grid levels.')
    args.add_argument('--output', type=pathlib.Path, default=pathlib.Path('sa_results.json'), help='Output JSON path.')
    args.add_argument('--child', type=str, help='Run a single training run from a JSON payload. Used internally.')
    args = args.parse_args()

    if args.child:
        run_child(args.child)
        sys.exit(0)

    results = morris_screening(args)
    args.output.write_text(json.dumps(results))
    print(f'\nResults saved to {args.output}')

    plot_results(results, args.output.with_suffix('.png'))
