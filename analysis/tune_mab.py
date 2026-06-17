import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np
import optuna

from reinforced_lib.agents.mab import Exp3, Softmax
from ltc.agents import DCF, DiscountedThompsonSampling
from ltc.sim import InitialStateConf, cox_traffic
from ltc.sim.constants import INITIAL_CAPACITY, Actions
from ltc.utils.scan_states import Carry
from ltc.run import init_agents, init_traffic, rl_step


def build_mab(mab_type, params, num_actions=2):
    if mab_type == 'exp3':
        return Exp3(n_arms=num_actions, **params)
    elif mab_type == 'ts':
        return DiscountedThompsonSampling(n_arms=num_actions, **params)
    elif mab_type == 'softmax':
        return Softmax(n_arms=num_actions, **params)
    raise ValueError(f'Unknown MAB type: {mab_type}')


def run_sim(mab, n=10, n_epochs=200, n_steps=50, seed=42):
    key = jax.random.key(seed)
    actions = jnp.zeros(n, dtype=int)
    buffer_states = jnp.zeros(n, dtype=int)
    power_states = jnp.full(n, INITIAL_CAPACITY, dtype=int)
    channel_state = 0
    obs = jnp.zeros((n, 1, 6), dtype=int)
    rewards = jnp.zeros(n)
    terminals = jnp.full(n, False, dtype=bool)
    active = jnp.ones(n, dtype=bool)

    key, init_key = jax.random.split(key)
    mab_states, mab_step = init_agents(mab, init_key, n, has_obs=False)

    dcf = DCF()
    key, init_key = jax.random.split(key)
    legacy_states, legacy_step = init_agents(dcf, init_key, 0, has_obs=True)

    key, init_key = jax.random.split(key)
    traffic = cox_traffic(f3dB=1.0, loc=5.0, scale=0.0, initial_state=InitialStateConf.ZERO)
    traffic_states, traffic_step = init_traffic(traffic, init_key, n)

    rl_step_fn, _, _ = rl_step(
        mab_step, legacy_step, traffic_step, n, n, 0.05,
        noise_dims=0, zero_obs=False,
    )
    rl_step_fn = jax.jit(rl_step_fn)
    carry = Carry(
        mab_states, legacy_states, traffic_states, buffer_states, power_states,
        channel_state, key, obs, actions, rewards, terminals, active
    )
    all_outputs = []
    for epoch in range(n_epochs):
        global_steps = jnp.arange(epoch * n_steps, (epoch + 1) * n_steps, dtype=jnp.int32)
        carry, output = jax.lax.scan(rl_step_fn, carry, xs=global_steps)
        all_outputs.append(output)

    all_outputs = jax.tree.map(lambda *x: jnp.stack(x), *all_outputs)
    return all_outputs


def compute_metrics(all_outputs, n_epochs, n_steps):
    """Returns (fairness, throughput) measured on the converged tail of the run."""
    cutoff = int(0.7 * n_epochs)
    ch = np.array(all_outputs.channel_state[cutoff:])

    # Throughput: fraction of slots with exactly one successful transmission.
    throughput = float(np.mean(ch == 1))

    # Fairness: Jain's index over per-station successful transmissions.
    actions = np.array(all_outputs.actions[cutoff:])
    tx_mask = (actions == Actions.TX.value)
    ch_broadcast = (ch == 1)[..., None]
    succ_tx = (tx_mask & ch_broadcast).sum(axis=(0, 1)).astype(float)

    n = succ_tx.shape[0]
    if succ_tx.sum() == 0:
        fairness = 0.0
    else:
        fairness = float((succ_tx.sum() ** 2) / (n * (succ_tx ** 2).sum()))

    return fairness, throughput


def evaluate(mab_type, params, n_values, seeds, n_epochs=200, n_steps=50, agg='mean'):
    """Evaluate one parameter set across several network sizes and seeds.

    Returns (fairness_agg, throughput_agg, per_n) where per_n maps each n to its
    (fairness, throughput) averaged over seeds. Aggregation across n is the mean
    (smooth) or the worst-case min (robust 'works for small and large network').
    """
    reducer = np.mean if agg == 'mean' else np.min
    per_n = {}
    fair_per_n, thr_per_n = [], []

    for n in n_values:
        fair_seeds, thr_seeds = [], []
        for seed in seeds:
            try:
                mab = build_mab(mab_type, params)
                outputs = run_sim(mab, n=n, n_epochs=n_epochs, n_steps=n_steps, seed=seed)
                f, t = compute_metrics(outputs, n_epochs, n_steps)
            except Exception:
                f, t = 0.0, 0.0
            fair_seeds.append(f)
            thr_seeds.append(t)
        f_n, t_n = float(np.mean(fair_seeds)), float(np.mean(thr_seeds))
        per_n[n] = (f_n, t_n)
        fair_per_n.append(f_n)
        thr_per_n.append(t_n)

    return float(reducer(fair_per_n)), float(reducer(thr_per_n)), per_n


def suggest_params(trial, mab_type):
    if mab_type == 'exp3':
        return {
            'gamma': trial.suggest_float('gamma', 0.001, 0.5, log=True),
            'min_reward': -1.,
            'max_reward': 1.,
        }
    elif mab_type == 'ts':
        return {
            'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
            'beta': trial.suggest_float('beta', 1e-3, 10.0, log=True),
            'lam': trial.suggest_float('lam', 1e-3, 2.0, log=True),
            'mu': trial.suggest_float('mu', -1.0, 1.0),
            'gamma': trial.suggest_float('gamma', 0.5, 0.9999, log=True),
        }
    elif mab_type == 'softmax':
        return {
            'lr': trial.suggest_float('lr', 0.01, 10.0, log=True),
            'alpha': trial.suggest_float('alpha', 0., 1.0),
            'tau': trial.suggest_float('tau', 0.1, 10.0, log=True),
        }
    raise ValueError(f'Unknown MAB type: {mab_type}')


def make_objective(mab_type, n_values, seeds, n_epochs=200, n_steps=50, agg='mean'):
    def objective(trial):
        params = suggest_params(trial, mab_type)
        fairness, throughput, per_n = evaluate(
            mab_type, params, n_values, seeds, n_epochs=n_epochs, n_steps=n_steps, agg=agg
        )
        for n, (f_n, t_n) in per_n.items():
            trial.set_user_attr(f'fairness_n{n}', f_n)
            trial.set_user_attr(f'throughput_n{n}', t_n)
        # Two objectives: maximize fairness AND throughput -> Optuna returns a Pareto front.
        return fairness, throughput

    return objective


def pick_balanced(trials):
    """Knee point of the Pareto front: max of min-max normalized (fairness + throughput)."""
    fair = np.array([t.values[0] for t in trials])
    thr = np.array([t.values[1] for t in trials])

    def norm(x):
        rng = x.max() - x.min()
        return (x - x.min()) / rng if rng > 1e-12 else np.zeros_like(x)

    score = norm(fair) + norm(thr)
    return trials[int(np.argmax(score))]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Multi-objective tuning of MAB MAC agents for throughput & fairness, '
                    'robust across network sizes.'
    )
    parser.add_argument('--mab_type', required=True, choices=['exp3', 'ts', 'softmax'])
    parser.add_argument('--n_trials', type=int, default=100)
    parser.add_argument('--n_epochs', type=int, default=500)
    parser.add_argument('--n_steps', type=int, default=50)
    parser.add_argument('--n_values', type=str, default='5,10,20,30,40',
                        help='Comma-separated network sizes to tune across.')
    parser.add_argument('--seeds', type=str, default='1,42,101',
                        help='Comma-separated seeds averaged per network size.')
    parser.add_argument('--agg', choices=['mean', 'min'], default='mean',
                        help="How to aggregate metrics across n: 'mean' (smooth) or 'min' (worst-case robust).")
    parser.add_argument('--n_jobs', type=int, default=8)
    parser.add_argument('--out', type=str, default=None, help='Path to write the Pareto front JSON.')
    args = parser.parse_args()

    n_values = [int(x) for x in args.n_values.split(',')]
    seeds = [int(x) for x in args.seeds.split(',')]
    eval_kwargs = dict(n_epochs=args.n_epochs, n_steps=args.n_steps, agg=args.agg)

    default_params = {
        'exp3': {'gamma': 0.001, 'min_reward': -1.0, 'max_reward': 1.0},
        'ts': {'alpha': 0.75, 'beta': 0.06, 'lam': 0.004, 'mu': -0.2, 'gamma': 0.96},
        'softmax': {'lr': 0.02, 'alpha': 0.93, 'tau': 9.9},
    }
    print(f'Tuning {args.mab_type} | n_values={n_values} | seeds={seeds} | agg={args.agg}')
    f, t, per_n = evaluate(args.mab_type, default_params[args.mab_type], n_values, seeds, **eval_kwargs)
    print(f'  Default -> fairness={f:.4f}, throughput={t:.4f}  per_n={ {k: tuple(round(v, 3) for v in val) for k, val in per_n.items()} }')

    print(f'\nRunning Optuna multi-objective search ({args.n_trials} trials)...')
    study = optuna.create_study(
        directions=['maximize', 'maximize'],
        sampler=optuna.samplers.NSGAIISampler(seed=0),
    )
    study.optimize(
        make_objective(args.mab_type, n_values, seeds, **eval_kwargs),
        n_trials=args.n_trials, n_jobs=args.n_jobs,
    )

    front = sorted(study.best_trials, key=lambda tr: tr.values[1], reverse=True)
    print(f'\nPareto front ({len(front)} solutions):')
    print(f"  {'fairness':>9} {'throughput':>11}   params")
    for tr in front:
        print(f'  {tr.values[0]:>9.4f} {tr.values[1]:>11.4f}   {tr.params}')

    balanced = pick_balanced(front)
    print(f'\nBalanced (knee) pick -> fairness={balanced.values[0]:.4f}, throughput={balanced.values[1]:.4f}')
    print(f'Params: {balanced.params}')

    # Held-out verification on an unseen seed, across all network sizes.
    holdout_seed = max(seeds) + 1000
    best_params = default_params[args.mab_type].copy()
    best_params.update(balanced.params)
    vf, vt, vper_n = evaluate(args.mab_type, best_params, n_values, [holdout_seed], **eval_kwargs)
    print(f'\nHeld-out (seed={holdout_seed}) -> fairness={vf:.4f}, throughput={vt:.4f}')
    print(f'  per_n={ {k: tuple(round(v, 3) for v in val) for k, val in vper_n.items()} }')

    out_path = args.out or f'tune_mab_{args.mab_type}_pareto.json'
    payload = {
        'mab_type': args.mab_type,
        'n_values': n_values,
        'seeds': seeds,
        'agg': args.agg,
        'pareto_front': [
            {
                'fairness': tr.values[0],
                'throughput': tr.values[1],
                'params': tr.params,
                'per_n': {
                    n: [tr.user_attrs.get(f'fairness_n{n}'), tr.user_attrs.get(f'throughput_n{n}')]
                    for n in n_values
                },
            }
            for tr in front
        ],
        'balanced': {'fairness': balanced.values[0], 'throughput': balanced.values[1], 'params': balanced.params},
        'holdout': {'seed': holdout_seed, 'fairness': vf, 'throughput': vt},
    }
    with open(out_path, 'w') as fp:
        json.dump(payload, fp, indent=2)
    print(f'\nSaved Pareto front to {out_path}')
