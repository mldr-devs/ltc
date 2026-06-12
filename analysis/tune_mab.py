import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import argparse

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
    cutoff = int(0.7 * n_epochs)
    ch = np.array(all_outputs.channel_state[cutoff:])

    channel_success = float(np.mean(ch == 1))
    actions = np.array(all_outputs.actions[cutoff:])
    tx_mask = (actions == Actions.TX.value)
    ch_broadcast = (ch == 1)[..., None]
    succ_tx = (tx_mask & ch_broadcast).sum(axis=(0, 1)).astype(float)

    n = succ_tx.shape[0]
    if succ_tx.sum() == 0:
        fairness = 0.0
    else:
        fairness = float((succ_tx.sum() ** 2) / (n * (succ_tx ** 2).sum()))

    return fairness, channel_success


def evaluate(mab_type, params, n_epochs=200, n_steps=50, seed=42):
    n = 10
    num_actions = 2
    if mab_type == 'exp3':
        mab = Exp3(n_arms=num_actions, **params)
    elif mab_type == 'ts':
        mab = DiscountedThompsonSampling(n_arms=num_actions, **params)
    elif mab_type == 'softmax':
        mab = Softmax(n_arms=num_actions, **params)

    try:
        all_outputs = run_sim(mab, n=n, n_epochs=n_epochs, n_steps=n_steps, seed=seed)
        return compute_metrics(all_outputs, n_epochs, n_steps)
    except Exception as e:
        return 0.0, 0.0


def make_objective(mab_type, n_epochs=200, n_steps=50):
    def objective(trial):
        if mab_type == 'exp3':
            params = {
                'gamma': trial.suggest_float('gamma', 0.001, 0.5, log=True),
                'min_reward': -1.,
                'max_reward': 1.,
            }
        elif mab_type == 'ts':
            params = {
                'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
                'beta': trial.suggest_float('beta', 1e-3, 10.0, log=True),
                'lam': trial.suggest_float('lam', 1e-3, 2.0, log=True),
                'mu': trial.suggest_float('mu', -1.0, 1.0),
                'gamma': trial.suggest_float('gamma', 0.5, 0.9999, log=True),
            }
        elif mab_type == 'softmax':
            params = {
                'lr': trial.suggest_float('lr', 0.01, 10.0, log=True),
                'alpha': trial.suggest_float('alpha', 0., 1.0),
                'tau': trial.suggest_float('tau', 0.1, 10.0, log=True),
            }

        fairness, channel_success = evaluate(mab_type, params, n_epochs=n_epochs, n_steps=n_steps)

        if fairness < 0.7:
            return -1. + fairness
        elif channel_success < 0.30:
            return -1. + channel_success
        else:
            return fairness + channel_success

    return objective


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mab_type', required=True, choices=['exp3', 'ts', 'softmax'])
    parser.add_argument('--n_trials', type=int, default=100)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--n_steps', type=int, default=50)
    args = parser.parse_args()

    default_params = {
        'exp3': {'gamma': 0.05, 'min_reward': -1.0, 'max_reward': 1.0},
        'ts': {'alpha': 1., 'beta': 1., 'lam': 1., 'mu': 0., 'gamma': 0.9},
        'softmax': {'lr': 0.01, 'alpha': 0.1, 'tau': 1.0},
    }
    print(f"Evaluating default params for {args.mab_type}...")
    f, cs = evaluate(args.mab_type, default_params[args.mab_type], args.n_epochs, args.n_steps)
    print(f"  Default -> fairness={f:.4f}, channel_success={cs:.4f}")

    print(f"\nRunning Optuna search ({args.n_trials} trials)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(make_objective(args.mab_type, args.n_epochs, args.n_steps), n_trials=args.n_trials, n_jobs=8, show_progress_bar=True)

    best = study.best_trial
    print(f"\nBest trial: value={best.value:.4f}")
    print(f"Params: {best.params}")

    best_params = default_params[args.mab_type].copy()
    best_params.update(best.params)
    f, cs = evaluate(args.mab_type, best_params, args.n_epochs, args.n_steps)
    print(f"Verification -> fairness={f:.4f}, channel_success={cs:.4f}")
