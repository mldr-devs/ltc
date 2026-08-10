import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'

import argparse
from dataclasses import dataclass, replace
from functools import partial

import cloudpickle
import jax
import jax.numpy as jnp
import lz4.frame
import optax
from reinforced_lib.agents import AgentState
from tqdm import trange

from ltc.baselines.dcf import DCF
from ltc.baselines.qlbt import QLBT, QLBTNetwork
from ltc.sim import InitialStateConf, ModelState, cox_traffic, simulate, transmission_outcome
from ltc.sim.constants import (
    COLLISION_PENALTY, CS_CONSUMPTION, IDLE_CONSUMPTION, INITIAL_CAPACITY, NO_TX_REWARD,
    TX_CONSUMPTION, TX_REWARD, Actions
)
from ltc.utils.history import build_history_filename, ensure_clean_git_worktree, get_short_commit_hash

"""
QLBT observes its own feature set, unrelated to the one ltc.run builds:
[action, channel, bias, d2lt_i, d2lt_mi, d2lt, ret_c, buffer]
"""
ACTION, CHANNEL, BIAS, D2LT_I, D2LT_MI, D2LT, RET_C, BUFFER = range(8)
N_FEATURES = 8


class QLBTDCF(DCF):
    FEATURES = (BUFFER, CHANNEL, RET_C)


@jax.tree_util.register_dataclass
@dataclass
class Carry:
    drl_states: AgentState
    legacy_states: AgentState
    traffic_states: ModelState
    buffer_states: jax.Array
    power_states: jax.Array
    d2lt: jax.Array
    throughputs: jax.Array
    channel_state: int
    key: jax.random.PRNGKey
    obs: jax.Array
    actions: jax.Array
    rewards: jax.Array
    terminals: jax.Array


@jax.tree_util.register_dataclass
@dataclass
class Output:
    legacy_states: AgentState
    observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    terminals: jax.Array
    buffer_states: jax.Array
    power_states: jax.Array
    new_frames: jax.Array
    channel_state: int
    d2lt: jax.Array


def update_d2lt(d2lt, actions, channel_state):
    return jnp.where((channel_state == 1) & (actions == Actions.TX.value), 0, d2lt + 1)


def process_output_i(power_state, new_buffer_state, d2lt, idx, channel_state, obs, action):
    *_, ret_c, _ = obs[-1]
    ret_c = jnp.where(action == Actions.TX.value, jnp.where(channel_state == 1, 0, ret_c + 1), ret_c)

    d2lt_i = d2lt[idx]
    d2lt_mi = jnp.min(jnp.where(jnp.arange(d2lt.shape[0]) == idx, jnp.iinfo(d2lt.dtype).max, d2lt))
    denominator = jnp.maximum(d2lt_i + d2lt_mi, 1)
    d2lt_i, d2lt_mi = d2lt_i / denominator, d2lt_mi / denominator

    channel_state = jnp.where(action == Actions.TX.value, channel_state == 1, jnp.abs(channel_state))
    power = jnp.where(
        action == Actions.TX.value, power_state - TX_CONSUMPTION,
        jnp.where(
            action == Actions.CS.value, power_state - CS_CONSUMPTION,
            jnp.where(
                action == Actions.IDLE.value, power_state - IDLE_CONSUMPTION,
                power_state
            )
        )
    )

    obs_t = jnp.array([action, channel_state, 1, d2lt_i, d2lt_mi, d2lt[idx], ret_c, new_buffer_state])
    obs = jnp.roll(obs, -1, axis=0)
    obs = obs.at[-1].set(obs_t)

    return obs, power


def process_output(key, new_buffer_states, power_states, d2lt, throughputs, channel_state, obs, actions, terminals):
    tx_action = (actions == Actions.TX.value).astype(int)
    thr_t = (tx_action & (channel_state == 1)).astype(int)
    throughputs = jnp.roll(throughputs, -1, axis=1)
    throughputs = throughputs.at[:, -1].set(thr_t)

    priority = new_buffer_states / (throughputs.mean(axis=1) + 1e-6)
    opt_action = (priority == priority.max()).astype(int)
    probs = opt_action / opt_action.sum()
    opt_action = jax.random.choice(key, actions.shape[0], p=probs)
    opt_action = jnp.zeros_like(actions).at[opt_action].set(1)
    rewards_ind = 2 * (tx_action == opt_action).astype(float) - 1
    rewards_ind = jnp.where(terminals, 0., rewards_ind)

    reward_tot = jnp.where(
        channel_state == -1, COLLISION_PENALTY,
        jnp.where(
            channel_state == 0, NO_TX_REWARD,
            jnp.where(
                (tx_action & (d2lt == d2lt.max())).sum() > 0, TX_REWARD,
                d2lt[jnp.argmax(tx_action)] / jnp.maximum(d2lt.sum(), 1)
            )
        )
    )
    rewards = jnp.concatenate([jnp.array([reward_tot]), rewards_ind])

    idxs = jnp.arange(new_buffer_states.shape[0])
    obs, powers = jax.vmap(process_output_i, in_axes=(0, 0, None, 0, None, 0, 0))(
        power_states, new_buffer_states, d2lt, idxs, channel_state, obs, actions
    )

    return obs, rewards, throughputs, powers


def init_legacy_agents(agent, key, n):
    keys = jax.random.split(key, n)
    states = jax.vmap(agent.init)(keys)
    step_fn = jax.vmap(partial(legacy_step_fn, agent))
    return states, step_fn


def legacy_step_fn(agent, state, key, obs, action, reward, terminal, outcome):
    update_key, sample_key = jax.random.split(key)
    state = agent.update(state, update_key, obs, action, reward, terminal, outcome=outcome)
    return state, agent.sample(state, sample_key, obs)


def drl_step_fn(agent, state, key, obs, action, rewards, terminal, wait):
    update_key, sample_key = jax.random.split(key)
    state = agent.update(state, update_key, obs, action, rewards, terminal)
    action = jax.lax.cond(
        wait,
        lambda *_: jnp.full_like(action, Actions.CS.value),
        lambda state, key, obs: agent.sample(state, key, obs),
        state, sample_key, obs
    )
    return state, action


def rl_step(drl_step, legacy_step, traffic_step, n, n_drl, phy_error_prob):
    def rl_step_fn(c, _):
        key, drl_key, legacy_keys, traffic_key, process_key, sim_key = jax.random.split(c.key, 6)
        legacy_keys = jax.random.split(legacy_keys, n - n_drl)
        traffic_keys = jax.random.split(traffic_key, n)

        drl_states, drl_actions = drl_step(
            c.drl_states, drl_key, c.obs[:n_drl], c.actions[:n_drl], c.rewards[:n_drl + 1],
            c.terminals[:n_drl], c.channel_state != 0
        )
        legacy_states, legacy_actions = legacy_step(
            c.legacy_states, legacy_keys, c.obs[n_drl:], c.actions[n_drl:], c.rewards[n_drl + 1:],
            c.terminals[n_drl:], transmission_outcome(c.actions, c.channel_state)[n_drl:]
        )
        actions = jnp.concatenate([drl_actions, legacy_actions])

        traffic_states, new_frames = traffic_step(c.traffic_states, traffic_keys)
        buffer_states, channel_state, _ = simulate(
            c.buffer_states, new_frames, actions, sim_key, error_probability=phy_error_prob
        )
        d2lt = update_d2lt(c.d2lt, actions, channel_state)
        obs, rewards, throughputs, powers = process_output(
            process_key, buffer_states, c.power_states, c.d2lt, c.throughputs, channel_state, c.obs,
            actions, c.terminals
        )
        terminals = jnp.logical_or(c.terminals, powers < 0)

        c = Carry(
            drl_states, legacy_states, traffic_states, buffer_states, powers, d2lt, throughputs,
            channel_state, key, obs, actions, rewards, terminals
        )
        o = Output(
            legacy_states, obs, actions, rewards, terminals, buffer_states, powers,
            (new_frames > 0).astype(int), channel_state, d2lt
        )
        return c, o

    return rl_step_fn


def setup_args():
    parser = argparse.ArgumentParser(description='Run the QLBT baseline.')
    parser.add_argument('--n', type=int, default=5, help='Total number of stations.')
    parser.add_argument('--n_drl', type=int, help='Number of QLBT stations. Defaults to all of them.')
    parser.add_argument('--n_epochs', type=int, default=1, help='Number of training epochs to run.')
    parser.add_argument('--n_steps', type=int, default=20000, help='Number of steps per epoch.')
    parser.add_argument('--window_size', type=int, default=10, help='Size of the observation window.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--throughput_window', type=int, default=1000, help='Window used for the throughput fairness term.')
    parser.add_argument('--traffic_type', type=str, default='saturated', choices=['constant', 'saturated', 'bursty', 'custom'], help='Traffic model to use.')
    parser.add_argument('--loc', type=float, default=5.0, help='loc traffic generator parameter.')
    parser.add_argument('--scale', type=float, default=0.0, help='scale traffic generator parameter.')
    parser.add_argument('--f3dB', type=float, default=1.0, help='f3dB traffic generator parameter.')
    parser.add_argument('--phy_error_prob', type=float, default=0.05, help='Probability of error in phy channel.')
    parser.add_argument('--skip_git_check', action='store_true', default=False, help='Skip clean git worktree check.')
    return parser.parse_args()


if __name__ == '__main__':
    args = setup_args()
    if not args.skip_git_check:
        ensure_clean_git_worktree()
    commit_hash = get_short_commit_hash()

    n = args.n
    n_drl = args.n_drl if args.n_drl is not None else n
    if not 0 < n_drl <= n:
        raise ValueError(f'--n_drl must be between 1 and {n}.')

    window_size = args.window_size
    seed = args.seed
    num_actions = 2

    key = jax.random.key(seed)
    actions = jnp.zeros(n, dtype=int)
    buffer_states = jnp.zeros(n, dtype=int)
    power_states = jnp.full(n, INITIAL_CAPACITY, dtype=int)
    channel_state = 0
    obs = jnp.zeros((n, window_size, N_FEATURES), dtype=float)
    rewards = jnp.zeros(n + 1)
    terminals = jnp.full(n, False, dtype=bool)
    d2lt = jnp.zeros(n, dtype=int)
    throughputs = jnp.zeros((n, args.throughput_window), dtype=int)

    drl = QLBT(
        q_network=QLBTNetwork(num_actions),
        obs_space_shape=(n_drl, window_size, N_FEATURES + 1),
        act_space_size=num_actions,
        optimizer=optax.rmsprop(5e-4),
        experience_replay_buffer_size=500,
        experience_replay_batch_size=32,
        experience_replay_steps=1,
        discount=0.5,
        epsilon=1.0,
        epsilon_decay=0.998,
        epsilon_min=0.01,
        tau=0.02
    )
    key, init_key = jax.random.split(key)
    drl_states = drl.init(init_key)
    drl_step = partial(drl_step_fn, drl)

    key, init_key = jax.random.split(key)
    legacy_states, legacy_step = init_legacy_agents(QLBTDCF(), init_key, n - n_drl)

    key, init_key = jax.random.split(key)

    if args.traffic_type == 'constant':
        traffic = cox_traffic(f3dB=1.0, loc=-1.0, scale=0.0, initial_state=InitialStateConf.ZERO)
    elif args.traffic_type == 'saturated':
        traffic = cox_traffic(f3dB=1.0, loc=5.0, scale=0.0, initial_state=InitialStateConf.ZERO)
    elif args.traffic_type == 'bursty':
        traffic = cox_traffic(f3dB=0.1, loc=-5.0, scale=5.0, initial_state=InitialStateConf.ZERO)
    elif args.traffic_type == 'custom':
        traffic = cox_traffic(f3dB=args.f3dB, loc=args.loc, scale=args.scale, initial_state=InitialStateConf.ZERO)
    else:
        raise ValueError(f'Unknown traffic type: {args.traffic_type}')

    traffic_states = jax.vmap(traffic.init)(jax.random.split(init_key, n))
    traffic_step = jax.vmap(traffic.sample)

    rl_step_fn = jax.jit(rl_step(drl_step, legacy_step, traffic_step, n, n_drl, args.phy_error_prob))
    carry = Carry(
        drl_states, legacy_states, traffic_states, buffer_states, power_states, d2lt, throughputs,
        channel_state, key, obs, actions, rewards, terminals
    )
    all_outputs = []

    for _ in trange(args.n_epochs):
        carry, output = jax.lax.scan(rl_step_fn, carry, length=args.n_steps)
        all_outputs.append(output)

    all_outputs = jax.tree.map(lambda *x: jnp.stack(x), *all_outputs)
    filename = build_history_filename(n, n_drl, seed, commit_hash)

    with lz4.frame.open(filename, 'wb') as f:
        cloudpickle.dump((carry.drl_states, all_outputs, vars(args)), f)
