import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

from dataclasses import replace
from functools import partial

import argparse
import cloudpickle
import jax
import jax.numpy as jnp
import lz4.frame
import optax
from jax_tqdm import scan_tqdm

from ltc.agents import BayesianDDQN, DDQN, DLMANetwork, QNetwork, QNetworkDropout, StochasticVariationalNetwork, DCF, QALOHA, EBALOHA, FWALOHA, TDMA
from ltc.sim import InitialStateConf, cox_traffic, process_output, simulate
from ltc.sim.constants import INITIAL_CAPACITY, Actions
from ltc.utils.scan_states import Carry, Output
from ltc.utils.plots import plot_all, plot_first


def init_agents(agent, key, n):
    keys = jax.random.split(key, n)
    states = jax.vmap(agent.init)(keys)
    step_fn = jax.vmap(partial(agent_step, agent))
    return states, step_fn


def agent_step(agent, state, key, obs, action, reward, terminal, active):
    update_key, sample_key = jax.random.split(key)

    def power_on(state, update_key, sample_key, obs, action, reward, terminal):
        state = agent.update(state, update_key, obs, action, reward, terminal)
        action = agent.sample(state, sample_key, obs)
        return state, action

    def power_off(state, update_key, sample_key, obs, action, reward, terminal):
        return state, Actions.IDLE.value

    return jax.lax.cond(
        jnp.logical_or(~active, terminal), power_off, power_on,
        state, update_key, sample_key, obs, action, reward, terminal
    )


def init_traffic(traffic, key, n):
    keys = jax.random.split(key, n)
    states = jax.vmap(traffic.init)(keys)
    step_fn = jax.vmap(traffic.sample)
    return states, step_fn


def rl_step(drl_step, legacy_step, traffic_step, n, n_drl, n_bins=50, n_switch=None, n_final=None):
    def rl_step_fn(c, step):
        if n_switch is not None:
            active = jnp.where(step == n_switch, jnp.ones_like(c.active).at[n_final:].set(False), c.active)
        else:
            active = c.active
        
        key, drl_keys, legacy_keys, traffic_key = jax.random.split(c.key, 4)
        drl_keys = jax.random.split(drl_keys, n_drl)
        legacy_keys = jax.random.split(legacy_keys, n - n_drl)
        traffic_keys = jax.random.split(traffic_key, n)

        drl_states, drl_actions = drl_step(
            c.drl_states, drl_keys, c.obs[:n_drl], c.actions[:n_drl], c.rewards[:n_drl], c.terminals[:n_drl], active[:n_drl]
        )
        legacy_states, legacy_actions = legacy_step(
            c.legacy_states, legacy_keys, c.obs[n_drl:], c.actions[n_drl:], c.rewards[n_drl:], c.terminals[n_drl:], active[n_drl:]
        )
        actions = jnp.concatenate([drl_actions, legacy_actions])

        traffic_states, new_frames = traffic_step(c.traffic_states, traffic_keys)
        buffer_states, channel_state = simulate(c.buffer_states, new_frames, actions)
        obs, rewards, powers = process_output(c.buffer_states, buffer_states, c.power_states, channel_state, c.obs, actions, c.terminals)
        terminals = jnp.logical_or(c.terminals, powers < 0)

        if n_drl > 0:
            params = c.drl_states.params['model'] if 'model' in c.drl_states.params else c.drl_states.params
            flat_params, _ = jax.tree.flatten(params)
            flat_params = jax.tree.map(lambda x: x.reshape(n_drl, -1), flat_params)
            flat_params = jnp.hstack(flat_params)
            hist, bin_edges = jax.vmap(jnp.histogram, in_axes=(0, None))(flat_params, n_bins)
        else:
            hist, bin_edges = None, None

        c = Carry(
            drl_states, legacy_states, traffic_states, buffer_states, powers,
            channel_state, key, obs, actions, rewards, terminals, active
        )
        o = Output(
            legacy_states, obs, actions, rewards, terminals, buffer_states, powers,
            (new_frames > 0).astype(int), channel_state, active, hist, bin_edges
        )
        return c, o

    return rl_step_fn


def setup_args():
    parser = argparse.ArgumentParser(description="Run the RL network simulation with configurable parameters.")
    parser.add_argument('--n', type=int, default=5, help='Initial number of agents in the simulation.')
    parser.add_argument('--n_drl', type=int, default=5, help='Number of DRL agents in the simulation.')
    parser.add_argument('--n_final', type=int, help='Final number of agents in the simulation.')
    parser.add_argument('--n_steps', type=int, default=2000, help='Number of steps.')
    parser.add_argument('--window_size', type=int, default=1, help='Size of the observation window for each agent.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--save_plots', action='store_true', default=False, help='Whether to save the generated plots.')
    parser.add_argument('--traffic_type', type=str, default='saturated', choices=['constant', 'saturated', 'bursty'],help="Traffic model to use: 'constant', 'saturated', or 'bursty'.")
    parser.add_argument('--legacy_type', type=str, default='tdma', choices=['q-aloha', 'eb-aloha', 'fw-aloha', 'tdma'], help="Legacy agent type to use: 'q-aloha', 'eb-aloha', 'fw-aloha', or 'tdma'.")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = setup_args()

    n_drl = args.n_drl
    n_init = args.n
    n_final = args.n_final if args.n_final is not None else n_init
    n = max(n_init, n_final)
    n_steps = args.n_steps
    window_size = args.window_size
    seed = args.seed
    traffic_type = args.traffic_type
    legacy_type = args.legacy_type

    key = jax.random.key(seed)
    actions = jnp.zeros(n, dtype=int)
    buffer_states = jnp.zeros(n, dtype=int)
    power_states = jnp.full(n, INITIAL_CAPACITY, dtype=int)
    channel_state = 0
    obs = jnp.zeros((n, window_size, 3), dtype=int)
    rewards = jnp.zeros(n)
    terminals = jnp.full(n, False, dtype=bool)
    active = jnp.ones(n, dtype=bool).at[n_init:].set(False)

    drl = DDQN(
        q_network=DLMANetwork(),
        obs_space_shape=obs.shape[1:],
        act_space_size=2,
        optimizer=optax.rmsprop(1e-2),
        experience_replay_buffer_size=500,
        experience_replay_batch_size=32,
        experience_replay_steps=1,
        discount=0.9,
        epsilon=0.1,
        epsilon_decay=0.9999,
        epsilon_min=0.005,
        tau=0.02
    )
    key, init_key = jax.random.split(key)
    drl_states, drl_step = init_agents(drl, init_key, n_drl)
    drl_states = drl_states.replace(prev_env_state=drl_states.prev_env_state.astype(int))

    key, init_key = jax.random.split(key)

    if legacy_type == 'q-aloha':
        legacy = QALOHA(q=0.5)
    elif legacy_type == 'eb-aloha':
        legacy = EBALOHA(window_size=4, max_backoff=2)
    elif legacy_type == 'fw-aloha':
        legacy = FWALOHA(window_size=4)
    elif legacy_type == 'tdma':
        legacy = TDMA(state_size=10, assigned_slots=5)
    else:
        raise ValueError(f'Unknown legacy type: {legacy_type}')

    legacy_states, legacy_step = init_agents(legacy, init_key, n - n_drl)

    key, init_key = jax.random.split(key)

    if traffic_type == 'constant':
        traffic = cox_traffic(f3dB=1.0, loc=-1.0, scale=0.0, initial_state=InitialStateConf.ZERO)
    elif traffic_type == 'saturated':
        traffic = cox_traffic(f3dB=1.0, loc=5.0, scale=0.0, initial_state=InitialStateConf.ZERO)
    elif traffic_type == 'bursty':
        traffic = cox_traffic(f3dB=0.1, loc=-5.0, scale=5.0, initial_state=InitialStateConf.ZERO)
    else:
        raise ValueError(f'Unknown traffic type: {traffic_type}')

    traffic_states, traffic_step = init_traffic(traffic, init_key, n)

    rl_step_fn = jax.jit(rl_step(drl_step, legacy_step, traffic_step, n, n_drl, n_switch=n_steps // 2, n_final=n_final))
    rl_step_fn = scan_tqdm(n_steps)(rl_step_fn)
    init_carry = Carry(
        drl_states, legacy_states, traffic_states, buffer_states, power_states,
        channel_state, key, obs, actions, rewards, terminals, active
    )
    all_outputs = []

    carry, output = jax.lax.scan(rl_step_fn, init_carry, jnp.arange(n_steps))
    all_outputs.append(output)

    all_outputs = jax.tree.map(lambda *x: jnp.stack(x), *all_outputs)
    filename = f'history_{n_init}_{n_final}_{seed}.pkl.lz4'

    with lz4.frame.open(filename, 'wb') as f:
        cloudpickle.dump((init_carry.drl_states, all_outputs), f)

    if args.save_plots:
        plot_all(filename)
        plot_first(filename)
