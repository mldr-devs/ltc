import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'

import argparse
from dataclasses import replace
from functools import partial

import cloudpickle
import jax
import jax.numpy as jnp
import lz4.frame
import optax
from jax_tqdm import scan_tqdm

from ltc.agents import DDQN, DCF, QNetwork
from ltc.sim import InitialStateConf, cox_traffic, process_output, simulate
from ltc.sim.constants import INITIAL_CAPACITY, Actions
from ltc.utils.scan_states import Carry, Output
from ltc.utils.plots import plot_all, plot_first


def init_agents(agent, key, n):
    keys = jax.random.split(key, n)
    states = jax.vmap(agent.init)(keys)
    step_fn = jax.vmap(partial(agent_step, agent))
    return states, step_fn


def agent_step(agent, state, key, obs, action, reward, terminal, wait):
    update_key, sample_key = jax.random.split(key)

    def power_on(state, update_key, sample_key, obs, action, reward, terminal, wait):
        def update(state, update_key, sample_key, obs, action, reward, terminal):
            state = agent.update(state, update_key, obs, action, reward, terminal)
            action = agent.sample(state, sample_key, obs)
            return state, action

        return jax.lax.cond(
            wait, lambda state, *_: (state, Actions.CS.value), update,
            state, update_key, sample_key, obs, action, reward, terminal
        )

    def power_off(state, update_key, sample_key, obs, action, reward, terminal, wait):
        return state, Actions.IDLE.value

    return jax.lax.cond(
        terminal, power_off, power_on,
        state, update_key, sample_key, obs, action, reward, terminal, wait
    )


def init_traffic(traffic, key, n):
    keys = jax.random.split(key, n)
    states = jax.vmap(traffic.init)(keys)
    step_fn = jax.vmap(traffic.sample)
    return states, step_fn


def rl_step(drl_step, legacy_step, traffic_step, n, n_drl, n_bins=50):
    def rl_step_fn(c, _):
        key, drl_keys, legacy_keys, traffic_key = jax.random.split(c.key, 4)
        drl_keys = jax.random.split(drl_keys, n_drl)
        legacy_keys = jax.random.split(legacy_keys, n - n_drl)
        traffic_keys = jax.random.split(traffic_key, n)

        drl_states, drl_actions = drl_step(
            c.drl_states, drl_keys, c.obs[:n_drl], c.actions[:n_drl], c.rewards[:n_drl],
            c.terminals[:n_drl], (c.actions[:n_drl] == Actions.TX.value) | (c.channel_state != 0)
        )
        legacy_states, legacy_actions = legacy_step(
            c.legacy_states, legacy_keys, c.obs[n_drl:], c.actions[n_drl:], c.rewards[n_drl:],
            c.terminals[n_drl:], jnp.zeros(n - n_drl, dtype=bool)
        )
        actions = jnp.concatenate([drl_actions, legacy_actions])

        traffic_states, new_frames = traffic_step(c.traffic_states, traffic_keys)
        buffer_states, channel_state, d2lt = simulate(c.buffer_states, new_frames, actions, c.d2lt)
        obs, rewards, powers = process_output(c.buffer_states, buffer_states, c.power_states, c.d2lt, channel_state, c.obs, actions, c.terminals)
        terminals = jnp.logical_or(c.terminals, powers < 0)

        global_obs = jnp.concatenate([actions, d2lt / d2lt.sum()])

        if n_drl > 0:
            params = c.drl_states.params['model'] if 'model' in c.drl_states.params else c.drl_states.params
            flat_params, _ = jax.tree.flatten(params)
            flat_params = jax.tree.map(lambda x: x.reshape(n_drl, -1), flat_params)
            flat_params = jnp.hstack(flat_params)
            hist, bin_edges = jax.vmap(jnp.histogram, in_axes=(0, None))(flat_params, n_bins)
        else:
            hist, bin_edges = None, None

        c = Carry(
            drl_states, legacy_states, traffic_states, buffer_states, powers, d2lt,
            channel_state, key, obs, actions, rewards, terminals
        )
        o = Output(
            legacy_states, obs, actions, rewards, terminals, buffer_states, powers,
            (new_frames > 0).astype(int), channel_state, hist, bin_edges
        )
        return c, o

    return rl_step_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the RL network simulation with configurable parameters.")
    parser.add_argument('--n', type=int, default=10, help='Total number of agents in the simulation.')
    parser.add_argument('--n_drl', type=int, default=5, help='Number of DRL agents.')
    parser.add_argument('--n_epochs', type=int, default=40, help='Number of training epochs to run.')
    parser.add_argument('--n_steps', type=int, default=2000, help='Number of steps per epoch.')
    parser.add_argument('--window_size', type=int, default=20, help='Size of the observation window for each agent.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--save-plots', action='store_true', default=False, help='Whether to save the generated plots.')
    parser.add_argument('--traffic_type', type=str, default='saturated', choices=['constant', 'saturated', 'bursty'],help="Traffic model to use: 'constant', 'saturated', or 'bursty'.")
    args = parser.parse_args()

    n = args.n
    n_drl = args.n_drl
    n_epochs = args.n_epochs
    n_steps = args.n_steps
    window_size = args.window_size
    seed = args.seed
    traffic_type = args.traffic_type

    key = jax.random.key(seed)
    num_actions = 2
    actions = jnp.zeros(n, dtype=int)
    buffer_states = jnp.zeros(n, dtype=int)
    power_states = jnp.full(n, INITIAL_CAPACITY, dtype=int)
    channel_state = 0
    obs = jnp.zeros((n, window_size, 7), dtype=int)
    rewards = jnp.zeros(n)
    terminals = jnp.full(n, False, dtype=bool)
    d2lt = jnp.zeros(n, dtype=int)

    drl = DDQN(
        q_network=QNetwork(num_actions, num_layers=2, dim=32, num_heads=2),
        obs_space_shape=obs.shape[1:],
        act_space_size=num_actions,
        optimizer=optax.adam(1e-4),
        experience_replay_buffer_size=10000,
        experience_replay_batch_size=128,
        experience_replay_steps=5,
        discount=1.0,
        epsilon=1.0,
        epsilon_decay=0.999,
        epsilon_min=0.001,
        tau=0.01
    )
    key, init_key = jax.random.split(key)
    drl_states, drl_step = init_agents(drl, init_key, n_drl)
    drl_states = drl_states.replace(prev_env_state=drl_states.prev_env_state.astype(int))

    dcf = DCF()
    key, init_key = jax.random.split(key)
    legacy_states, legacy_step = init_agents(dcf, init_key, n - n_drl)

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

    rl_step_fn = jax.jit(rl_step(drl_step, legacy_step, traffic_step, n, n_drl))
    rl_step_fn = scan_tqdm(n_steps)(rl_step_fn)
    init_carry = Carry(
        drl_states, legacy_states, traffic_states, buffer_states, power_states, d2lt,
        channel_state, key, obs, actions, rewards, terminals
    )
    all_outputs = []

    for epoch in range(n_epochs):
        print(f'Epoch {epoch + 1}/{n_epochs}')
        carry, output = jax.lax.scan(rl_step_fn, init_carry, jnp.arange(n_steps))
        init_carry = replace(
            init_carry,
            drl_states=carry.drl_states,
            key=carry.key
        )
        all_outputs.append(output)

    all_outputs = jax.tree.map(lambda *x: jnp.stack(x), *all_outputs)
    filename = f'history_{n}_{n_drl}_{seed}.pkl.lz4'

    with lz4.frame.open(filename, 'wb') as f:
        cloudpickle.dump((init_carry.drl_states, all_outputs), f)

    if args.save_plots:
        plot_all(filename)
        plot_first(filename)
