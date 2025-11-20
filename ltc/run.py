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

from ltc.agents import DCF, QLBT, QLBTNetwork
from ltc.sim import InitialStateConf, cox_traffic, process_output, simulate
from ltc.sim.constants import INITIAL_CAPACITY, Actions
from ltc.utils.scan_states import Carry, Output
from ltc.utils.plots import plot_first


def init_agents(agent, key, n, apply_vmap):
    map_fn = jax.vmap if apply_vmap else lambda f: f
    keys = jax.random.split(key, n) if apply_vmap else key
    states = map_fn(agent.init)(keys)
    step_fn = map_fn(partial(agent_step, agent))
    return states, step_fn


def agent_step(agent, state, key, obs, action, reward, terminal, wait):
    def agent_fn(state, key, obs, action, reward, terminal):
        update_key, sample_key = jax.random.split(key)
        state = agent.update(state, update_key, obs, action, reward, terminal)
        action = agent.sample(state, sample_key, obs)
        return state, action
    
    def wait_fn(state, key, obs, action, reward, terminal):
        return state, jnp.full_like(action, Actions.CS.value)

    return jax.lax.cond(jnp.any(wait), wait_fn, agent_fn, state, key, obs, action, reward, terminal)


def init_traffic(traffic, key, n):
    keys = jax.random.split(key, n)
    states = jax.vmap(traffic.init)(keys)
    step_fn = jax.vmap(traffic.sample)
    return states, step_fn


def rl_step(drl_step, legacy_step, traffic_step, n, n_drl):
    def rl_step_fn(c, _):
        key, drl_keys, legacy_keys, traffic_key = jax.random.split(c.key, 4)
        legacy_keys = jax.random.split(legacy_keys, n - n_drl)
        traffic_keys = jax.random.split(traffic_key, n)

        drl_states, drl_actions = drl_step(
            c.drl_states, drl_keys, c.obs[:n_drl], c.actions[:n_drl], c.rewards[:n_drl + 1],
            c.terminals[:n_drl], False
        )
        legacy_states, legacy_actions = legacy_step(
            c.legacy_states, legacy_keys, c.obs[n_drl:], c.actions[n_drl:], c.rewards[n_drl + 1:],
            c.terminals[n_drl:], jnp.zeros(n - n_drl, dtype=bool)
        )
        actions = jnp.concatenate([drl_actions, legacy_actions])

        traffic_states, new_frames = traffic_step(c.traffic_states, traffic_keys)
        buffer_states, channel_state, d2lt = simulate(c.buffer_states, new_frames, actions, c.d2lt)
        obs, rewards, throughputs, powers = process_output(
            c.buffer_states, buffer_states, c.power_states, c.d2lt, c.throughputs, channel_state, c.obs, actions, c.terminals
        )
        terminals = jnp.logical_or(c.terminals, powers < 0)

        c = Carry(
            drl_states, legacy_states, traffic_states, buffer_states, powers, d2lt, throughputs,
            channel_state, key, obs, actions, rewards, terminals
        )
        o = Output(
            legacy_states, obs, actions, rewards, terminals, buffer_states, powers,
            (new_frames > 0).astype(int), channel_state, None, None
        )
        return c, o

    return rl_step_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the RL network simulation with configurable parameters.")
    parser.add_argument('--n', type=int, default=5, help='Total number of agents in the simulation.')
    parser.add_argument('--n_drl', type=int, default=5, help='Number of DRL agents.')
    parser.add_argument('--n_epochs', type=int, default=1, help='Number of training epochs to run.')
    parser.add_argument('--n_steps', type=int, default=20000, help='Number of steps per epoch.')
    parser.add_argument('--window_size', type=int, default=10, help='Size of the observation window for each agent.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--save-plots', action='store_false', default=True, help='Whether to save the generated plots.')
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
    obs = jnp.zeros((n, window_size, 8), dtype=float)
    rewards = jnp.zeros(n + 1)
    terminals = jnp.full(n, False, dtype=bool)
    d2lt = jnp.zeros(n, dtype=int)
    throughputs = jnp.zeros((n, 1000), dtype=int)

    drl = QLBT(
        q_network=QLBTNetwork(num_actions),
        obs_space_shape=(n_drl, obs.shape[1], obs.shape[2] + 1),
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
    drl_states, drl_step = init_agents(drl, init_key, n_drl, apply_vmap=False)

    dcf = DCF()
    key, init_key = jax.random.split(key)
    legacy_states, legacy_step = init_agents(dcf, init_key, n - n_drl, apply_vmap=True)

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
        drl_states, legacy_states, traffic_states, buffer_states, power_states, d2lt, throughputs,
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
        plot_first(filename)
