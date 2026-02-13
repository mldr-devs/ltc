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
from tqdm import trange

from ltc.agents import BayesianDDQN, DCF, QNetwork, StochasticVariationalNetwork
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
    def rl_step_coroutine(c, step):
        if n_switch is not None:
            active = jnp.where(step == n_switch, jnp.ones_like(c.active).at[n_final:].set(False), c.active)
        else:
            active = c.active
        
        key, drl_keys, legacy_keys, traffic_key, reward_key = jax.random.split(c.key, 5)
        drl_keys = jax.random.split(drl_keys, n_drl)
        legacy_keys = jax.random.split(legacy_keys, n - n_drl)
        traffic_keys = jax.random.split(traffic_key, n)

        drl_states, drl_actions = yield drl_step(
            c.drl_states, drl_keys, c.obs[:n_drl], c.actions[:n_drl], c.rewards[:n_drl], c.terminals[:n_drl], active[:n_drl]
        )
        legacy_states, legacy_actions = legacy_step(
            c.legacy_states, legacy_keys, c.obs[n_drl:], c.actions[n_drl:], c.rewards[n_drl:], c.terminals[n_drl:], active[n_drl:]
        )
        actions = jnp.concatenate([drl_actions, legacy_actions])

        traffic_states, new_frames = traffic_step(c.traffic_states, traffic_keys)
        buffer_states, channel_state = simulate(c.buffer_states, new_frames, actions)
        obs, rewards, powers = process_output(
            c.buffer_states, buffer_states, c.power_states, channel_state, c.obs, actions, c.terminals, reward_key
        )
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
        yield c, o

    def rl_step_fn(*args, **kwargs):
        gen = rl_step_coroutine(*args, **kwargs)
        intercepted = next(gen)
        return gen.send(intercepted)

    def pre_rl_fn(*args, **kwargs):
        gen = rl_step_coroutine(*args, **kwargs)
        intercepted = next(gen)
        return intercepted

    def post_rl_fn(intermediate, *args, **kwargs):
        gen = rl_step_coroutine(*args, **kwargs)
        # Unused computations will be DCE-ed
        _ = next(gen)
        return gen.send(intermediate)

    return rl_step_fn, pre_rl_fn, post_rl_fn


def setup_args():
    parser = argparse.ArgumentParser(description="Run the RL network simulation with configurable parameters.")
    parser.add_argument('--n', type=int, default=5, help='Initial number of agents in the simulation.')
    parser.add_argument('--n_final', type=int, help='Final number of agents in the simulation.')
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of training epochs to run.')
    parser.add_argument('--n_steps', type=int, default=2000, help='Number of steps per epoch.')
    parser.add_argument('--window_size', type=int, default=1, help='Size of the observation window for each agent.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--save_plots', action='store_true', default=False, help='Whether to save the generated plots.')
    parser.add_argument('--loc', type=float, default=5.0, help='loc traffic generator parameter.')
    parser.add_argument('--scale', type=float, default=0.0, help='scale traffic generator parameter')
    parser.add_argument('--f3dB', type=float, default=1.0, help='f3dB traffic generator parameter')
    parser.add_argument('--traffic_type', type=str, default='saturated', choices=['constant', 'saturated', 'bursty', 'custom'],help="Traffic model to use: 'constant', 'saturated', 'bursty', or 'custom'.")
    parser.add_argument('--legacy_type', type=str, default='tdma', choices=['q-aloha', 'eb-aloha', 'fw-aloha', 'tdma'], help="Legacy agent type to use: 'q-aloha', 'eb-aloha', 'fw-aloha', or 'tdma'.")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = setup_args()

    n_init = args.n
    n_final = args.n_final if args.n_final is not None else n_init
    n = max(n_init, n_final)
    n_epochs = args.n_epochs
    n_steps = args.n_steps
    window_size = args.window_size
    seed = args.seed
    traffic_type = args.traffic_type

    loc = args.loc
    scale = args.scale
    f3dB = args.f3dB

    key = jax.random.key(seed)
    num_actions = 2
    actions = jnp.zeros(n, dtype=int)
    buffer_states = jnp.zeros(n, dtype=int)
    power_states = jnp.full(n, INITIAL_CAPACITY, dtype=int)
    channel_state = 0
    obs = jnp.zeros((n, window_size, 5), dtype=int)
    rewards = jnp.zeros(n)
    terminals = jnp.full(n, False, dtype=bool)
    active = jnp.ones(n, dtype=bool).at[n_init:].set(False)

    lr_schedule = optax.cosine_decay_schedule(init_value=1e-4, decay_steps=60000, alpha=0.01)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(lr_schedule, b1=0.95, b2=0.95)
    )

    drl = BayesianDDQN(
        q_network=StochasticVariationalNetwork(QNetwork(num_actions, num_layers=1, dim=64, num_heads=4)),
        obs_space_shape=obs.shape[1:],
        act_space_size=num_actions,
        optimizer=optimizer,
        experience_replay_buffer_size=30000,
        experience_replay_batch_size=128,
        experience_replay_steps=5,
        discount=0.95,
        epsilon=1.0,
        epsilon_decay=0.999,
        epsilon_min=0.0,
        tau=0.05
    )
    key, init_key = jax.random.split(key)
    drl_states, drl_step = init_agents(drl, init_key, n)
    drl_states = drl_states.replace(prev_env_state=drl_states.prev_env_state.astype(int))

    dcf = DCF()
    key, init_key = jax.random.split(key)
    legacy_states, legacy_step = init_agents(dcf, init_key, 0)

    key, init_key = jax.random.split(key)

    if traffic_type == 'constant':
        traffic = cox_traffic(f3dB=1.0, loc=-1.0, scale=0.0, initial_state=InitialStateConf.ZERO)
    elif traffic_type == 'saturated':
        traffic = cox_traffic(f3dB=1.0, loc=5.0, scale=0.0, initial_state=InitialStateConf.ZERO)
    elif traffic_type == 'bursty':
        traffic = cox_traffic(f3dB=0.1, loc=-5.0, scale=5.0, initial_state=InitialStateConf.ZERO)
    elif traffic_type == 'custom':
        traffic = cox_traffic(f3dB=f3dB, loc=loc, scale=scale, initial_state=InitialStateConf.ZERO)
    else:
        raise ValueError(f'Unknown traffic type: {traffic_type}')

    traffic_states, traffic_step = init_traffic(traffic, init_key, n)

    rl_step_fn, _, _ = rl_step(drl_step, legacy_step, traffic_step, n, n)
    rl_step_fn = jax.jit(rl_step_fn)
    carry = Carry(
        drl_states, legacy_states, traffic_states, buffer_states, power_states,
        channel_state, key, obs, actions, rewards, terminals, active
    )
    all_outputs = []

    for epoch in trange(n_epochs):
        if epoch == int(n_epochs / 2):
            carry = replace(carry, active=jnp.ones_like(carry.active).at[n_final:].set(False))
        
        carry, output = jax.lax.scan(rl_step_fn, carry, length=n_steps)
        all_outputs.append(output)

    all_outputs = jax.tree.map(lambda *x: jnp.stack(x), *all_outputs)
    filename = f'history_{n_init}_{n_final}_{seed}.pkl.lz4'

    with lz4.frame.open(filename, 'wb') as f:
        cloudpickle.dump((carry.drl_states, all_outputs), f)

    if args.save_plots:
        plot_all(filename)
