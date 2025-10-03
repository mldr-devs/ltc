import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

from dataclasses import replace
from functools import partial

import cloudpickle
import jax
import jax.numpy as jnp
import lz4.frame
import optax
from tqdm import trange

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


def agent_step(agent, state, key, obs, action, reward, terminal):
    update_key, sample_key = jax.random.split(key)

    def power_on(state, update_key, sample_key, obs, action, reward, terminal):
        state = agent.update(state, update_key, obs, action, reward, terminal)
        action = agent.sample(state, sample_key, obs)
        return state, action

    def power_off(state, update_key, sample_key, obs, action, reward, terminal):
        return state, Actions.IDLE.value

    return jax.lax.cond(
        terminal, power_off, power_on,
        state, update_key, sample_key, obs, action, reward, terminal
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

        drl_states, drl_actions = drl_step(c.drl_states, drl_keys, c.obs[:n_drl], c.actions[:n_drl], c.rewards[:n_drl], c.terminals[:n_drl])
        legacy_states, legacy_actions = legacy_step(c.legacy_states, legacy_keys, c.obs[n_drl:], c.actions[n_drl:], c.rewards[n_drl:], c.terminals[n_drl:])
        actions = jnp.concatenate([drl_actions, legacy_actions])

        traffic_states, new_frames = traffic_step(c.traffic_states, traffic_keys)
        buffer_states, channel_state = simulate(c.buffer_states, new_frames, actions)
        obs, rewards, powers = process_output(c.buffer_states, buffer_states, c.power_states, channel_state, c.obs, actions, c.terminals)
        terminals = jnp.logical_or(c.terminals, powers < 0)

        params = c.drl_states.params['model'] if 'model' in c.drl_states.params else c.drl_states.params
        flat_params, _ = jax.tree.flatten(params)
        flat_params = jax.tree.map(lambda x: x.reshape(n_drl, -1), flat_params)
        flat_params = jnp.hstack(flat_params)
        hist, bin_edges = jax.vmap(jnp.histogram, in_axes=(0, None))(flat_params, n_bins)

        c = Carry(
            drl_states, legacy_states, traffic_states, buffer_states, powers,
            channel_state, key, obs, actions, rewards, terminals
        )
        o = Output(
            legacy_states, obs, actions, rewards, terminals, buffer_states, powers,
            (new_frames > 0).astype(int), channel_state, hist, bin_edges
        )
        return c, o

    return rl_step_fn


if __name__ == '__main__':
    n, n_drl = 2, 1
    n_epochs, n_steps = 1, 500000
    window_size = 20
    seed = 42
    traffic_type = 'saturated'  # 'saturated', 'bursty'
    legacy_type = 'q-aloha'     # 'q-aloha', 'eb-aloha', 'fw-aloha', 'tdma'

    key = jax.random.key(seed)
    actions = jnp.zeros(n, dtype=int)
    buffer_states = jnp.zeros(n, dtype=int)
    power_states = jnp.full(n, INITIAL_CAPACITY, dtype=int)
    channel_state = 0
    obs = jnp.zeros((n, window_size, 3), dtype=int)
    rewards = jnp.zeros(n)
    terminals = jnp.full(n, False, dtype=bool)

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
        legacy = QALOHA(q=0.2)
    elif legacy_type == 'eb-aloha':
        legacy = EBALOHA(window_size=4, max_backoff=3)
    elif legacy_type == 'fw-aloha':
        legacy = FWALOHA(window_size=8)
    elif legacy_type == 'tdma':
        legacy = TDMA(state_size=10, assigned_slots=5)
    else:
        raise ValueError(f'Unknown legacy type: {legacy_type}')

    legacy_states, legacy_step = init_agents(legacy, init_key, n - n_drl)

    key, init_key = jax.random.split(key)

    if traffic_type == 'saturated':
        traffic = cox_traffic(f3dB=1.0, loc=0.0, scale=0.0, initial_state=InitialStateConf.ZERO)
    elif traffic_type == 'bursty':
        traffic = cox_traffic(f3dB=0.1, loc=-5.0, scale=5.0, initial_state=InitialStateConf.ZERO)
    else:
        raise ValueError(f'Unknown traffic type: {traffic_type}')

    traffic_states, traffic_step = init_traffic(traffic, init_key, n)

    rl_step_fn = jax.jit(rl_step(drl_step, legacy_step, traffic_step, n, n_drl))
    init_carry = Carry(
        drl_states, legacy_states, traffic_states, buffer_states, power_states,
        channel_state, key, obs, actions, rewards, terminals
    )
    all_outputs = []

    for _ in trange(n_epochs):
        carry, output = jax.lax.scan(rl_step_fn, init_carry, length=n_steps)
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

    # plot_all(filename)
    plot_first(filename)
