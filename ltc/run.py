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

from ltc.agents import BayesianDDQN, DCF, QNetwork, QNetworkDropout, StochasticVariationalNetwork
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


def rl_step(drl_step, dcf_step, traffic_step, n, n_drl, n_bins=50):
    def rl_step_fn(c, _):
        key, drl_keys, dcf_keys, traffic_key = jax.random.split(c.key, 4)
        drl_keys = jax.random.split(drl_keys, n_drl)
        dcf_keys = jax.random.split(dcf_keys, n - n_drl)
        traffic_keys = jax.random.split(traffic_key, n)

        drl_states, drl_actions = drl_step(c.drl_states, drl_keys, c.obs[:n_drl], c.actions[:n_drl], c.rewards[:n_drl], c.terminals[:n_drl])
        # dcf_states, dcf_actions = dcf_step(c.dcf_states, dcf_keys, c.obs[n_drl:], c.actions[n_drl:], c.rewards[n_drl:], c.terminals[n_drl:])
        # actions = jnp.concatenate([drl_actions, dcf_actions])
        actions = drl_actions

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
            drl_states, dcf_states, traffic_states, buffer_states, powers,
            channel_state, key, obs, actions, rewards, terminals
        )
        o = Output(
            dcf_states, obs, actions, rewards, terminals, buffer_states, powers,
            (new_frames > 0).astype(int), channel_state, hist, bin_edges
        )
        return c, o

    return rl_step_fn


if __name__ == '__main__':
    n, n_drl = 10, 10
    n_epochs, n_steps = 50, 10000
    window_size = 5
    seed = 42

    key = jax.random.key(seed)
    actions = jnp.zeros(n, dtype=int)
    buffer_states = jnp.zeros(n, dtype=int)
    power_states = jnp.full(n, INITIAL_CAPACITY, dtype=int)
    channel_state = 0
    obs = jnp.zeros((n, window_size, 2), dtype=int).at[:, -1].set(INITIAL_CAPACITY)
    rewards = jnp.zeros(n)
    terminals = jnp.full(n, False, dtype=bool)

    drl = BayesianDDQN(
        q_network=StochasticVariationalNetwork(QNetworkDropout()),
        obs_space_shape=obs.shape[1:],
        act_space_size=3,
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

    key, init_key = jax.random.split(key)
    # dcf = DCF()
    # dcf_states, dcf_step = init_agents(dcf, init_key, n - n_drl)
    dcf_states = None
    dcf_step = None

    key, init_key = jax.random.split(key)
    traffic = cox_traffic(f3dB=0.1, loc=-5.0, scale=5.0, initial_state=InitialStateConf.ZERO)
    traffic_states, traffic_step = init_traffic(traffic, init_key, n)

    rl_step_fn = jax.jit(rl_step(drl_step, dcf_step, traffic_step, n, n_drl))
    init_carry = Carry(
        drl_states, dcf_states, traffic_states, buffer_states, power_states,
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

    plot_all(filename)
    plot_first(filename)
