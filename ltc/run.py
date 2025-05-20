from dataclasses import dataclass, replace
from functools import partial

import cloudpickle
import jax
import jax.numpy as jnp
import lz4.frame
import optax
from reinforced_lib.agents import AgentState
from tqdm import trange

from ltc.agents import BayesianDDQN, DCF, QNetwork, QNetworkDropout, StochasticVariationalNetwork
from ltc.sim import InitialStateConf, ModelState, cox_traffic, process_output, simulate
from ltc.sim.constants import INITIAL_CAPACITY, Actions
from ltc.utils.plots import plot_all


@jax.tree_util.register_dataclass
@dataclass
class Carry:
    drl_states: AgentState
    dcf_states: AgentState
    traffic_states: ModelState
    buffer_states: jax.Array
    power_states: jax.Array
    channel_state: int
    key: jax.random.PRNGKey
    obs: jax.Array
    actions: jax.Array
    rewards: jax.Array
    terminals: jax.Array


@jax.tree_util.register_dataclass
@dataclass
class Output:
    dcf_states: AgentState
    observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    terminals: jax.Array
    buffer_states: jax.Array
    power_states: jax.Array
    channel_state: int


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


def rl_step(drl_step, dcf_step, traffic_step, n, n_drl):
    def rl_step_fn(c, _):
        key, drl_keys, dcf_keys, traffic_key = jax.random.split(c.key, 4)
        drl_keys = jax.random.split(drl_keys, n_drl)
        dcf_keys = jax.random.split(dcf_keys, n - n_drl)
        traffic_keys = jax.random.split(traffic_key, n)

        drl_states, drl_actions = drl_step(c.drl_states, drl_keys, c.obs[:n_drl], c.actions[:n_drl], c.rewards[:n_drl], c.terminals[:n_drl])
        dcf_states, dcf_actions = dcf_step(c.dcf_states, dcf_keys, c.obs[n_drl:], c.actions[n_drl:], c.rewards[n_drl:], c.terminals[n_drl:])
        actions = jnp.concatenate([drl_actions, dcf_actions])

        traffic_states, new_frames = traffic_step(c.traffic_states, traffic_keys)
        buffer_states, channel_state = simulate(c.buffer_states, new_frames, actions)
        obs, rewards, powers = process_output(c.buffer_states, buffer_states, c.power_states, channel_state, c.obs, actions, c.terminals)
        terminals = jnp.logical_or(c.terminals, powers < 0)

        c = Carry(
            drl_states, dcf_states, traffic_states, buffer_states, powers,
            channel_state, key, obs, actions, rewards, terminals
        )
        o = Output(dcf_states, obs, actions, rewards, terminals, buffer_states, powers, channel_state)
        return c, o

    return rl_step_fn


if __name__ == '__main__':
    n, n_drl = 10, 5
    n_epochs, n_steps = 500, 2000
    window_size = 5
    seed = 42

    key = jax.random.key(seed)
    actions = jnp.zeros(n, dtype=int)
    buffer_states = jnp.zeros(n, dtype=int)
    power_states = jnp.full(n, INITIAL_CAPACITY, dtype=int)
    channel_state = 0
    obs = jnp.zeros((n, window_size, 4), dtype=int)
    rewards = jnp.zeros(n)
    terminals = jnp.full(n, False, dtype=bool)

    drl = BayesianDDQN(
        q_network=StochasticVariationalNetwork(QNetworkDropout()),
        obs_space_shape=obs.shape[1:],
        act_space_size=3,
        optimizer=optax.adam(1e-3),
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
    dcf_states, dcf_step = init_agents(dcf, init_key, n - n_drl)

    key, init_key = jax.random.split(key)
    traffic = cox_traffic(f3dB=1.0, loc=0.0, scale=0.0, initial_state=InitialStateConf.ZERO)
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
