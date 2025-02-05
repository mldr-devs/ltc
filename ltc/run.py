from dataclasses import dataclass
from functools import partial

import cloudpickle
import jax
import jax.numpy as jnp
import lz4.frame
import optax
from jax_tqdm import scan_tqdm
from reinforced_lib.agents import AgentState
from reinforced_lib.agents.deep import DDQN

from ltc.agents import DCF, QNetwork
from ltc.sim import InitialStateConf, ModelState, cox_traffic, process_output, simulate


@jax.tree_util.register_dataclass
@dataclass
class Carry:
    drl_states: AgentState
    dcf_states: AgentState
    traffic_states: ModelState
    buffer_states: jax.Array
    channel_state: int
    key: jax.random.PRNGKey
    obs: jax.Array
    actions: jax.Array
    rewards: jax.Array
    terminal: bool


@jax.tree_util.register_dataclass
@dataclass
class Output:
    dcf_states: AgentState
    actions: jax.Array
    rewards: jax.Array
    buffer_states: jax.Array
    channel_state: int


def init_agents(agent, key, n):
    keys = jax.random.split(key, n)
    states = jax.vmap(agent.init)(keys)
    step_fn = jax.vmap(partial(agent_step, agent), in_axes=(0, 0, 0, 0, 0, None))
    return states, step_fn


def agent_step(agent, state, key, obs, action, reward, terminal):
    update_key, sample_key = jax.random.split(key)
    state = agent.update(state, update_key, obs, action, reward, terminal)
    action = agent.sample(state, sample_key, obs)
    return state, action


def init_traffic(traffic, key, n):
    keys = jax.random.split(key, n)
    states = jax.vmap(traffic.init)(keys)
    step_fn = jax.vmap(traffic.sample, in_axes=(0, 0))
    return states, step_fn


def rl_step(drl_step, dcf_step, traffic_step, n, n_drl):
    def rl_step_fn(c, _):
        key, drl_keys, dcf_keys, traffic_key = jax.random.split(c.key, 4)
        drl_keys = jax.random.split(drl_keys, n_drl)
        dcf_keys = jax.random.split(dcf_keys, n - n_drl)
        traffic_keys = jax.random.split(traffic_key, n)

        drl_states, drl_actions = drl_step(c.drl_states, drl_keys, c.obs[:n_drl], c.actions[:n_drl], c.rewards[:n_drl], c.terminal)
        dcf_states, dcf_actions = dcf_step(c.dcf_states, dcf_keys, c.obs[n_drl:], c.actions[n_drl:], c.rewards[n_drl:], c.terminal)
        actions = jnp.concatenate([drl_actions, dcf_actions])

        traffic_states, new_frames = traffic_step(c.traffic_states, traffic_keys)
        new_buffer_states, channel_state = simulate(c.buffer_states, new_frames, actions)
        obs, rewards = process_output(c.buffer_states, new_buffer_states, actions, channel_state, c.obs)

        c = Carry(drl_states, dcf_states, traffic_states, new_buffer_states, channel_state, key, obs, actions, rewards, c.terminal)
        o = Output(dcf_states, actions, rewards, new_buffer_states, channel_state)
        return c, o

    return rl_step_fn


if __name__ == '__main__':
    n, n_drl = 5, 1
    n_steps = 10000
    window_size = 50
    seed = 42

    key = jax.random.key(seed)
    actions = jnp.zeros(n, dtype=int)
    buffer_states = jnp.zeros(n, dtype=int)
    channel_state = 0
    obs = jnp.zeros((n, window_size, 3), dtype=int)
    rewards = jnp.zeros(n)
    terminal = False

    drl = DDQN(
        q_network=QNetwork(),
        obs_space_shape=(window_size, 3),
        act_space_size=2,
        optimizer=optax.adam(3e-4),
        experience_replay_buffer_size=10000,
        experience_replay_batch_size=128,
        experience_replay_steps=5,
        discount=0.99,
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
    rl_step_fn = scan_tqdm(n_steps)(rl_step_fn)
    carry = Carry(drl_states, dcf_states, traffic_states, buffer_states, channel_state, key, obs, actions, rewards, terminal)
    carry, output = jax.lax.scan(rl_step_fn, carry, jnp.arange(n_steps))

    with lz4.frame.open(f'history_{n}_{n_drl}_{seed}.pkl.lz4', 'wb') as f:
        cloudpickle.dump((carry, output), f)
