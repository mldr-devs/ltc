from functools import partial

import jax
import jax.numpy as jnp
import optax
from reinforced_lib.agents.deep import DDQN

from ltc.sim import generate_frames, process_output, simulate
from ltc.agents import DCF, QNetwork


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


def rl_step(carry, step, *, drl_step, dcf_step, n, n_drl):
    (drl_states, dcf_states), (buffer_states, channel_state), key, obs, actions, rewards, terminal = carry

    key, drl_keys, dcf_keys = jax.random.split(key, 3)
    drl_keys = jax.random.split(drl_keys, n_drl)
    dcf_keys = jax.random.split(dcf_keys, n - n_drl)

    drl_states, drl_actions = drl_step(drl_states, drl_keys, obs[:n_drl], actions[:n_drl], rewards[:n_drl], terminal)
    dcf_states, dcf_actions = dcf_step(dcf_states, dcf_keys, obs[n_drl:], actions[n_drl:], rewards[n_drl:], terminal)
    actions = jnp.concatenate([drl_actions, dcf_actions])

    new_frames = generate_frames(n, step)
    buffer_states, channel_state = simulate(buffer_states, new_frames, actions)
    buffer_states, obs, rewards = process_output(buffer_states, actions, channel_state, obs)

    carry = (drl_states, dcf_states), (buffer_states, channel_state), key, obs, actions, rewards, terminal
    return carry, step + 1


if __name__ == '__main__':
    n, n_drl = 10, 2
    n_steps = 1000
    window_size = 5
    key = jax.random.key(42)

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

    rl_step_fn = jax.jit(partial(rl_step, drl_step=drl_step, dcf_step=dcf_step, n=n, n_drl=n_drl))
    init = (drl_states, dcf_states), (buffer_states, channel_state), key, obs, actions, rewards, terminal
    (states, *_), _ = jax.lax.scan(rl_step_fn, init, jnp.arange(n_steps))
