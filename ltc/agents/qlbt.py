from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from chex import Array, PRNGKey, Scalar
from reinforced_lib.agents.deep.ddqn import DDQN, DDQNState
from reinforced_lib.utils.experience_replay import ExperienceReplay
from reinforced_lib.utils.jax_utils import forward

from ltc.sim.constants import Actions


class QLBT(DDQN):
    @staticmethod
    def loss_fn(
            params: dict,
            key: PRNGKey,
            state: DDQNState,
            batch: tuple,
            q_network: nn.Module,
            discount: Scalar
    ) -> tuple[Scalar, dict]:
        states, _, rewards_tot, terminals, next_states = batch
        q_key, q_target_key = jax.random.split(key)

        rewards_ind = next_states[..., -1, -1]
        b, n = rewards_ind.shape
        beta = n

        q_values, net_state = forward(q_network, params, state.net_state, q_key, states)
        q_tot, q_ind = q_values[..., :1], q_values[..., 1:n + 1]
        q_values = q_values[..., n + 1:].reshape(b, n, -1)
        new_actions = jnp.argmax(q_values, axis=-1)
        next_states = next_states.at[..., -1, 0].set(new_actions)

        q_values_target, _ = forward(q_network, state.params_target, state.net_state_target, q_target_key, next_states)
        q_tot_target, q_ind_target = q_values_target[..., :1], q_values_target[..., 1:n + 1]

        target_tot = rewards_tot + (1 - terminals) * discount * q_tot_target
        target_ind = rewards_ind + (1 - terminals) * discount * q_ind_target

        target_tot = jax.lax.stop_gradient(target_tot)
        target_ind = jax.lax.stop_gradient(target_ind)
        loss = optax.l2_loss(q_tot, target_tot).mean() + beta * optax.l2_loss(q_ind, target_ind).mean()

        return loss, net_state

    @staticmethod
    def update(
            state: DDQNState,
            key: PRNGKey,
            env_state: Array,
            actions: Array,
            rewards: Scalar,
            terminal: Array,
            step_fn: Callable,
            er: ExperienceReplay,
            experience_replay_steps: int,
            epsilon_decay: Scalar,
            epsilon_min: Scalar,
            tau: Scalar
    ) -> DDQNState:
        filled_rewards = jnp.repeat(rewards[1:].reshape(-1, 1), env_state.shape[1], axis=1)[..., None]
        env_state = jnp.concatenate([env_state, filled_rewards], axis=-1)

        replay_buffer = er.append(state.replay_buffer, state.prev_env_state, 0, rewards[0], False, env_state)
        batch_key, network_key = jax.random.split(key)

        loss_params = (network_key, state, er.sample(replay_buffer, batch_key))
        params, net_state, opt_state, _ = step_fn(state.params, loss_params, state.opt_state)
        params_target, net_state_target = optax.incremental_update((params, net_state), (state.params_target, state.net_state_target), tau)

        return DDQNState(
            params=params,
            net_state=net_state,
            params_target=params_target,
            net_state_target=net_state_target,
            opt_state=opt_state,
            replay_buffer=replay_buffer,
            prev_env_state=env_state,
            epsilon=jax.lax.max(state.epsilon * epsilon_decay, epsilon_min)
        )

    @staticmethod
    def sample(
            state: DDQNState,
            key: PRNGKey,
            env_state: Array,
            wait: Array,
            q_network: nn.Module,
            act_space_size: int
    ) -> int:
        network_key, action_key = jax.random.split(key)
        dummy_state = jnp.zeros_like(env_state[..., 0])[..., None]
        env_state = jnp.concatenate([env_state, dummy_state], axis=-1)

        q, _ = forward(q_network, state.params, state.net_state, network_key, env_state)
        n = (q.shape[-1] - 1) // (act_space_size + 1)
        q = q[0, n + 1:].reshape(n, act_space_size)

        max_q = (q == q.max(axis=-1, keepdims=True)).astype(float)
        probs = (1 - state.epsilon) * max_q / jnp.sum(max_q, axis=-1, keepdims=True) + state.epsilon / q.shape[-1]

        action = jax.random.categorical(action_key, jnp.log(probs), axis=-1)
        return jnp.where(wait, Actions.CS.value, action)
