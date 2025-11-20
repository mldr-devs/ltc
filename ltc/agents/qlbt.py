from functools import partial
from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from chex import Array, PRNGKey, Scalar
from reinforced_lib.agents.deep.ddqn import DDQN, DDQNState
from reinforced_lib.utils.experience_replay import ExperienceReplay
from reinforced_lib.utils.jax_utils import forward


class QLBT(DDQN):
    def __init__(
            self, 
            q_network: nn.Module, 
            *args, 
            optimizer: optax.GradientTransformation, 
            discount: Scalar, 
            **kwargs
    ) -> None:
        super().__init__(q_network, *args, optimizer=optimizer, discount=discount, **kwargs)
        QLBT.loss_fn = jax.jit(partial(QLBT.loss_fn, q_network=q_network, discount=discount))
        QLBT.gradient_step = jax.jit(partial(QLBT.gradient_step, optimizer=optimizer), static_argnames=['n'])

    @staticmethod
    def loss_fn(
            params: dict,
            key: PRNGKey,
            state: DDQNState,
            batch: tuple,
            idx: int,
            q_network: nn.Module,
            discount: Scalar,
            beta: Scalar = 1.0
    ) -> tuple[Scalar, dict]:
        states, _, rewards_tot, terminals, next_states = batch

        rewards_ind = next_states[..., -1, -1]
        b, n = rewards_ind.shape

        (q_values, _), net_state = forward(q_network, params, state.net_state, key, states, state.epsilon)
        q_tot, q_ind = q_values[..., :1], q_values[..., 1:n + 1]
        q_values = q_values[..., n + 1:].reshape(b, n, -1)
        new_actions = jnp.argmax(q_values, axis=-1)
        next_states = next_states.at[..., -1, 0].set(new_actions)

        (q_values_target, _), _ = forward(q_network, state.params_target, state.net_state_target, key, next_states)
        q_tot_target, q_ind_target = q_values_target[..., :1], q_values_target[..., 1:n + 1]

        target_tot = rewards_tot + (1 - terminals) * discount * q_tot_target
        target_ind = rewards_ind + (1 - terminals) * discount * q_ind_target

        target_tot = jax.lax.stop_gradient(target_tot)
        target_ind = jax.lax.stop_gradient(target_ind)

        def scan_fn(i, loss_i):
            return i + 1, jnp.where(jax.lax.bitwise_or(idx == i, idx == n), loss_i, jax.lax.stop_gradient(loss_i))

        loss_tot = optax.l2_loss(q_tot, target_tot).mean()
        loss_ind = optax.l2_loss(q_ind, target_ind).mean(axis=0)
        _, loss_ind = jax.lax.scan(scan_fn, 0, loss_ind)

        loss = loss_tot + beta * loss_ind.sum()
        return loss, net_state
    
    @staticmethod
    def combine_grads(grads: dict, aux: any, n: int) -> tuple[dict, any]:
        def scan_fn(g, i):
            return g, jax.tree.map(lambda x: x[i, i], g)
        
        aux = jax.tree.map(lambda x: x[-1], aux)
        grads_mix = jax.tree.map(lambda x: x[-1], grads['MixingNetwork_0'])
        _, grads_q = jax.lax.scan(scan_fn, grads['VmapQNetwork_0'], jnp.arange(n))
        
        grads = {'MixingNetwork_0': grads_mix, 'VmapQNetwork_0': grads_q}
        return grads, aux

    @staticmethod
    def gradient_step(
            objective: any, 
            loss_params: tuple, 
            opt_state: optax.OptState, 
            n: int,
            optimizer: optax.GradientTransformation
    ) -> tuple[any, any, optax.OptState]:
        vmaped_loss_fn = jax.vmap(jax.grad(QLBT.loss_fn, has_aux=True), in_axes=(None, None, None, None, 0))
        grads, aux = vmaped_loss_fn(objective, *loss_params, jnp.arange(n + 1))
        grads, aux = QLBT.combine_grads(grads, aux, n)
        updates, opt_state = optimizer.update(grads, opt_state, objective)
        objective = optax.apply_updates(objective, updates)
        return objective, aux, opt_state

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
        params, net_state, opt_state = QLBT.gradient_step(state.params, loss_params, state.opt_state, env_state.shape[0])
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
            q_network: nn.Module,
            act_space_size: int
    ) -> int:
        dummy_state = jnp.zeros_like(env_state[..., 0])[..., None]
        env_state = jnp.concatenate([env_state, dummy_state], axis=-1)
        (_, a), _ = forward(q_network, state.params, state.net_state, key, env_state, state.epsilon)
        return a[0]
