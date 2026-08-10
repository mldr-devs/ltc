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


def add_batch_dim(x):
    return x[None, ...] if x.ndim == 3 else x


class QNetwork(nn.Module):
    num_actions: int
    rnn_dim = 32
    fc_dim = 32

    @nn.compact
    def __call__(self, s):
        scan_gru = nn.scan(nn.GRUCell, variable_broadcast='params', split_rngs={'params': False}, in_axes=1, out_axes=1)
        h = scan_gru(self.rnn_dim).initialize_carry(jax.random.PRNGKey(0), s[:, 0].shape)

        _, s = scan_gru(self.rnn_dim)(h, s)
        s = nn.Dense(self.fc_dim)(s[:, -1])
        s = nn.relu(s)
        s = nn.Dense(self.num_actions)(s)

        return s


class MixingNetwork(nn.Module):
    fc_dim: int = 32

    @nn.compact
    def __call__(self, qs, g):
        b, n = qs.shape
        _, g_feat = g.shape

        W1 = self.param('W1', nn.initializers.lecun_normal(), (self.fc_dim * n, g_feat))
        b1 = self.param('b1', nn.initializers.lecun_normal(), (self.fc_dim, g_feat))
        W2 = self.param('W2', nn.initializers.lecun_normal(), ((n + 1) * self.fc_dim, g_feat))
        b2a = self.param('b2a', nn.initializers.lecun_normal(), (self.fc_dim, g_feat))
        b2b = self.param('b2b', nn.initializers.lecun_normal(), ((n + 1), self.fc_dim))

        W1s = jnp.abs(g @ W1.T).reshape(b, self.fc_dim, n)
        b1s = g @ b1.T
        W2s = jnp.abs(g @ W2.T).reshape(b, n + 1, self.fc_dim)
        b2s = g @ b2a.T
        b2s = nn.relu(b2s)
        b2s = b2s @ b2b.T

        def fwd(q, W1, b1, W2, b2):
            s = q.flatten()
            s = jnp.dot(s, W1.T) + b1.T
            s = nn.elu(s)
            s = jnp.dot(s, W2.T) + b2.T
            return s

        s = jax.vmap(fwd)(qs, W1s, b1s, W2s, b2s)
        return s


class QLBTNetwork(nn.Module):
    num_actions: int

    @nn.compact
    def __call__(self, s, epsilon=0.0):
        BatchQNetwork = nn.vmap(
            QNetwork,
            in_axes=1, out_axes=1,
            variable_axes={'params': 0},
            split_rngs={'params': True}
        )

        s = add_batch_dim(s)
        ss = s[..., :-4]  # remove auxiliary features (raw d2lt, ret_c, buffer_state, and reward)
        actions, d2lt = s[..., -1, 0], s[..., -1, 5]
        d2lt = d2lt / jnp.maximum(d2lt.sum(axis=-1, keepdims=True), 1)
        g = jnp.concatenate([actions, d2lt], axis=-1)

        q_ind = BatchQNetwork(self.num_actions)(ss)
        max_q = (q_ind == q_ind.max(axis=-1, keepdims=True)).astype(float)
        probs = (1 - epsilon) * max_q / jnp.sum(max_q, axis=-1, keepdims=True) + epsilon / self.num_actions
        acts = jax.random.categorical(self.make_rng('rlib'), jnp.log(probs), axis=-1)
        q_ind_inp = jnp.take_along_axis(q_ind, acts[..., None], axis=-1).squeeze(-1)
        
        q_mix = MixingNetwork()(q_ind_inp, g)
        q_vals = jnp.concatenate([q_mix, q_ind.reshape(s.shape[0], -1)], axis=-1)

        return q_vals, acts


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
        _, n = rewards_ind.shape

        (q_values, _), net_state = forward(q_network, params, state.net_state, key, states)
        q_tot, q_ind = q_values[..., :1], q_values[..., 1:n + 1]

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
        params, net_state, opt_state = jax.lax.cond(
            er.is_ready(replay_buffer),
            lambda p, lp, os: QLBT.gradient_step(p, lp, os, env_state.shape[0]),
            lambda p, lp, os: (p, lp[1].net_state, os),
            state.params, loss_params, state.opt_state
        )
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
