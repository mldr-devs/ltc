from functools import partial

import jax
import jax.numpy as jnp
import flax.linen as nn

from ltc.sim.constants import Features
from ltc.sim.features import raw_action, select_features


def add_batch_dim(x):
    return x[None, ...] if x.ndim == 2 else x


def dlma_observation(obs):
    channel = select_features(obs, (Features.CHANNEL,))[..., 0]
    buffer = select_features(obs, (Features.BUFFER,))[..., 0]
    return jnp.stack([channel, raw_action(obs), buffer], axis=-1)


def dlma_reward(rewards, channel_state, actions, new_buffer_states, terminals):
    return jnp.where(terminals, 0., jnp.full_like(rewards, channel_state == 1, dtype=float))


@jax.vmap
@jax.vmap
def prepare_observation(obs):
    def prepare_action(obs):
        return jax.nn.one_hot(obs[1], 2)

    def prepare_state(obs):
        """
        **WARNING**: This function assumes that there are only two nodes in the network (1 DLMA and 1 legacy)!

        1. ch = -1, act = 0  (collision)
        2. ch = 1, act = 0   (DLMA transmits)
        3. ch = 1, act = 1   (legacy transmits)
        4. ch = 0, act = 1   (idle)

        s = max(ch + 2 * act, 0)

        1. s = max(-1 + 0, 0) = 0  (collision)
        2. s = max(1 + 0, 0) = 1   (DLMA transmits)
        3. s = max(1 + 2, 0) = 3   (legacy transmits)
        4. s = max(0 + 2, 0) = 2   (idle)
        """

        state = obs[:-1]  # remove buffer state
        state = state[0] + 2 * state[1]
        state = jnp.maximum(state, 0)
        state = jax.nn.one_hot(state, 4)
        return state

    def prepare_rewards(obs):
        """
        **WARNING**: This function assumes that there are only two nodes in the network (1 DLMA and 1 legacy)!

        1. ch = -1, act = 0  (collision)        -> DLMA reward = 0, legacy reward = 0
        2. ch = 1, act = 0   (DLMA transmits)   -> DLMA reward = 1, legacy reward = 0
        3. ch = 1, act = 1   (legacy transmits) -> DLMA reward = 0, legacy reward = 1
        4. ch = 0, act = 1   (idle)             -> DLMA reward = 0, legacy reward = 0
        """

        state = obs[:-1]  # remove buffer state
        no_tx = jnp.maximum(state[0], 0)
        dlma_tx = no_tx * (1 - state[1])
        legacy_tx = no_tx * state[1]
        reward = jnp.array([dlma_tx, legacy_tx])
        return reward

    action = prepare_action(obs)
    state = prepare_state(obs)
    reward = prepare_rewards(obs)
    obs = jnp.concatenate([action, state, reward], axis=0)
    return obs


class DLMANetwork(nn.Module):
    num_actions: int = 2
    FEATURES = (Features.CHANNEL, Features.ACTION_TX, Features.ACTION_CS, Features.BUFFER)

    @nn.compact
    def __call__(self, s, training=True):
        dense = partial(nn.Dense, kernel_init=nn.initializers.he_normal())

        s = add_batch_dim(s)
        s = prepare_observation(dlma_observation(s))

        b, *_ = s.shape
        x = s.reshape(b, -1)

        for _ in range(2):
            x = dense(64)(x)
            x = nn.relu(x)

        for _ in range(2):
            x_res = x
            x = dense(64)(x)
            x = nn.relu(x)
            x = dense(64)(x)
            x = nn.relu(x)
            x = x + x_res

        x = dense(self.num_actions)(x)
        return x
