from functools import partial

import jax
import jax.numpy as jnp
from reinforced_lib.agents.deep.ddqn import DDQN
from reinforced_lib.utils.jax_utils import forward


def time_uniform_prior(max_duration, tx_slots):
    weights = jnp.concatenate([
        jnp.full((max_duration,), 1.0 / tx_slots),
        jnp.ones((max_duration,)),
    ])
    return weights / jnp.sum(weights)


class AsymmetricExplorationDDQN(DDQN):
    """DDQN whose epsilon-greedy explores along ``exploration_prior`` instead of uniformly."""

    def __init__(self, q_network, act_space_size, exploration_prior, **kwargs):
        super().__init__(q_network=q_network, act_space_size=act_space_size, **kwargs)
        self.sample = jax.jit(partial(
            self.sample_with_prior,
            q_network=q_network,
            act_space_size=act_space_size,
            prior=exploration_prior,
        ))

    @staticmethod
    def sample_with_prior(state, key, env_state, q_network, act_space_size, prior):
        network_key, action_key = jax.random.split(key)
        q, _ = forward(q_network, state.params, state.net_state, network_key, env_state)
        max_q = (q == q.max()).astype(float)
        probs = (1 - state.epsilon) * max_q / jnp.sum(max_q) + state.epsilon * prior
        return jax.random.choice(action_key, act_space_size, p=probs.flatten())
