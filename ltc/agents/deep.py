import jax
from reinforced_lib.agents.deep import DDQN
from reinforced_lib.agents.deep.ddqn import DDQNState
import flax.linen as nn
from chex import dataclass, Array, PRNGKey, Scalar, Shape
from reinforced_lib.utils.jax_utils import forward
import jax.numpy as jnp
import optax

class BayesianDDQN(DDQN):
    @staticmethod
    def loss_fn(
            params: dict,
            key: PRNGKey,
            state: DDQNState,
            batch: tuple,
            q_network: nn.Module,
            discount: Scalar
    ) -> tuple[Scalar, dict]:
        states, actions, rewards, terminals, next_states = batch
        q_key, q_target_key = jax.random.split(key)

        q_values, net_state = forward(q_network, params, state.net_state,
                                      q_key, states)
        q_values = jnp.take_along_axis(q_values, actions.astype(int), axis=-1)

        q_values_target, variables = forward(q_network, state.params_target,
                                     state.net_state_target, q_target_key,
                                     next_states)
        target = rewards + (1 - terminals) * discount * jnp.max(
            q_values_target, axis=-1, keepdims=True)

        kl = sum(jax.tree.leaves(variables['loss']))

        target = jax.lax.stop_gradient(target)
        loss = optax.l2_loss(q_values, target).mean() + kl

        return loss, net_state


