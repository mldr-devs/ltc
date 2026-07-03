from functools import partial

import jax
import jax.numpy as jnp
from chex import dataclass, Array, Scalar, PRNGKey

from reinforced_lib.agents import BaseAgent, AgentState


def _backoff_sample(key: PRNGKey, Q: Array, L: Scalar) -> int:
    diff = Q[0] - Q[1]
    prob_tx = jnp.where(diff > 0, 1.0, jnp.where(diff < 0, 0.0, 1.0 / (L + 1)))
    return jnp.where(jax.random.uniform(key) < prob_tx, 0, 1)


@dataclass
class MTOALState(AgentState):
    Q: Array


class MTOAL(BaseAgent):
    def __init__(self, alpha: Scalar, Q_th: Scalar, L: int) -> None:
        self.init = jax.jit(partial(self.init))
        self.update = jax.jit(partial(self.update, alpha=alpha, Q_th=Q_th))
        self.sample = jax.jit(partial(self.sample, L=L))

    @staticmethod
    def init(key: PRNGKey) -> MTOALState:
        return MTOALState(Q=jnp.zeros(2))

    @staticmethod
    def update(state: MTOALState, key: PRNGKey, action: int, reward: Scalar, alpha: Scalar, Q_th: Scalar) -> MTOALState:
        Q = state.Q.at[action].add(alpha * (reward - state.Q[action]))
        Q = Q.at[action].set(jnp.where(Q[action] <= Q_th, 0.0, Q[action]))
        return MTOALState(Q=Q)

    @staticmethod
    def sample(state: MTOALState, key: PRNGKey, L: Scalar) -> int:
        return _backoff_sample(key, state.Q, L)


@dataclass
class MTOAGState(AgentState):
    Q: Array
    W: Scalar


class MTOAG(BaseAgent):
    def __init__(self, alpha: Scalar, M: int, L: int) -> None:
        self.init = jax.jit(partial(self.init))
        self.update = jax.jit(partial(self.update, alpha=alpha, M=M))
        self.sample = jax.jit(partial(self.sample, L=L))

    @staticmethod
    def init(key: PRNGKey) -> MTOAGState:
        return MTOAGState(Q=jnp.zeros(2), W=jnp.zeros((), dtype=jnp.int32))

    @staticmethod
    def update(state: MTOAGState, key: PRNGKey, action: int, reward: Scalar, alpha: Scalar, M: int) -> MTOAGState:
        Q = state.Q.at[action].add(alpha * (reward - state.Q[action]))
        W = jnp.where(Q[action] > 0, state.W + 1, state.W)
        reset = W >= M
        Q = Q.at[action].set(jnp.where(reset, 0.0, Q[action]))
        W = jnp.where(reset, 0, W)
        return MTOAGState(Q=Q, W=W)

    @staticmethod
    def sample(state: MTOAGState, key: PRNGKey, L: Scalar) -> int:
        return _backoff_sample(key, state.Q, L)
