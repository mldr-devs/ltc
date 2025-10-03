from functools import partial

import jax
import jax.numpy as jnp
from chex import dataclass
from reinforced_lib.agents import BaseAgent, AgentState

from ltc.sim.constants import Actions


@dataclass
class QALOHAState(AgentState):
    pass


class QALOHA(BaseAgent):
    def __init__(self, q):
        self.init = jax.jit(self.init)
        self.update = jax.jit(self.update)
        self.sample = jax.jit(partial(self.sample, q=q))

    @staticmethod
    def init(key):
        return QALOHAState()

    @staticmethod
    def update(state, key, env_state, action, reward, terminal):
        return state

    @staticmethod
    def sample(state, key, env_state, q):
        *_, buffer = env_state[-1]

        return jnp.where(
            buffer == 0,
            Actions.IDLE.value,
            jnp.where(
                jax.random.uniform(key) < q,
                Actions.TX.value,
                Actions.IDLE.value
            )
        )
