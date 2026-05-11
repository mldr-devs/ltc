from functools import partial

import jax
import jax.numpy as jnp
from chex import dataclass
from reinforced_lib.agents import BaseAgent, AgentState

from ltc.sim.constants import Actions


@dataclass
class FWALOHAState(AgentState):
    backoff: int


class FWALOHA(BaseAgent):
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.init = jax.jit(partial(self.init, window_size=window_size))
        self.update = jax.jit(partial(self.update, window_size=window_size))
        self.sample = jax.jit(self.sample)

    @staticmethod
    def init(key, window_size: int):
        backoff = jax.random.randint(key, shape=(), minval=0, maxval=window_size)
        return FWALOHAState(backoff=backoff)

    @staticmethod
    def update(state, key, env_state, action, reward, terminal, window_size: int):
        reset_value = jax.random.randint(key, shape=(), minval=0, maxval=window_size)
        new_backoff = state.backoff - 1
        new_backoff = jnp.where(new_backoff < 0, reset_value, new_backoff)
        return FWALOHAState(backoff=new_backoff)

    @staticmethod
    def sample(state, key, env_state):
        buffer, *_ = env_state[-1]

        return jnp.where(
            (buffer > 0) & (state.backoff == 0),
            Actions.TX.value,
            Actions.IDLE.value
        )
