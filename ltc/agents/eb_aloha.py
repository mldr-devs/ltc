from functools import partial

import jax
import jax.numpy as jnp
from chex import dataclass
from reinforced_lib.agents import BaseAgent, AgentState

from ltc.sim.constants import Actions


@dataclass
class EBALOHAState(AgentState):
    backoff: int
    ret_c: int


class EBALOHA(BaseAgent):
    def __init__(self, window_size: int, max_backoff: int):
        self.window_size = window_size
        self.max_backoff = max_backoff
        self.init = jax.jit(partial(self.init, window_size=window_size))
        self.update = jax.jit(partial(self.update, window_size=window_size, max_backoff=max_backoff))
        self.sample = jax.jit(self.sample)

    @staticmethod
    def init(key, window_size: int):
        ret_c = 0
        backoff = jax.random.randint(key, shape=(), minval=0, maxval=window_size)
        return EBALOHAState(backoff=backoff, ret_c=ret_c)

    @staticmethod
    def update(state, key, env_state, action, reward, terminal, window_size: int, max_backoff: int):
        collision = (action == Actions.TX.value) & (reward == 0)
        success = (action == Actions.TX.value) & (reward > 0)

        new_ret_c = jnp.where(
            collision,
            jnp.minimum(state.ret_c + 1, max_backoff),
            jnp.where(success, 0, state.ret_c)
        )
        window = window_size * (2 ** new_ret_c)

        reset_value = jax.random.randint(key, shape=(), minval=0, maxval=window)
        new_backoff = state.backoff - 1
        new_backoff = jnp.where(new_backoff < 0, reset_value, new_backoff)

        return EBALOHAState(backoff=new_backoff, ret_c=new_ret_c)

    @staticmethod
    def sample(state, key, env_state):
        buffer, *_ = env_state[-1]

        return jnp.where(
            (buffer > 0) & (state.backoff == 0),
            Actions.TX.value,
            Actions.IDLE.value
        )
