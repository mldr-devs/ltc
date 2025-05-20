import jax
import jax.numpy as jnp
from chex import dataclass
from reinforced_lib.agents import BaseAgent, AgentState

from ltc.sim.constants import Actions


@dataclass
class DCFState(AgentState):
    cw: int
    backoff: int


class DCF(BaseAgent):
    CW_MIN = 16
    CW_MAX = 1024

    def __init__(self):
        self.init = jax.jit(self.init)
        self.update = jax.jit(self.update)
        self.sample = jax.jit(self.sample)

    @staticmethod
    def init(key):
        cw = DCF.CW_MIN
        backoff = jax.random.randint(key, (), 0, cw)
        return DCFState(cw=cw, backoff=backoff)

    @staticmethod
    def update(state, key, env_state, action, reward, terminal):
        def reset():
            return DCF.init(key)

        def freeze():
            return state

        def countdown():
            backoff = jax.lax.max(state.backoff - 1, 0)
            return DCFState(cw=state.cw, backoff=backoff)

        def double_cw():
            cw = jax.lax.min(2 * state.cw, DCF.CW_MAX)
            backoff = jax.random.randint(key, (), 0, state.cw)
            return DCFState(cw=cw, backoff=backoff)

        buffer, channel, ret_c, _ = env_state[-1]

        return jax.lax.cond(
            buffer == 0,
            reset,
            lambda: jax.lax.cond(
                jax.lax.bitwise_and(state.backoff > 0, channel == 0),
                countdown,
                lambda: jax.lax.cond(
                    jax.lax.bitwise_and(state.backoff > 0, channel != 0),
                    freeze,
                    lambda: jax.lax.cond(
                        jax.lax.bitwise_or(reward > 0, ret_c == 0),
                        reset,
                        lambda: jax.lax.cond(
                            reward < 0,
                            double_cw,
                            freeze
                        )
                    )
                )
            )
        )

    @staticmethod
    def sample(state, key, env_state):
        buffer, channel, _, _ = env_state[-1]

        return jnp.where(
            buffer == 0,
            Actions.IDLE.value,
            jnp.where(
                jax.lax.bitwise_and(state.backoff == 0, channel == 0),
                Actions.TX.value,
                Actions.CS.value
            )
        )
