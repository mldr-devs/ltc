from functools import partial

import jax
import jax.numpy as jnp
from chex import dataclass
from reinforced_lib.agents import BaseAgent, AgentState

from ltc.sim.constants import Actions, Features
from ltc.sim.features import select_features


@dataclass
class Idle_sense_state(AgentState):
    cw: float
    backoff: int
    tx_counter: int
    idle_counter: int


class Idle_sense(BaseAgent):
    FEATURES = (Features.BUFFER, Features.CHANNEL)

    # Paper's 5.68 is Eq. 12 under 802.11b timings (T_c / T_SLOT = 68.17). Here a collision,
    # a success and an idle slot all cost one slot and nothing is sensed before transmitting,
    # so eta = 1 - T_SLOT / T_c = 0, Eq. 8 gives zeta = 1 and n_i = e^-1 / (1 - e^-1).
    N_TARGET = 0.582
    CW = 16

    def __init__(self, eps: float = 0.001, inv_alpha: float = 1.2, update_interval: int = 50):
        self.init = jax.jit(self.init)
        self.update = jax.jit(partial(
            self.update, eps=eps, inv_alpha=inv_alpha, update_interval=update_interval
        ))
        self.sample = jax.jit(self.sample)

    @staticmethod
    def init(key):
        cw = jnp.float32(Idle_sense.CW)
        backoff = jax.random.randint(key, (), 0, Idle_sense.CW)
        return Idle_sense_state(cw=cw, backoff=backoff, tx_counter=0, idle_counter=0)

    @staticmethod
    def update(state, key, env_state, action, reward, terminal,
               eps: float, inv_alpha: float, update_interval: int):
        buffer, channel = select_features(env_state, Idle_sense.FEATURES)[-1]

        def on_transmission():
            new_tx_counter = state.tx_counter + 1

            def update_CW():
                n_idle = state.idle_counter / new_tx_counter
                new_cw = jax.lax.cond(
                    n_idle < Idle_sense.N_TARGET,
                    lambda: state.cw * inv_alpha,
                    lambda: 2 * state.cw / (2 + eps * state.cw)
                )
                new_cw = jnp.maximum(new_cw, 2.0)
                return Idle_sense_state(cw=new_cw, backoff=state.backoff,
                                        tx_counter=0, idle_counter=0)

            def freeze_CW():
                return Idle_sense_state(cw=state.cw, backoff=state.backoff,
                                        tx_counter=new_tx_counter,
                                        idle_counter=state.idle_counter)

            return jax.lax.cond(
                new_tx_counter % update_interval == 0,
                update_CW, freeze_CW
            )

        def on_idle():
            return Idle_sense_state(cw=state.cw, backoff=state.backoff,
                                    tx_counter=state.tx_counter,
                                    idle_counter=state.idle_counter + 1)

        state = jax.lax.cond(channel != 0, on_transmission, on_idle)

        def backoff_reset():
            new_backoff = jax.random.randint(key, (), 0, jnp.maximum(jnp.round(state.cw), 1).astype(int))
            return Idle_sense_state(cw=state.cw, backoff=new_backoff,
                                    tx_counter=state.tx_counter,
                                    idle_counter=state.idle_counter)

        def countdown():
            return Idle_sense_state(cw=state.cw, backoff=state.backoff - 1,
                                    tx_counter=state.tx_counter,
                                    idle_counter=state.idle_counter)

        return jax.lax.cond(
            jax.lax.bitwise_and(buffer != 0, state.backoff > 0),
            countdown,
            backoff_reset
        )

    @staticmethod
    def sample(state, key, env_state):
        buffer, _ = select_features(env_state, Idle_sense.FEATURES)[-1]

        return jnp.where(
            buffer == 0,
            Actions.IDLE.value,
            jnp.where(
                state.backoff == 0,
                Actions.TX.value,
                Actions.CS.value
            )
        )
