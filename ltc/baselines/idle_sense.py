import jax
import jax.numpy as jnp
from chex import dataclass
from reinforced_lib.agents import BaseAgent, AgentState

from ltc.sim.constants import Actions, Features
from ltc.sim.features import select_features


@dataclass
class Idle_sense_state(AgentState):
    cw: int
    backoff: int
    tx_counter: int
    idle_counter: int


class Idle_sense(BaseAgent):
    USES_AUX = True
    FEATURES = (Features.BUFFER, Features.CHANNEL, Features.RET_C)

    CW = 16
    EPS = 0.001
    ALPHA = 1.2
    N_TARGET = 5.68
    UPDATE_TX_INTERVAL = 5

    def __init__(self):
        self.init = jax.jit(self.init)
        self.update = jax.jit(self.update)
        self.sample = jax.jit(self.sample)

    @staticmethod
    def init(key):
        cw = Idle_sense.CW
        backoff = jax.random.randint(key, (), 0, cw)
        return Idle_sense_state(cw=cw, backoff=backoff, tx_counter=0, idle_counter=0)

    @staticmethod
    def update(state, key, env_state, action, reward, terminal, successful_station_id=-1, outcome=0):
        buffer, channel, ret_c = select_features(env_state, Idle_sense.FEATURES)[-1]

        # Step 1: update CW estimator based on channel activity
        def on_transmission():
            new_tx_counter = state.tx_counter + 1

            def update_CW():
                n_idle = state.idle_counter / new_tx_counter
                new_cw = jax.lax.cond(
                    n_idle < Idle_sense.N_TARGET,
                    lambda: state.cw * Idle_sense.ALPHA,
                    lambda: 2 * state.cw / (2 + Idle_sense.EPS * state.cw)
                )
                new_cw = jnp.maximum(jnp.round(new_cw).astype(jnp.int32), 2)
                return Idle_sense_state(cw=new_cw, backoff=state.backoff,
                                        tx_counter=0, idle_counter=0)

            def freeze_CW():
                return Idle_sense_state(cw=state.cw, backoff=state.backoff,
                                        tx_counter=new_tx_counter,
                                        idle_counter=state.idle_counter)

            return jax.lax.cond(
                new_tx_counter % Idle_sense.UPDATE_TX_INTERVAL == 0,
                update_CW, freeze_CW
            )

        def on_idle():
            return Idle_sense_state(cw=state.cw, backoff=state.backoff,
                                    tx_counter=state.tx_counter,
                                    idle_counter=state.idle_counter + 1)

        state = jax.lax.cond(channel != 0, on_transmission, on_idle)

        # Step 2: update backoff
        def full_reset():
            # Buffer empty — fully reinitialize
            return Idle_sense.init(key)

        def backoff_reset():
            # After TX attempt — draw new backoff, preserve estimator
            new_backoff = jax.random.randint(key, (), 0, jnp.maximum(state.cw.astype(int), 1))
            return Idle_sense_state(cw=state.cw, backoff=new_backoff,
                                    tx_counter=state.tx_counter,
                                    idle_counter=state.idle_counter)

        def countdown():
            return Idle_sense_state(cw=state.cw,
                                    backoff=jax.lax.max(state.backoff - 1, 0),
                                    tx_counter=state.tx_counter,
                                    idle_counter=state.idle_counter)

        def freeze():
            return state

        return jax.lax.cond(
            buffer == 0,
            full_reset,
            lambda: jax.lax.cond(
                jax.lax.bitwise_and(state.backoff > 0, channel == 0),
                countdown,
                lambda: jax.lax.cond(
                    jax.lax.bitwise_and(state.backoff > 0, channel != 0),
                    freeze,
                    lambda: jax.lax.cond(
                        jax.lax.bitwise_or(outcome > 0, ret_c == 0),
                        backoff_reset,
                        freeze
                    )
                )
            )
        )

    @staticmethod
    def sample(state, key, env_state):
        buffer, channel, _ = select_features(env_state, Idle_sense.FEATURES)[-1]

        return jnp.where(
            buffer == 0,
            Actions.IDLE.value,
            jnp.where(
                jax.lax.bitwise_and(state.backoff == 0, channel == 0),
                Actions.TX.value,
                Actions.CS.value
            )
        )
