from functools import partial

import jax
import jax.numpy as jnp
from chex import dataclass
from reinforced_lib.agents import BaseAgent, AgentState

from ltc.sim.constants import Actions, Features
from ltc.sim.features import select_features


@dataclass
class TDMAState(AgentState):
    counter: int
    tdma_slots: jnp.ndarray


class TDMA(BaseAgent):
    FEATURES = (Features.BUFFER,)

    def __init__(self, state_size: int, assigned_slots: int):
        self.state_size = state_size
        self.assigned_slots = assigned_slots
        self.init = jax.jit(partial(self.init, state_size=state_size, assigned_slots=assigned_slots))
        self.update = jax.jit(self.update)
        self.sample = jax.jit(self.sample)

    @staticmethod
    def init(key, state_size: int, assigned_slots: int):
        tdma_slots = jnp.zeros(state_size, dtype=int).at[:assigned_slots].set(1)
        tdma_slots = jax.random.permutation(key, tdma_slots)
        return TDMAState(counter=0, tdma_slots=tdma_slots)

    @staticmethod
    def update(state, key, env_state, action, reward, terminal):
        new_count = (state.counter + 1) % state.tdma_slots.shape[0]
        return TDMAState(counter=new_count, tdma_slots=state.tdma_slots)

    @staticmethod
    def sample(state, key, env_state):
        buffer = select_features(env_state, TDMA.FEATURES)[-1, 0]

        return jnp.where(
            (buffer > 0) & (state.tdma_slots[state.counter] == 1),
            Actions.TX.value,
            Actions.IDLE.value
        )
