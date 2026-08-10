from functools import partial

import jax
import jax.numpy as jnp
from chex import dataclass
from reinforced_lib.agents import BaseAgent, AgentState

from ltc.sim.constants import Actions, COLLISION_PENALTY, Features, TX_REWARD
from ltc.sim.features import select_features


@dataclass
class StatelessQLearningState(AgentState):
    q: jax.Array
    rewards: jax.Array
    slot_timer: int
    slot: int


def stateless_q_reward(rewards, channel_state, actions, new_buffer_states, terminals):
    transmitted = actions == Actions.TX.value
    outcome = jnp.where(channel_state == 1, TX_REWARD, COLLISION_PENALTY)
    return jnp.where(terminals, 0., jnp.where(transmitted, outcome, 0.))


class StatelessQLearning(BaseAgent):
    FEATURES = (Features.BUFFER,)

    def __init__(self, n_slots: int = 5, lr: float = 0.1):
        self.n_slots = n_slots
        self.init = jax.jit(partial(self.init, n_slots=n_slots))
        self.update = jax.jit(partial(self.update, n_slots=n_slots, lr=lr))
        self.sample = jax.jit(self.sample)

    @staticmethod
    def init(key, n_slots: int):
        return StatelessQLearningState(
            q=jnp.zeros(n_slots),
            rewards=jnp.zeros(n_slots),
            slot_timer=n_slots - 1,
            slot=jax.random.randint(key, (), 0, n_slots),
        )

    @staticmethod
    def update(state, key, env_state, action, reward, terminal, n_slots: int, lr: float):
        rewards = state.rewards.at[state.slot_timer].set(reward)

        def end_of_frame():
            q = jnp.where(rewards != 0, state.q + lr * (rewards - state.q), state.q)
            best = q == jnp.max(q)
            slot = jax.random.choice(key, n_slots, p=best / jnp.sum(best))
            return StatelessQLearningState(q=q, rewards=jnp.zeros(n_slots), slot_timer=0, slot=slot)

        def next_slot():
            return StatelessQLearningState(
                q=state.q, rewards=rewards, slot_timer=state.slot_timer + 1, slot=state.slot
            )

        return jax.lax.cond(state.slot_timer == n_slots - 1, end_of_frame, next_slot)

    @staticmethod
    def sample(state, key, env_state):
        buffer = select_features(env_state, StatelessQLearning.FEATURES)[-1, 0]
        return jnp.where(
            (state.slot_timer == state.slot) & (buffer > 0), Actions.TX.value, Actions.IDLE.value
        )
