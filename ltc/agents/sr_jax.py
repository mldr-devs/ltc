from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from chex import dataclass, PRNGKey, Array
from reinforced_lib.agents import BaseAgent, AgentState

from ltc.sim.constants import Features
from ltc.sim.features import select_features
from ltc.symbolic.util import SimplexCode


@dataclass
class SRJaxState(AgentState):
    parameters: Any


class SRJaxAgent(BaseAgent):
    FEATURES = tuple(Features)

    def __init__(self, sr_model, equation_index: int, n_actions: int = 2):
        jaxeq = sr_model.jax(equation_index)
        callable_fn = jax.jit(jaxeq["callable"])
        parameters = jaxeq["parameters"]
        simplex = SimplexCode(T=n_actions)

        self.init = jax.jit(partial(self.init, parameters=parameters))
        self.update = jax.jit(self.update)
        self.sample = jax.jit(partial(self.sample, callable_fn=callable_fn, simplex=simplex))

    @staticmethod
    def init(key: PRNGKey, parameters: Any) -> SRJaxState:
        return SRJaxState(parameters=parameters)

    @staticmethod
    def update(
        state: SRJaxState,
        key: PRNGKey,
        env_state: Array,
        action: Array,
        reward: Array,
        terminal: Array,
    ) -> SRJaxState:
        return state

    @staticmethod
    def sample(
        state: SRJaxState,
        key: PRNGKey,
        env_state: Array,
        callable_fn,
        simplex: SimplexCode,
    ) -> Array:
        # env_state: [window_size, n_features] raw int obs
        env_state = select_features(env_state, SRJaxAgent.FEATURES)
        x = env_state.reshape(-1).astype(jnp.float32)[jnp.newaxis]  # [1, w*f]
        yhat = callable_fn(x, state.parameters)                     # [T-1] or [1*(T-1)]
        codes = yhat.reshape(1, simplex.T - 1)                      # [1, T-1]
        return simplex.decode(codes)[0]                             # scalar
