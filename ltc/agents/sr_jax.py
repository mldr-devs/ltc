from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from chex import dataclass, PRNGKey, Array
from reinforced_lib.agents import BaseAgent, AgentState

from ltc.sim.constants import Features
from ltc.sim.features import select_features
from ltc.symbolic.util import SimplexCode, history_reshape


@dataclass
class SRJaxState(AgentState):
    parameters: Any


class SRJaxAgent(BaseAgent):
    FEATURES = tuple(Features)

    def __init__(
        self, sr_model, equation_index: int, n_actions: int = 2, n_features: int | None = None,
        stochastic: bool = False, temperature: float = 1.0,
    ):
        feature_names = getattr(sr_model, 'feature_names_in_', None)
        expected = 0 if feature_names is None else len(feature_names)
        if n_features is not None and expected and n_features != expected:
            # PySR's jax callable indexes X by fit-time column, and JAX clamps out-of-range
            # indices instead of raising, so a mismatch silently evaluates the expression on
            # the wrong columns and the policy degenerates to a constant action.
            raise ValueError(
                f'The symbolic model was fitted on {expected} features but the simulation supplies '
                f'{n_features}. Set --window_size {expected // len(Features)} to match the history '
                f'the model was distilled from.'
            )

        jaxeq = sr_model.jax(equation_index)
        callable_fn = jax.jit(jaxeq["callable"])
        parameters = jaxeq["parameters"]
        simplex = SimplexCode(T=n_actions)

        self.init = jax.jit(partial(self.init, parameters=parameters))
        self.update = jax.jit(self.update)
        self.sample = jax.jit(partial(
            self.sample, callable_fn=callable_fn, simplex=simplex,
            stochastic=stochastic, temperature=temperature,
        ))

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
        stochastic: bool,
        temperature: float,
    ) -> Array:
        # env_state: [window_size, n_features] raw int obs
        env_state = select_features(env_state, SRJaxAgent.FEATURES)
        # a single obs is a one-step, one-agent history: [1, 1, w, f] -> [1, w*f]
        x = history_reshape(env_state[jnp.newaxis, jnp.newaxis]).astype(jnp.float32)
        yhat = callable_fn(x, state.parameters)                     # [T-1] or [1*(T-1)]
        codes = yhat.reshape(1, simplex.T - 1)                      # [1, T-1]

        if stochastic:
            # See Forester.sample: one shared deterministic policy puts every
            # station in lockstep. Unlike the forest's votes these logits are
            # uncalibrated inner products, so the temperature does real work here.
            return jax.random.categorical(key, simplex.logits(codes)[0] / temperature)

        return simplex.decode(codes)[0]                             # scalar
