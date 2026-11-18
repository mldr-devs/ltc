from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from chex import dataclass, PRNGKey, Array
from reinforced_lib.agents import BaseAgent, AgentState

from ltc.symbolic.util import SimplexCode


def fronts(sr_model) -> list:
    equations = sr_model.equations_
    return equations if isinstance(equations, list) else [equations]


def capped_indices(sr_model, complexity_cap: int | None):
    """Per output, the most complex front equation within ``complexity_cap``; None defers to PySR."""
    if complexity_cap is None:
        return None
    indices = [max(np.flatnonzero(f['complexity'].to_numpy() <= complexity_cap), default=0) for f in fronts(sr_model)]
    indices = [int(i) for i in indices]
    return indices if len(indices) > 1 else indices[0]


@dataclass
class SRJaxState(AgentState):
    parameters: Any


class SRJaxAgent(BaseAgent):
    def __init__(
        self, sr_model, complexity_cap: int | None = None, n_actions: int = 2, n_features: int | None = None,
        stochastic: bool = False, temperature: float = 1.0,
    ):
        feature_names = getattr(sr_model, 'feature_names_in_', None)
        if n_features is not None and feature_names is not None and n_features != len(feature_names):
            raise ValueError(
                f'The symbolic model was fitted on {len(feature_names)} features but the simulation supplies '
                f'{n_features}. Match --window_size to the history the model was distilled from.'
            )

        index = capped_indices(sr_model, complexity_cap)
        rows = sr_model.get_best(index=index)
        for k, row in enumerate(rows if isinstance(rows, list) else [rows]):
            print(f'SR output {k}: complexity {row["complexity"]}, loss {row["loss"]:.5g}: {row["equation"]}')

        jaxeqs = sr_model.jax(index)
        jaxeqs = jaxeqs if isinstance(jaxeqs, list) else [jaxeqs]
        callables = tuple(jax.jit(eq['callable']) for eq in jaxeqs)
        parameters = [eq['parameters'] for eq in jaxeqs]
        simplex = SimplexCode(T=n_actions)
        if len(callables) != simplex.T - 1:
            raise ValueError(f'The model has {len(callables)} outputs, {n_actions} actions need {simplex.T - 1}.')

        self.init = jax.jit(partial(self.init, parameters=parameters))
        self.update = jax.jit(self.update)
        self.sample = jax.jit(partial(
            self.sample, callables=callables, simplex=simplex, stochastic=stochastic, temperature=temperature,
        ))

    @staticmethod
    def init(key: PRNGKey, parameters: Any) -> SRJaxState:
        return SRJaxState(parameters=parameters)

    @staticmethod
    def update(state: SRJaxState, key: PRNGKey, env_state: Array, action: Array, reward: Array, terminal: Array) -> SRJaxState:
        return state

    @staticmethod
    def sample(
        state: SRJaxState, key: PRNGKey, env_state: Array, callables, simplex: SimplexCode,
        stochastic: bool, temperature: float,
    ) -> Array:
        x = env_state.reshape(1, -1).astype(jnp.float32)
        codes = jnp.stack([f(x, p).reshape(()) for f, p in zip(callables, state.parameters)])[None]
        if stochastic:
            logits = jnp.log(jnp.clip(simplex.probs(codes)[0], 1e-12)) / temperature
            return jax.random.categorical(key, logits)
        return simplex.decode(codes)[0]
