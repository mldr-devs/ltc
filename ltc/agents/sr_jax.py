from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from chex import dataclass, PRNGKey, Array
from reinforced_lib.agents import BaseAgent, AgentState

from ltc.sim.constants import Features
from ltc.sim.features import select_features
from ltc.symbolic.util import LogitCode, SimplexCode, history_reshape


@dataclass
class SRJaxState(AgentState):
    parameters: Any


def template_callable(sr_model, equation_index, n_features):
    """JAX callable returning the T-1 logits of a TemplateExpressionSpec model.

    PySR refuses .jax() and .sympy() on templates, but the equation is plain text
    -- "f1 = #1 * 2.0; f2 = #2" over the sub-expression's own arguments -- so
    sympy plus PySR's own sympy2jax covers it. Returns (callable, parameters) in
    the shape .jax() would have.
    """
    import re
    import sympy
    from pysr.export_jax import sympy2jax

    equation = sr_model.get_best(index=equation_index)['equation']
    symbols = sympy.symbols(f'x0:{n_features}')
    parts = [p.split('=', 1)[1].strip() for p in equation.split(';')]
    converted = [
        sympy2jax(sympy.sympify(re.sub(r'#(\d+)', lambda m: f'x{int(m.group(1)) - 1}', part)), symbols)
        for part in parts
    ]
    callables, parameters = zip(*converted)

    def stacked(x, params):
        # A constant sub-expression evaluates to a scalar, so broadcast before
        # stacking -- and constants are exactly what these fits keep selecting.
        rows = jnp.shape(x)[0]
        return jnp.stack(
            [jnp.broadcast_to(jnp.asarray(c(x, p)), (rows,)) for c, p in zip(callables, params)],
            axis=-1,
        )

    return stacked, list(parameters)


class SRJaxAgent(BaseAgent):
    FEATURES = tuple(Features)

    def __init__(
        self, sr_model, equation_index: int | None = None, n_actions: int = 2, n_features: int | None = None,
        stochastic: bool = False, temperature: float = 1.0, scale: float = 1.0,
        coding: str = 'simplex',
    ):
        # A multiclass template carries the label's dummy columns d1..d_{T-1} in X
        # alongside the real features; only the latter reach the sub-expressions.
        templated = type(getattr(sr_model, 'expression_spec_', None)).__name__ == 'TemplateExpressionSpec'
        feature_names = getattr(sr_model, 'feature_names_in_', None)
        expected = 0 if feature_names is None else len(feature_names)
        if templated:
            expected -= n_actions - 1
        if n_features is not None and expected and n_features != expected:
            # PySR's jax callable indexes X by fit-time column, and JAX clamps out-of-range
            # indices instead of raising, so a mismatch silently evaluates the expression on
            # the wrong columns and the policy degenerates to a constant action.
            raise ValueError(
                f'The symbolic model was fitted on {expected} features but the simulation supplies '
                f'{n_features}. Set --window_size {expected // len(Features)} to match the history '
                f'the model was distilled from.'
            )

        # None lets PySR pick off its own Pareto front, by whatever criterion its
        # model_selection is set to, instead of pinning an index that may be far from
        # the knee: on the bursty run the front's own choice scored a loss of 0.0014
        # where the previously hardcoded index 2 scored 0.91.
        selected = sr_model.get_best(index=equation_index)
        if not isinstance(selected, list):
            print(
                f'SR equation {selected.name} (complexity {selected["complexity"]}, '
                f'loss {selected["loss"]:.5g}): {selected["equation"]}'
            )

        if templated:
            fn, parameters = template_callable(sr_model, equation_index, expected)
        else:
            jaxeq = sr_model.jax(equation_index)
            fn, parameters = jaxeq["callable"], jaxeq["parameters"]
        callable_fn = jax.jit(fn)
        # 'logit' reads the expression as a logit against a zero reference class;
        # 'simplex' as a simplex codeword, where scale is the ScaleCalibrator
        # constant and only sharpens the sampling distribution, never the argmax.
        if coding == 'logit':
            codec = LogitCode(T=n_actions)
        else:
            codec = SimplexCode(T=n_actions, scale=scale)
            if stochastic and scale != 1.0:
                print(f'SR probability decoder calibrated with scale a={scale:.4g}')

        self.init = jax.jit(partial(self.init, parameters=parameters))
        self.update = jax.jit(self.update)
        self.sample = jax.jit(partial(
            self.sample, callable_fn=callable_fn, codec=codec,
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
        codec: SimplexCode | LogitCode,
        stochastic: bool,
        temperature: float,
    ) -> Array:
        # env_state: [window_size, n_features] raw int obs
        env_state = select_features(env_state, SRJaxAgent.FEATURES)
        # a single obs is a one-step, one-agent history: [1, 1, w, f] -> [1, w*f]
        x = history_reshape(env_state[jnp.newaxis, jnp.newaxis]).astype(jnp.float32)
        yhat = callable_fn(x, state.parameters)                     # [T-1] or [1*(T-1)]
        codes = yhat.reshape(1, codec.T - 1)                      # [1, T-1]

        if stochastic:
            # See Forester.sample: one shared deterministic policy puts every
            # station in lockstep. codec.probs already returns probabilities, so
            # they go into categorical as logs -- a softmax on top of them would
            # squash every gap to a factor of at most e and flatten the policy
            # (0.988 -> 0.739 for CS on the nonsaturated run).
            logits = jnp.log(jnp.clip(codec.probs(codes)[0], 1e-12)) / temperature
            return jax.random.categorical(key, logits)

        return codec.decode(codes)[0]                             # scalar
