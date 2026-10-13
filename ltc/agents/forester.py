from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from chex import dataclass, PRNGKey, Array
from reinforced_lib.agents import BaseAgent, AgentState

from ltc.sim.constants import Features
from ltc.sim.features import select_features
from ltc.symbolic.jax_random_forest import JaxRandomForest, _forest_forward


@dataclass
class ForesterState(AgentState):
    """Empty on purpose: nothing about this policy is per-station.

    The forest used to live here. ``ltc.run`` builds the agent states with
    ``jax.vmap(agent.init)`` and then threads them through ``jax.lax.scan`` as the
    carry, so every array in the state is replicated once per station and stays
    live for the whole rollout -- identical copies of the same thresholds. A
    1500-tree forest is 39 MiB, which is 393 MiB across ten stations before the
    scan doubles it for its input and output arguments; on the saturated run that
    reached 6.6 GB of scan arguments and exhausted the GPU.

    The tree arrays are constants, so they are closed over by ``sample`` the way
    ``max_depth`` already was, and leave the carry empty.
    """


class Forester(BaseAgent):
    """Agent whose policy is a scikit-learn random forest.

    Mirrors `SRJaxAgent`, with the symbolic expression replaced by a fitted
    `RandomForestClassifier` converted to JAX arrays by `JaxRandomForest`.
    """

    FEATURES = tuple(Features)

    def __init__(self, forest, n_actions: int = 2, stochastic: bool = False, temperature: float = 1.0):
        jrf = JaxRandomForest.from_sklearn(forest)

        if not jrf.is_classifier:
            raise ValueError('Forester requires a RandomForestClassifier.')

        classes = jnp.asarray(forest.classes_, dtype=jnp.int32)

        if classes.shape[0] > n_actions:
            raise ValueError(
                f'Forest predicts {classes.shape[0]} classes, but the action space has {n_actions}.'
            )

        parameters = {
            'feature': jrf.feature,
            'threshold': jrf.threshold,
            'left': jrf.left,
            'right': jrf.right,
            'value': jrf.value,
            'classes': classes,
        }

        self.init = jax.jit(self.init)
        self.update = jax.jit(self.update)
        self.sample = jax.jit(partial(
            self.sample, parameters=parameters, max_depth=jrf.max_depth,
            stochastic=stochastic, temperature=temperature,
        ))

    @staticmethod
    def init(key: PRNGKey) -> ForesterState:
        return ForesterState()

    @staticmethod
    def update(
        state: ForesterState,
        key: PRNGKey,
        env_state: Array,
        action: Array,
        reward: Array,
        terminal: Array,
    ) -> ForesterState:
        return state

    @staticmethod
    def sample(
        state: ForesterState,
        key: PRNGKey,
        env_state: Array,
        parameters: Any,
        max_depth: int,
        stochastic: bool,
        temperature: float,
    ) -> Array:
        # env_state: [window_size, n_features] raw int obs
        env_state = select_features(env_state, Forester.FEATURES)
        x = env_state.reshape(-1).astype(jnp.float32)[jnp.newaxis]  # [1, w*f]
        p = parameters
        probs = _forest_forward(
            p['feature'], p['threshold'], p['left'], p['right'], p['value'],
            x, max_depth,
        )                                                           # [1, n_classes]

        if stochastic:
            # Draw from the forest's own vote distribution. A shared deterministic
            # policy makes every station pick the same action from the same
            # observation, which locks the network into permanent collisions.
            index = jax.random.categorical(key, jnp.log(probs[0]) / temperature)
        else:
            index = jnp.argmax(probs[0])

        return p['classes'][index]                                  # scalar
