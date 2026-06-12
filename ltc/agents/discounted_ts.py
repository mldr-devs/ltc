from functools import partial

import jax
import jax.numpy as jnp
from chex import PRNGKey, Scalar

from reinforced_lib.agents.mab import NormalThompsonSampling
from reinforced_lib.agents.mab.normal_thompson_sampling import NormalThompsonSamplingState


class DiscountedThompsonSampling(NormalThompsonSampling):
    def __init__(self, n_arms: int, alpha: Scalar, beta: Scalar, lam: Scalar, mu: Scalar, gamma: Scalar) -> None:
        super().__init__(n_arms, alpha, beta, lam, mu)
        self.update = jax.jit(partial(self._update, gamma=gamma))

    @staticmethod
    def _update(
            state: NormalThompsonSamplingState,
            key: PRNGKey,
            action: int,
            reward: Scalar,
            gamma: Scalar,
    ) -> NormalThompsonSamplingState:
        lam = state.lam * gamma
        alpha = state.alpha * gamma
        beta = state.beta * gamma
        l = lam[action]
        mu = state.mu[action]
        return NormalThompsonSamplingState(
            alpha=alpha.at[action].add(0.5),
            beta=beta.at[action].add((l * jnp.square(reward - mu)) / (2 * (l + 1))),
            lam=lam.at[action].add(1),
            mu=state.mu.at[action].set((mu * l + reward) / (l + 1)),
        )
