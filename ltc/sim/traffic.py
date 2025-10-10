from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.signal as signal
from jax import tree

from ltc.sim.constants import TAU


@jax.tree_util.register_dataclass
@dataclass
class ModelState:
    x: jax.Array


@jax.tree_util.register_dataclass
@dataclass
class ModelStats:
    mean: float
    variance: float
    acf_lag1: float

    @property
    def fano_factor(self) -> float:
        """https://en.wikipedia.org/wiki/Fano_factor"""
        return self.variance / self.mean if self.mean != 0 else np.inf


class TrafficModel(NamedTuple):
    init: Callable[[jax.Array], ModelState]
    sample: Callable[[ModelState, jax.Array], Tuple[ModelState, jax.Array]]
    stats: Callable[[], ModelStats]


def _ss_step(
    x: jax.Array, u: jax.Array, A: jax.Array, B: jax.Array, C: jax.Array, D: jax.Array
) -> tuple[jax.Array, jax.Array]:
    x_next = A @ x + B * u
    y = C @ x + D * u
    return y.squeeze(), x_next


class InitialStateConf(Enum):
    ZERO = 0
    NORMAL = 1


def cox_traffic(
    f3dB: float,
    order: int = 1,
    loc: float = 0.0,
    scale: float = 1.0,
    initial_state=InitialStateConf.ZERO,
) -> TrafficModel:
    """
    Model factory.

    This function creates a pair of functions used to initialize and sample form traffic model.
    The model is a Cox process.
    For each TXOP the number of frames is drawn from Poisson distribution whose rate parameter is controlled by the filtered white noise.

    For filtration, we use Butterworth filter. The :param f3dB: controls the rage of autocompletion in a traffic pattern (lower f results in longer dependence).

    :param f3dB: Cutoff frequency, suggested values is `1/(k TAU)` where `k` is the number of steps we expect an explosive traffic pattern.
    :param order: Filter order
    :param loc: Noise location
    :param scale: Noise scale
    :param initial_state: How initial state is constructed
    :return: Model
    """

    fs = 1 / TAU
    nyquist = fs / 2
    normalized_cutoff = f3dB / nyquist
    b, a = signal.butter(order, normalized_cutoff, btype="low", analog=False)

    A, B, C, D = tree.map(jnp.asarray, signal.tf2ss(b, a))
    n_states = A.shape[0]

    # for scalar signals
    B, C, D = tree.map(lambda x: x.flatten(), (B, C, D))

    def init(key: jax.Array) -> ModelState:
        match initial_state:
            case InitialStateConf.ZERO:
                return ModelState(x=jnp.zeros((n_states,)))
            case InitialStateConf.NORMAL:
                return ModelState(x=jax.random.normal(key, (n_states,)))
            case _:
                raise Exception(f"Unknown initial_state {initial_state}")

    def step(state: ModelState, key: jax.Array) -> tuple[ModelState, jax.Array]:
        state_key, obs_key = jax.random.split(key)
        u = loc + scale * jax.random.normal(state_key)
        y, x = _ss_step(state.x, u, A, B, C, D)
        n = jax.random.poisson(obs_key, lam=jnp.exp(y))
        return ModelState(x=x), n

    def stats() -> ModelStats:
        n = int(fs/normalized_cutoff) # time to reach steady state
        #Numerator (`b`) and denominator (`a`)
        t, (y,) = signal.dimpulse(system=(b,a,1),n=n)

        log_rate_loc = loc*y.sum()
        log_rate_var = np.sum(scale*np.square(y))
        # https://en.wikipedia.org/wiki/Log-normal_distribution
        rate_loc = np.exp(log_rate_loc+0.5*log_rate_var)
        rate_var = (np.exp(log_rate_var)-1)*np.exp(2*log_rate_loc+log_rate_var)

        # https://en.wikipedia.org/wiki/Law_of_total_expectation
        # https://en.wikipedia.org/wiki/Law_of_total_variance

        count_mean = rate_loc
        count_var = rate_loc + rate_var


        log_rate_lag_1_acv = scale**2 * np.dot(y[:-1,0],y[1:,0])
        rate_acv = np.exp(2*log_rate_loc + log_rate_var)*(np.exp(log_rate_lag_1_acv)-1)
        count_lag_1_acv = rate_acv
        count_lag_1_acf = count_lag_1_acv/count_var

        return ModelStats(mean=count_mean, variance=count_var, acf_lag1=count_lag_1_acf)

    return TrafficModel(init, step, stats)
