
from dataclasses import dataclass

from enum import Enum
from typing import Tuple,Callable, NamedTuple

import scipy.signal as signal
import jax

import jax.numpy as jnp
from jax import tree

from ltc.sim.constants import TAU


@jax.tree_util.register_dataclass
@dataclass
class ModelState:
    x: jax.Array


class TrafficModel(NamedTuple):
    init: Callable[[jax.Array],ModelState]
    sample: Callable[[ModelState, jax.Array], Tuple[ModelState,jax.Array]]

def _ss_step(x:jax.Array,u:jax.Array, A:jax.Array, B:jax.Array, C:jax.Array, D:jax.Array)->tuple[jax.Array, jax.Array]:
    x_next = A @ x + B * u
    y = C @ x + D * u
    return y.squeeze(), x_next

class InitialStateConf(Enum):
    ZERO=0
    NORMAL=1

def cox_traffic(f3dB: float, order:int=1, loc:float=0., scale:float=1.0, initial_state=InitialStateConf.ZERO)->TrafficModel:
    """ Model factory.

    This function creates a pair of functions used to initialize and sample form traffic model.
    The model is a Cox process.
    For each txop the number of frames is drawn from Poisson distribution whose rate parameter is controlled by the filtered white noise.

    For filtration, we use Butterworth filter. The :param f3dB: controls the rage of autocompletion in a traffic pattern (lower f results in longer dependence).

    :param f3dB: Cutoff frequency, suggested values is `1/(k TAU)` where `k` is the number of steps we expect an explosive traffic pattern.
    :param order: Filter order
    :param loc: Noise location
    :param scale: Noise scale
    :param initial_state: How initial state is constructed
    :return: Model
    """
    fs = 1/TAU
    nyquist = fs / 2
    normalized_cutoff = f3dB / nyquist
    b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)


    A, B, C, D = tree.map(jnp.asarray,signal.tf2ss(b, a))
    n_states = A.shape[0]

    # for scalar signals
    B, C, D = tree.map(lambda x: x.flatten(),(B,C,D))


    def init(key:jax.Array):
        match initial_state:
            case InitialStateConf.ZERO:
                return ModelState(x=jnp.zeros((n_states,)))
            case InitialStateConf.NORMAL:
                return ModelState(x=jax.random.normal(key, (n_states,)))
            case _:
                raise Exception(f'Unknow initial_state {initial_state}')

    def step(state:ModelState, key:jax.Array)->tuple[ModelState,jax.Array]:
        state_key, obs_key = jax.random.split(key)
        u = loc+ scale*jax.random.normal(state_key)
        y,x = _ss_step(state.x,u, A,B,C,D )
        n = jax.random.poisson(obs_key,lam=jnp.exp(y))
        return ModelState(x=x),n

    return TrafficModel(init, step)


def generate_frames(n, t):
    # TODO implement
    return jnp.zeros(n, dtype=int)
