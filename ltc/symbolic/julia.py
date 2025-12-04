import os

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'

import argparse
from dataclasses import replace, dataclass
from functools import partial

import cloudpickle
import jax
import jax.numpy as jnp
import lz4.frame
import optax


from ltc.agents import BayesianDDQN, DCF, QNetwork, StochasticVariationalNetwork
from ltc.sim import InitialStateConf, cox_traffic, process_output, simulate
from ltc.sim.constants import INITIAL_CAPACITY, Actions
from ltc.utils.scan_states import Carry, Output
from ltc.utils.plots import plot_all, plot_first


from ltc.run import rl_step as rl_step
from ltc.run import setup_args
_len = len

def make_obs(window_size, n):
    obs = jnp.zeros((n, window_size, 5), dtype=jnp.int32).at[:, -1].set(INITIAL_CAPACITY)
    return obs

from ltc.symbolic.run import main
