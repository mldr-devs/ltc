import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"


import jax.numpy as jnp


from ltc.sim.constants import INITIAL_CAPACITY, Actions  # noqa: F401
from ltc.utils.scan_states import Carry, Output  # noqa: F401


from ltc.run import rl_step as rl_step
from ltc.run import setup_args as setup_args

_len = len


def make_obs(window_size, n):
    obs = (
        jnp.zeros((n, window_size, 5), dtype=jnp.int32).at[:, -1].set(INITIAL_CAPACITY)
    )
    return obs


from ltc.symbolic.run import main  # noqa: E402, F401
