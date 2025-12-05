import os
from typing import Callable, NamedTuple

from ltc.agents.dcf import DCF
from ltc.sim.traffic import InitialStateConf, cox_traffic

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"


import jax
import jax.numpy as jnp


from ltc.sim.constants import INITIAL_CAPACITY, Actions  # noqa: F401
from ltc.utils.scan_states import Carry, Output  # noqa: F401


from ltc.run import init_agents, init_traffic, rl_step as rl_step
from ltc.run import setup_args as setup_args

_len = len


def make_obs(window_size, n):
    obs = (
        jnp.zeros((n, window_size, 5), dtype=jnp.int32).at[:, -1].set(INITIAL_CAPACITY)
    )
    return obs


from ltc.symbolic.run import SymbolicAgents, main  # noqa: E402, F401


class Sim(NamedTuple):
    init: Callable
    update: Callable


def make_sim(
    n,
    n_drl,
    traffic_type,
    channel_state,
    obs,
    actions,
    rewards,
    terminals,
    buffer_states,
    power_states,
) -> Sim:
    num_actions = len(Actions)

    def init(key):
        key, init_key = jax.random.split(key)
        drl = SymbolicAgents(
            n_drl, obs_space_shape=obs.shape[1:], act_space_size=num_actions
        )
        drl_states, drl_step = init_agents(drl, init_key, n_drl)
        drl_states = drl_states.replace(
            prev_env_state=drl_states.prev_env_state.astype(int)
        )

        dcf = DCF()
        key, init_key = jax.random.split(key)
        legacy_states, legacy_step = init_agents(dcf, init_key, n - n_drl)

        key, init_key = jax.random.split(key)

        if traffic_type == "constant":
            traffic = cox_traffic(
                f3dB=1.0, loc=-1.0, scale=0.0, initial_state=InitialStateConf.ZERO
            )
        elif traffic_type == "saturated":
            traffic = cox_traffic(
                f3dB=1.0, loc=5.0, scale=0.0, initial_state=InitialStateConf.ZERO
            )
        elif traffic_type == "bursty":
            traffic = cox_traffic(
                f3dB=0.1, loc=-5.0, scale=5.0, initial_state=InitialStateConf.ZERO
            )
        else:
            raise ValueError(f"Unknown traffic type: {traffic_type}")

        traffic_states, traffic_step = init_traffic(traffic, init_key, n)
        init_carry = Carry(
            drl_states,
            legacy_states,
            traffic_states,
            buffer_states,
            power_states,
            channel_state,
            key,
            obs,
            actions,
            rewards,
            terminals,
        )
        return init_carry

    def update(carry, action):
        pass

    return Sim(init, update)
