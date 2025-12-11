import dataclasses
import os
from typing import Callable, NamedTuple

from ltc.agents.dcf import DCF
from ltc.sim.traffic import InitialStateConf, cox_traffic

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import cloudpickle
import jax
import jax.numpy as jnp

import lz4.frame


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


class Sim:
    init: Callable
    step: Callable

    def __init__(
        self,
        args,
    ):
        n = args.n
        n_drl = args.n_drl
        n_epochs = args.n_epochs
        n_steps = args.n_steps
        window_size = args.window_size
        seed = args.seed
        traffic_type = args.traffic_type

        key = jax.random.key(seed)
        num_actions = len(Actions)
        actions = jnp.zeros(n, dtype=int)
        buffer_states = jnp.zeros(n, dtype=int)
        power_states = jnp.full(n, INITIAL_CAPACITY, dtype=int)
        channel_state = 0
        #s,t,f
        obs = jnp.zeros((n, window_size, 5), dtype=int).at[:, -1].set(INITIAL_CAPACITY)
        rewards = jnp.zeros(n)
        terminals = jnp.full(n, False, dtype=bool)

        drl = SymbolicAgents(
            n_drl, obs_space_shape=obs.shape[1:], act_space_size=num_actions
        )

        dcf = DCF()

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

        def init(key: jax.Array) -> Carry:
            key, init_key = jax.random.split(key)
            drl_states, drl_step = init_agents(drl, init_key, n_drl)
            drl_states = drl_states.replace(
                prev_env_state=drl_states.prev_env_state.astype(int)
            )

            key, init_key = jax.random.split(key)
            legacy_states, legacy_step = init_agents(dcf, init_key, n - n_drl)

            key, init_key = jax.random.split(key)
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

        # self.action_shape = jax.eval_shape(pre_rl_fn, init(),None)

        def step(carry: Carry, action: jax.Array) -> tuple[Carry, Output]:
            key, init_key = jax.random.split(carry.key)
            drl_states, drl_step = init_agents(drl, init_key, n_drl)
            drl_states = drl_states.replace(
                prev_env_state=drl_states.prev_env_state.astype(int)
            )

            key, init_key = jax.random.split(key)
            _, legacy_step = init_agents(dcf, init_key, n - n_drl)

            key, init_key = jax.random.split(key)
            _, traffic_step = init_traffic(traffic, init_key, n)

            _, _, post_rl_fn = rl_step(
                drl_step, legacy_step, traffic_step, n, n_drl
            )
            c = dataclasses.replace(carry, key=key)
            # Intenals is a tuple of states and actions
            c, o = post_rl_fn((c.drl_states, action), c, None)

            return c, o

        def shape_fn(c):
            def _f(c):
                key, init_key = jax.random.split(c.key)
                drl_states, drl_step = init_agents(drl, init_key, n_drl)
                drl_states = drl_states.replace(
                    prev_env_state=drl_states.prev_env_state.astype(int)
                )

                key, init_key = jax.random.split(key)
                _, legacy_step = init_agents(dcf, init_key, n - n_drl)

                key, init_key = jax.random.split(key)
                _, traffic_step = init_traffic(traffic, init_key, n)

                _, pre_rl_fn, _ = rl_step(
                    drl_step, legacy_step, traffic_step, n, n_drl
                )
                a = pre_rl_fn(c, None)
                return a

            return jax.eval_shape(_f, c)

        self.init = jax.jit(init)
        self.step = jax.jit(step)
        self.shape_fn = shape_fn

@dataclasses.dataclass
class Results:
    n: int
    n_drl: int
    seed: int
    all_results: list = dataclasses.field(default_factory=list)

    def append(self, result):
        self.all_results.append(result)

    def save(self, path: str):
        all_results = jax.tree.map(lambda *x: jnp.stack(x), *self.all_results)
        filename = f'history_{self.n}_{self.n_drl}_{self.seed}.pkl.lz4'
        with lz4.frame.open(filename, 'wb') as f:
            cloudpickle.dump((None, all_results), f)