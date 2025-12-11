import argparse
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from chex import Shape, Scalar, Array


from ltc.agents import DCF
from ltc.run import init_agents, init_traffic, rl_step
from ltc.sim import cox_traffic, InitialStateConf
from ltc.sim.constants import Actions, INITIAL_CAPACITY
from ltc.utils.scan_states import Carry


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class SymbolicAgenState:
    params: dict
    prev_env_state: jax.Array
    replace = dataclasses.replace


class SymbolicAgents:
    def __init__(self, num_regressors, obs_space_shape: Shape,
                 act_space_size: int,
                 experience_replay_buffer_size: int = 10000,
                 experience_replay_batch_size: int = 64,
                 experience_replay_steps: int = 5, discount: Scalar = 0.99,
                 epsilon: Scalar = 1.0, epsilon_decay: Scalar = 0.999,
                 epsilon_min: Scalar = 0.001, tau: Scalar = 0.01):
        self.args = locals()
        self.n = num_regressors
        self.obs_space_shape = obs_space_shape if jnp.ndim(
            obs_space_shape) > 0 else (obs_space_shape,)
        self.act_space_size = act_space_size

        def init(key: jax.Array) -> SymbolicAgentState:
            # DUmmy params to simulate dqn params
            return SymbolicAgentState(params=dict(x=jnp.zeros((2,))),
                                     prev_env_state=jnp.zeros(obs_space_shape))

        def update(state: SymbolicAgentState, key: jax.Array, env_state: Array,
                   action: Array, reward: Scalar,
                   terminal: bool, ) -> SymbolicAgentState:
            return state

        def sample(state: SymbolicAgentState, key: jax.Array, X) -> any:
            # precaution to avoid DCE
            return jnp.zeros((), jnp.int32)

        def _pyjax_sample(env_state, i):
            print('sampling agent', i)

            reg = self.agents[int(i)]
            try:
                q = reg.predict(env_state.reshape(1, -1))
            except:
                q = np.random.normal(size=(self.act_space_size,))
            return jnp.asarray(q)

        self.init = jax.jit(init)
        self.update = jax.jit(update)
        self.sample = jax.jit(sample)


def main():
    parser = argparse.ArgumentParser(
        description="Run the RL network simulation with configurable parameters.")
    parser.add_argument('--n', type=int, default=5,
                        help='Total number of agents in the simulation.')
    parser.add_argument('--n_drl', type=int, default=5,
                        help='Number of DRL agents.')
    parser.add_argument('--n_epochs', type=int, default=50,
                        help='Number of training epochs to run.')
    parser.add_argument('--n_steps', type=int, default=5000,
                        help='Number of steps per epoch.')
    parser.add_argument('--window_size', type=int, default=1,
                        help='Size of the observation window for each agent.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--save-plots', action='store_true', default=False,
                        help='Whether to save the generated plots.')
    parser.add_argument('--traffic_type', type=str, default='saturated',
                        choices=['constant', 'saturated', 'bursty'],
                        help="Traffic model to use: 'constant', 'saturated', or 'bursty'.")
    args = parser.parse_args()

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
    obs = jnp.zeros((n, window_size, 5), dtype=int).at[:, -1].set(
        INITIAL_CAPACITY)
    rewards = jnp.zeros(n)
    terminals = jnp.full(n, False, dtype=bool)

    sq = SymbolicAgents(n_drl, obs_space_shape=obs.shape[1:],
                        act_space_size=num_actions)
    key, init_key = jax.random.split(key)
    drl_states, drl_step = init_agents(sq, init_key, n_drl)
    drl_states = drl_states.replace(
        prev_env_state=drl_states.prev_env_state.astype(int))

    key, init_key = jax.random.split(key)

    if traffic_type == 'constant':
        traffic = cox_traffic(f3dB=1.0, loc=-1.0, scale=0.0,
                              initial_state=InitialStateConf.ZERO)
    elif traffic_type == 'saturated':
        traffic = cox_traffic(f3dB=1.0, loc=5.0, scale=0.0,
                              initial_state=InitialStateConf.ZERO)
    elif traffic_type == 'bursty':
        traffic = cox_traffic(f3dB=0.1, loc=-5.0, scale=5.0,
                              initial_state=InitialStateConf.ZERO)
    else:
        raise ValueError(f'Unknown traffic type: {traffic_type}')

    traffic_states, traffic_step = init_traffic(traffic, init_key, n)

    dcf = DCF()
    key, init_key = jax.random.split(key)
    legacy_states, legacy_step = init_agents(dcf, init_key, n - n_drl)

    rl_step_fn, _, _ = rl_step(drl_step, legacy_step, traffic_step, n, n_drl)
    init_carry = Carry(drl_states, legacy_states, traffic_states,
        buffer_states, power_states, channel_state, key, obs, actions, rewards,
        terminals)
    new_carry = rl_step_fn(init_carry, 0)

    ...

    ...
    return new_carry


if __name__ == "__main__":
    main()
