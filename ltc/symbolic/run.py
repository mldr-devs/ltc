import argparse
import dataclasses
from abc import abstractmethod

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from chex import Shape, Scalar, Array
from reinforced_lib.agents import BaseAgent
from reinforced_lib.utils.experience_replay import ReplayBuffer, \
    experience_replay

from ltc.agents import BayesianDDQN, StochasticVariationalNetwork, QNetwork, \
    DCF
from ltc.run import init_agents, init_traffic, rl_step
from ltc.sim import cox_traffic, InitialStateConf
from ltc.symbolic.nn2sym import Target
from ltc.sim.constants import Actions, INITIAL_CAPACITY
from ltc.symbolic.regressor import Regressor
from ltc.utils.scan_states import Carry


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class SymbolicAgenState:
    dummy: jax.Array # delme?
    replay_buffer: ReplayBuffer
    prev_env_state: jax.Array
    epsilon: jax.Array

    replace = dataclasses.replace

class SymbolicAgents:
    def __init__(self, num_regressors,
                 obs_space_shape: Shape,
                 act_space_size: int,
                 experience_replay_buffer_size: int = 10000,
                 experience_replay_batch_size: int = 64,
                 experience_replay_steps: int = 5,
                 discount: Scalar = 0.99,
                 epsilon: Scalar = 1.0,
                 epsilon_decay: Scalar = 0.999,
                 epsilon_min: Scalar = 0.001,
                 tau: Scalar = 0.01
                 ):
        self.args = locals()
        self.n = num_regressors
        self.agents = [Regressor() for _ in range(n)]

        assert experience_replay_buffer_size > experience_replay_batch_size > 0
        assert 0.0 <= discount <= 1.0
        assert 0.0 <= epsilon <= 1.0
        assert 0.0 <= epsilon_decay <= 1.0
        assert 0.0 <= epsilon_min <= epsilon
        assert 0.0 <= tau <= 1.0
        self.obs_space_shape = obs_space_shape if jnp.ndim(obs_space_shape) > 0 else (obs_space_shape,)
        self.act_space_size = act_space_size
        er = experience_replay(
            experience_replay_buffer_size,
            experience_replay_batch_size,
            self.obs_space_shape,
            (1,)
        )
        def init(key: jax.Array) -> SymbolicAgenState:
            replay_buffer = er.init()
            return SymbolicAgenState(key, replay_buffer=replay_buffer,
                                     prev_env_state=jnp.zeros(obs_space_shape),
                                     epsilon=epsilon)

        #obs:int32[1,5], action:int32[], reward:float32[], terminal:bool[]
        @jax.custom_batching.custom_vmap
        def update(state: SymbolicAgenState,
                   key: jax.Array,
                   env_state: Array,
                   action: Array,
                   reward: Scalar,
                   terminal: bool,) -> SymbolicAgenState:
            # Agent is not designed for vmaped
            return state

        def _pyjax_fit(batch,i):
            print('fitting agent',i)
            return 0.0
            #hdf = pd.DataFrame(X)
            #self.agents[int(i)].fit(hdf, Y)



        @update.def_vmap
        def update_vmap(axis_size, in_batched,state: SymbolicAgenState, key, env_state,action,reward,terminal) -> SymbolicAgenState:
            replay_buffer = jax.vmap(er.append)(state.replay_buffer,
                                      state.prev_env_state, action, reward,
                                      terminal, env_state)
            #TODO split key, pass seed to pysr
            states, actions, rewards, terminals, next_states = jax.vmap(er.sample)(replay_buffer, key)

            for i in range(axis_size):
                args = (states[i], actions[i], rewards[i], terminals[i], next_states[i])
                jax.experimental.io_callback(_pyjax_fit, jax.ShapeDtypeStruct((),jnp.float32), args, i)
            next_state = SymbolicAgenState(
                dummy=state.dummy,
                replay_buffer=replay_buffer,
                prev_env_state=env_state,
                epsilon=jnp.maximum(state.epsilon * self.args['epsilon_decay'], self.args['epsilon_min'])
            )
            return next_state,in_batched[0]

        @jax.custom_batching.custom_vmap
        def sample(state: SymbolicAgenState, key: jax.Array, X) -> any:
            # precaution to avoid DCE
            return jnp.zeros((act_space_size,), jnp.int32)

        def _pyjax_sample(env_state,i):
            print('sampling agent',i)
            return jnp.asarray(act_space_size*[0])

        @sample.def_vmap
        def sample_vmap(axis_size, in_batched,state: SymbolicAgenState, key, env_state):
            s = [ jax.experimental.io_callback(_pyjax_sample, jax.ShapeDtypeStruct((act_space_size,),jnp.int32), env_state[i],i)
                  for i in range(axis_size)
                ]
            q = jnp.stack(s)

            @jax.vmap
            def _acc_fun(q,k,e):
                max_q = (q == q.max()).astype(float)
                probs = (1 - e) * max_q / jnp.sum(
                    max_q) + e / q.shape[0]
                return jax.random.choice(k, act_space_size,
                                  p=probs.flatten())

            return _acc_fun(q,key, state.epsilon), True

        self.init = jax.jit(init)
        self.update = jax.jit(update)
        self.sample = jax.jit(sample)




if __name__ == "__main__":
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

    # target = Target("history_5_5_42.pkl.lz4")
    # observations, qvals = target()
    #
    # fo = np.squeeze(observations)
    # COLUMNS = 'buffer,channel,ret_c,no_tx,batter'.split(',')
    # hdf = pd.DataFrame(fo, columns=COLUMNS)

    agents = SymbolicAgents(n_drl,obs_space_shape=obs.shape[1:],act_space_size=num_actions)
    state = jax.vmap(agents.init)(jax.random.split(jax.random.key(5),n_drl))

    X=jnp.ones((n_drl,4,5))
    y = jnp.ones((n_drl,4))

    next_states = jax.jit(jax.vmap(agents.update))(state,
                            jax.random.split(jax.random.key(6),n_drl),
                            env_state=obs,
                            action=jnp.zeros((n_drl,)),reward=rewards,terminal=terminals,
                            )
    sampled = jax.jit(jax.vmap(agents.sample))(next_states,
                            jax.random.split(jax.random.key(7),n_drl), X
                                               )

    key, init_key = jax.random.split(key)
    drl_states, drl_step = init_agents(agents, init_key, n_drl)

    # drl = BayesianDDQN(
    #     q_network=StochasticVariationalNetwork(
    #         QNetwork(num_actions, num_layers=2, dim=4, num_heads=2)),
    #     obs_space_shape=obs.shape[1:],
    #     act_space_size=num_actions,
    #     optimizer=optax.adam(3e-5, b1=0.95, b2=0.95),
    #     experience_replay_buffer_size=1000,
    #     experience_replay_batch_size=128,
    #     experience_replay_steps=5,
    #     discount=1.0,
    #     epsilon=1.0,
    #     epsilon_decay=0.999,
    #     epsilon_min=0.001,
    #     tau=0.01
    # )
    sq = SymbolicAgents(n_drl,obs_space_shape=obs.shape[1:],act_space_size=num_actions)
    key, init_key = jax.random.split(key)
    drl_states, drl_step = init_agents(sq, init_key, n_drl)
    drl_states = drl_states.replace(
        prev_env_state=drl_states.prev_env_state.astype(int))


    key, init_key = jax.random.split(key)

    if traffic_type == 'constant':
        traffic = cox_traffic(f3dB=1.0, loc=-1.0, scale=0.0, initial_state=InitialStateConf.ZERO)
    elif traffic_type == 'saturated':
        traffic = cox_traffic(f3dB=1.0, loc=5.0, scale=0.0, initial_state=InitialStateConf.ZERO)
    elif traffic_type == 'bursty':
        traffic = cox_traffic(f3dB=0.1, loc=-5.0, scale=5.0, initial_state=InitialStateConf.ZERO)
    else:
        raise ValueError(f'Unknown traffic type: {traffic_type}')

    traffic_states, traffic_step = init_traffic(traffic, init_key, n)

    dcf = DCF()
    key, init_key = jax.random.split(key)
    legacy_states, legacy_step = init_agents(dcf, init_key, n - n_drl)

    rl_step_fn = jax.jit(rl_step(drl_step, legacy_step, traffic_step, n, n_drl))
    init_carry = Carry(
        drl_states, legacy_states, traffic_states, buffer_states, power_states,
        channel_state, key, obs, actions, rewards, terminals
    )
    new_carry = rl_step_fn(init_carry,0)

    ...

    ...
