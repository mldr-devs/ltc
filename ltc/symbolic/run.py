import argparse
import dataclasses
from abc import abstractmethod

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from reinforced_lib.agents import BaseAgent

from ltc.symbolic.nn2sym import Target
from ltc.sim.constants import Actions, INITIAL_CAPACITY
from ltc.symbolic.regressor import Regressor

@jax.tree_util.register_dataclass
@dataclasses.dataclass
class SymbolicAgenState:
    dummy: jax.Array

class SymbolicAgents:
    def __init__(self, n):
        self.n = n
        self.agents = [Regressor() for _ in range(n)]


        def init(key:jax.Array)->SymbolicAgenState:
            return SymbolicAgenState(key)

        @jax.custom_batching.custom_vmap
        def update(state: SymbolicAgenState, key: jax.Array, X,y) -> SymbolicAgenState:
            return state

        def _pyjax_fit(X, Y,i):
            print('fitting agent',i)
            return 0.0
            #hdf = pd.DataFrame(X)
            #self.agents[int(i)].fit(hdf, Y)

        @update.def_vmap
        def update_vmap(axis_size, in_batched,state: SymbolicAgenState, key, X,y) -> SymbolicAgenState:
            for i in range(axis_size):
                jax.experimental.io_callback(_pyjax_fit, jax.ShapeDtypeStruct((),jnp.float32), X[i], y[i], i)
            return state,in_batched[0]


        self.init = jax.jit(init)
        self.update = jax.jit(update)


    # @staticmethod
    # @abstractmethod
    # def init(key) -> SymbolicAgenState:
    #     """
    #     Creates and initializes instance of the agent.
    #     """
    #
    #     return SymbolicAgenState()

    # @staticmethod
    # @abstractmethod
    # def update(state: SymbolicAgenState, key: jax.Array, *args, **kwargs) -> SymbolicAgenState:
    #     """
    #     Updates the state of the agent after performing some action and receiving a reward.
    #     """
    #
    #     pass

    @staticmethod
    @abstractmethod
    def sample(state: SymbolicAgenState, key: jax.Array, *args, **kwargs) -> any:
        """
        Selects the next action based on the current environment and agent state.
        """

        pass





@jax.custom_batching.custom_vmap
def predict(symbolic_agents: SymbolicAgents, observations):
    pass

@predict.def_vmap
def predict_vmap(axis_size, in_batched,symbolic_agents: SymbolicAgents, observations):
    pass

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

    agents = SymbolicAgents(1)
    state = jax.vmap(agents.init)(jax.random.split(jax.random.key(5),3))

    X=jnp.ones((3,4,5))
    y = jnp.ones((3,4))

    next_states = jax.jit(jax.vmap(agents.update))(state,
                            jax.random.split(jax.random.key(6),3), X,y
                            )


    def _pyjax_fit(X,Y):
        hdf = pd.DataFrame(X,columns=COLUMNS)
        agents.agents[0].fit(hdf, Y)
        return jnp.zeros(())



    @jax.custom_batching.custom_vmap
    def jax_fit(X,Y):
        print('jit')
        return jax.experimental.io_callback(_pyjax_fit, jax.ShapeDtypeStruct((),jnp.float32), X, Y)


    @jax_fit.def_vmap
    def jax_fit_vmap(axis_size, in_batched,X,Y):
        print('vmap')
        return jnp.stack([jax.experimental.io_callback(_pyjax_fit, jax.ShapeDtypeStruct((),jnp.float32), _X, _Y)
                          for _X,_Y in zip(X,Y)]),True

    jax.jit(jax.vmap(jax_fit))(jnp.expand_dims(jnp.asarray(fo),0),
                     jnp.expand_dims(jnp.asarray(qvals),0)
    )




    print("Model saved and agents updated.")

    ...
