import argparse

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from ltc.symbolic.nn2sym import Target
from ltc.sim.constants import Actions, INITIAL_CAPACITY
from ltc.symbolic.regressor import Regressor


@jax.tree_util.register_pytree_node_class
class SymbolicAgents:
    def __init__(self, n):
        self.n = n
        self.agents = [Regressor() for _ in range(n)]

    # def __init__(self,regressors:list[Regressor]):
    #     self.n = len(regressors)
    #     self.agents = regressors

    def update(self, observations, qvals):

        pass

    def predict(self, observations):
        pass

    def tree_flatten(self):
        return (None, (self.n, self.agents))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        _, agents = aux_data
        return cls(agents)

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

    target = Target("history_5_5_42.pkl.lz4")
    observations, qvals = target()

    fo = np.squeeze(observations)
    COLUMNS = 'buffer,channel,ret_c,no_tx,batter'.split(',')
    hdf = pd.DataFrame(fo, columns=COLUMNS)

    agents = SymbolicAgents(1)


    def _pyjax_fit(X,Y):
        hdf = pd.DataFrame(X,columns=COLUMNS)
        agents.agents[0].fit(hdf, Y)



    @jax.custom_batching.custom_vmap
    def jax_fit(X,Y):
        print('jit')
        return jax.experimental.io_callback(_pyjax_fit, None, X, Y)


    @jax_fit.def_vmap
    def jax_fit_vmap(axis_size, in_batched,X,Y):
        print('vmap')
        return jax.experimental.io_callback(_pyjax_fit, None, X[0], Y[0])

    jax.jit(jax.vmap(jax_fit))(jnp.expand_dims(jnp.asarray(fo),0),
                     jnp.expand_dims(jnp.asarray(qvals),0)
    )




    print("Model saved and agents updated.")

    ...
