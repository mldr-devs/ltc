from pysr import PySRRegressor

import cloudpickle
import jax
import jax.numpy as jnp
import lz4.frame
import numpy as np
import pandas as pd

from ltc.sim.constants import Actions
from .regressor import Regressor

from ltc.agents import QNetwork, StochasticVariationalNetwork
import xarray as xr

# from ltc.agents.svi import


class Target:

    def __init__(self, file_path: str, num_samples: int = 16):
        with lz4.frame.open(file_path, "rb") as f:
            self.ddqn_state, self.history = cloudpickle.load(f)

        self.num_samples = num_samples

        self.model = StochasticVariationalNetwork(
            QNetwork(num_actions=3, num_layers=4, dim=64, num_heads=4))

        def pred(params, state, obs, key):
            q_vals, _ = self.model.apply({'params': params, **state},
                                         obs, rngs=key,
                                         mutable=['loss'])
            # acts = jnp.argmax(q_vals, axis=-1).astype(int)
            return q_vals
        self.pred = pred
        # many_pred = jax.vmap(pred, in_axes=(None, None, None, 0))
        #
        # _pred = jax.vmap(many_pred, in_axes=(None, None, 0, 0))
        # self.pred = jax.jit(_pred)

    def __call__(self):
        # n_epochs, n_steps, n_agents, window_size, n_features = history.observations.shape
        # features: buffer_state (0 or 1), channel_state (0 - idle, 1 - busy, -1 - unknown), retransmission_counter (N+), num_of_slots_with_no_tx (N+), power_left (0 - INITIAL_CAPACITY)

        # params: agent, params...

        params, state = self.ddqn_state.params, self.ddqn_state.net_state
        # params_0, state_0 = jax.tree.map(lambda x: x[0], (params,
        #                                                   state))  # take the parameters and state of the first agent
        # observations_ = self.history.observations[
        #     -1, :, 0]  # last epoch, all steps, first agent

        observations = self.history.observations[-1,-500:,...] # 500 samples, 5 agents,1 window, 5 features
        observations = xr.DataArray(observations,
                                    dims=['step','agent','window','feature'],
                                    coords={'feature':['buffer','channel','ret_c','no_tx','battery']})
        s,a,w,f = observations.shape
        key = jax.random.key(42)

        keys = jax.random.split(key, num=a*self.num_samples)
        keys = jnp.reshape(keys, ( self.num_samples,a))

        pred = jax.vmap(self.pred, in_axes=(0,0,1,0), out_axes=1)
        pred = jax.vmap(pred, in_axes=(None,None,None,0), out_axes=-1)


        qvals=jax.jit(pred)(params, state, observations.data, keys)
        qvals = xr.DataArray(qvals,dims=['step','agent','action','sample'],
                             coords={'action':[a.name for a in Actions]})


        # pred_fn = jax.vmap(self.pred, in_axes=(None, None, 0, 0))
        # qvals = self.pred(params_0, state_0, observations, keys)
        return observations, qvals  # q_vals, _ = self.model.apply(  #     {"params": params_0, **state_0},  #     observations,  #     rngs=jax.random.key(42),  #     mutable=["loss"],  # )  # return observations, q_vals




if __name__ == '__main__':
    # TODO: Change the path to your data file
    target = Target("history_5_5_42.pkl.lz4")
    observations, qvals = target()

    observations.to_netcdf("observations.nc")
    qvals.to_netcdf("qvalues.nc")


    # b,t,f = observations.shape
    #
    # fo = np.reshape(observations,(b,t*f))
    # COLUMNS = 'buffer,channel,ret_c,no_tx,battery'.split(',')
    # assert len(COLUMNS) == f
    # C = []
    # for i in range(t):
    #     for col in COLUMNS:
    #         C.append(f'{col}_{i}')
    #
    # hdf = pd.DataFrame(fo, columns=C)
    #
    # model = Regressor()
    #
    # # model.fit(hdf, qvals)
    #
    # def _pyjax_fit(X,Y):
    #     hdf = pd.DataFrame(X,columns=COLUMNS)
    #     model.fit(hdf, Y)
    #
    #
    # @jax.jit
    # def jax_fit(X,Y):
    #     print('jit')
    #     return jax.experimental.io_callback(_pyjax_fit, None, X, Y)
    #
    # jax_fit(jnp.asarray(fo), jnp.asarray(qvals))
    # print(model)
