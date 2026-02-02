import cloudpickle
import jax
import jax.numpy as jnp
import lz4.frame
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import xarray as xr
import itertools as it
import functools as ft
from matplotlib.backends.backend_pdf import PdfPages

from ltc.agents import QNetwork, StochasticVariationalNetwork
from ltc.sim.constants import Actions


# from ltc.agents.svi import

def clean_observations(observations: xr.DataArray) -> xr.DataArray:
    """
    Clean observations by capping ret_c and no_tx values.

    Args:
        observations: xarray DataArray with feature dimension

    Returns:
        Cleaned xarray DataArray
    """
    cleaned = observations.copy()

    # Cap ret_c values > 8 to 9
    ret_c_mask = cleaned.sel(feature='ret_c') > 8
    cleaned.loc[dict(feature='ret_c')] = xr.where(ret_c_mask, 9,
        cleaned.sel(feature='ret_c'))

    # Cap no_tx values > 45 to 45
    no_tx_mask = cleaned.sel(feature='no_tx') > 45
    cleaned.loc[dict(feature='no_tx')] = xr.where(no_tx_mask, 45,
        cleaned.sel(feature='no_tx'))

    return cleaned


randomization = {'ret_c': stats.geom(p=0.3, loc=-1),
                 'no_tx': stats.geom(p=0.1, loc=-1),
                 'battery': stats.uniform(0, 10000.),
                 'channel': stats.randint(-1, 2),
                 'buffer': stats.randint(1, 2)}


class Target:

    def __init__(self, file_path: str, num_samples: int = 16):
        with lz4.frame.open(file_path, "rb") as f:
            self.ddqn_state, self.history = cloudpickle.load(f)

        self.num_samples = num_samples

        self.model = StochasticVariationalNetwork(
            QNetwork(num_actions=3, num_layers=4, dim=64, num_heads=4))

        def pred(params, state, obs, root_key):
            #obs [b,t,f]
            key = map(ft.partial(jax.random.fold_in, root_key), it.count())
            rngs = {'params': next(key), 'dropout': next(key),
                    'rlib': next(key)}

            q_vals, _ = self.model.apply({'params': params, **state}, obs,
                                         rngs=rngs, mutable=['loss'])
            # acts = jnp.argmax(q_vals, axis=-1).astype(int)
            return q_vals

        self.pred = pred

    def __call__(self):
        #agent, layer...
        params, state = self.ddqn_state.params, self.ddqn_state.net_state

        observations = self.history.observations[
            -1, -1000:, ...]  # last epoch, last 1000 samples, 5 agents,1 window, 5 features
        observations = xr.DataArray(np.asarray(observations),
                                    dims=['step', 'agent', 'window',
                                          'feature'], coords={
                'feature': ['buffer', 'channel', 'ret_c', 'no_tx', 'battery']})

        for i in range(observations.shape[-1]):
            print(i, np.unique(observations[..., i], return_counts=True))

        s, a, w, f = observations.shape
        key = jax.random.key(42)
        obs = jnp.transpose(observations.data,(1,0,2,3)) # aswf


        keys = jax.random.split(key, num=a * self.num_samples)
        keys = jnp.reshape(keys, (self.num_samples, a))
        pred = jax.vmap(self.pred)
        pred = jax.vmap(pred, in_axes=(None, None, None, 0), out_axes=-1)

        #pred = jax.vmap(self.pred, in_axes=(0, 0, 1, 0), out_axes=1)
        #pred = jax.vmap(pred, in_axes=(None, None, None, 0), out_axes=-1)

        qvals = jax.jit(pred)(params, state, obs, keys)
        qvals = xr.DataArray(np.asarray(qvals),
                             dims=['agent','step',  'action', 'sample'],
                             coords={'action': [a.name for a in Actions]})


        return observations, qvals  # q_vals, _ = self.model.apply(  #     {"params": params_0, **state_0},  #     observations,  #     rngs=jax.random.key(42),  #     mutable=["loss"],  # )  # return observations, q_vals


if __name__ == '__main__':
    # TODO: Change the path to your data file
    target = Target("history_5_5_42.pkl.lz4", num_samples=128)
    observations, qvals = target()
    observations = clean_observations(observations)

    ds = xr.Dataset({'observations': observations, 'qvalues': qvals})

    ds.to_netcdf("qnet.nc", engine="netcdf4")

    fig, axs = plt.subplots(5, 1, figsize=(10, 12))
    colors = plt.cm.tab10(np.linspace(0, 1, observations.agent.size))

    with PdfPages("histogram.pdf") as pdf:

        for i, feature in enumerate(observations.feature.values):
            for agent_idx, agent in enumerate(observations.agent.values):
                data = observations.sel(feature=feature,
                                        agent=agent).values.flatten()
                axs[i].hist(data, bins=120, alpha=0.5, color=colors[agent_idx],
                            label=f'Agent {agent}')
            axs[i].set_title(f'Feature: {feature}')
            axs[i].set_xlabel('Value')
            axs[i].set_ylabel('Frequency')
            if i == 0:
                axs[i].legend(loc='upper right')

        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.show()

        for sharex in [True, False]:

            fig, axs = plt.subplots(qvals.agent.size, 1, figsize=(8.27,
                                                                  11.69 * qvals.agent.size / 5),
                                    sharex=sharex)
            colors = plt.cm.tab10(np.linspace(0, 1, qvals.action.size))

            for agent_idx, agent in enumerate(qvals.agent.values):
                ax = axs[agent_idx] if qvals.agent.size > 1 else axs
                for action_idx, action in enumerate(qvals.action.values):
                    data = qvals.sel(agent=agent,
                                     action=action).values.flatten()
                    ax.hist(data, bins=20, alpha=0.5, color=colors[action_idx],
                            label=f'Action: {action}')
                ax.set_title(f'Agent {agent}')
                ax.set_xlabel('Q-value')
                ax.set_ylabel('Frequency')
                ax.legend(loc='upper right')

            plt.tight_layout()
            pdf.savefig(bbox_inches='tight')
            plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, qvals.action.size))

        # Normalize Q-values across all agents and samples for each action
        for action_idx, action in enumerate(qvals.action.values):
            data = qvals.sel(action=action).values.flatten()
            # Normalize
            data_normalized = (data - data.mean()) / data.std()
            ax.hist(data_normalized, bins=130, alpha=0.5,
                    color=colors[action_idx], label=f'Action: {action}')

        ax.set_title('Normalized Q-values (All Agents)')
        ax.set_xlabel('Normalized Q-value')
        ax.set_ylabel('Frequency')
        ax.legend(loc='upper right')

        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.show()
