import argparse
import functools as ft
import itertools as it
import math

import cloudpickle
import jax
import jax.numpy as jnp
import lz4.frame
import numpy as np
import scipy.stats as stats
import xarray as xr

from ltc.agents import QNetwork
from ltc.sim.constants import Actions
from ltc.utils.history import resolve_history_file, unpack_history


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
                 'channel': stats.randint(-1, 2),
                 # 'buffer': stats.randint(1, 2),
                 'buffer': stats.randint(0, 2),#Try empty buffer
                 'action':stats.randint(0, 2)}
F_ORDER = ['buffer', 'channel', 'ret_c', 'no_tx', 'action']
def make_xr(observations) -> xr.DataArray:
    observations = xr.DataArray(np.asarray(observations),
                                dims=['step', 'agent', 'window',
                                      'feature'], coords={
            'feature':F_ORDER })
    return observations

class Target:

    def __init__(self, file_path: str, num_samples: int = 16, randomize=True,shuffle_agents=True):
        with lz4.frame.open(file_path, "rb") as f:
            self.ddqn_state, self.history, _ = unpack_history(cloudpickle.load(f))

        self.num_samples = num_samples
        self.randomize = randomize
        self.shuffle_agents = shuffle_agents

        self.model = QNetwork(num_actions=2, num_layers=1, dim=64, num_heads=4)

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
        n_agents = jax.tree.leaves(params)[0].shape[0]
        if self.shuffle_agents:
            perm = jnp.asarray(np.random.permutation(range(n_agents)))
            perm_fun = lambda x: x[perm]  # noqa: E731
            params = jax.tree.map(perm_fun, params)
            state = jax.tree.map(perm_fun, state)

        observations = self.history.observations[
            -1, -1000:, ...]  # last epoch, last 1000 samples, 5 agents,1 window, 5 features


        s, n_agents, w, f = observations.shape
        if self.randomize:

            nel = math.prod(observations.shape[:-1])
            samples = [ randomization[name].rvs(nel).astype(observations.dtype) for name in F_ORDER]
            samples = np.stack(samples, axis=1)
            samples = np.reshape(samples, observations.shape)
            observations = make_xr(samples)
        else:
            observations = make_xr(observations)

        key = jax.random.key(42)
        obs = jnp.transpose(observations.data,(1,0,2,3)) # aswf


        keys = jax.random.split(key, num=n_agents * self.num_samples)
        keys = jnp.reshape(keys, (self.num_samples, n_agents))
        pred = jax.vmap(self.pred)
        pred = jax.vmap(pred, in_axes=(None, None, None, 0), out_axes=-1)


        qvals = jax.jit(pred)(params, state, obs, keys)
        qvals = xr.DataArray(np.asarray(qvals),
                             dims=['agent','step',  'action', 'sample'],
                             coords={'action': [a.name for a in Actions][0:2]})


        return observations, qvals

def dataset2csv(ds: xr.Dataset, out_name: str):
    obs = ds['observations']
    qvals = ds['qvalues']
    actions = qvals.argmax('action')

    # Expand obs to include sample dim and flatten
    X_all = obs.expand_dims(sample=qvals.sample).stack(
        flat=('agent', 'step', 'sample')).transpose('flat', 'window',
                                                    'feature').stack(
        y=('window', 'feature'))

    flat_coords = X_all.flat
    agent_ids = xr.DataArray([c[0] for c in flat_coords.values], dims='flat')
    sample_ids = xr.DataArray([c[2] for c in flat_coords.values], dims='flat')

    df = X_all.to_pandas()
    df.columns = [f'{f}_{w}' for w, f in df.columns]
    df['agent_id'] = agent_ids.values
    df['sample_id'] = sample_ids.values

    # Add Q-value columns for each action
    for action_name in qvals.action.values:
        q_flat = qvals.sel(action=action_name).stack(flat=('agent', 'step', 'sample'))
        df[f'q_{action_name}'] = q_flat.values

    # Add actions column (argmax)
    actions_flat = actions.stack(flat=('agent', 'step', 'sample'))
    df['action'] = actions_flat.values

    df.to_csv(out_name, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Q-network observations and Q-values from history file')
    parser.add_argument('--file', type=str, help='Path to history .pkl.lz4 file')
    parser.add_argument('--n', type=int, default=5, help='Number of agents used for filename matching')
    parser.add_argument('--n_drl', type=int, default=5, help='Number of DRL agents used for filename matching')
    parser.add_argument('--seed', type=int, default=42, help='Seed used for filename matching')

    parser.add_argument('--num-samples', type=int, default=1,
                        help='Number of samples (default: 1)')
    parser.add_argument('--randomize', '-r', action='store_true', default=False,
                        help='Randomize observations (default: False)')
    parser.add_argument('--shuffle-agents', '-s', action='store_true', default=False,
                        help='Shuffle agents (default: False)')
    args = parser.parse_args()

    history_file = resolve_history_file(n=args.n, n_drl=args.n_drl, seed=args.seed, file_path=args.file)
    target = Target(str(history_file))
    observations, qvals = target()
    observations = clean_observations(observations)
    N = args.num_samples

    out_name = f"{history_file}s{target.shuffle_agents}r{target.randomize}N{N}"


    ds = xr.Dataset({'observations': observations, 'qvalues': qvals})
    dataset2csv(ds, f"{out_name}.csv")
    ds.to_netcdf(f"{out_name}.nc", engine="netcdf4")



