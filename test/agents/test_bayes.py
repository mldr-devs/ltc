import unittest
import jax
import jax.numpy as jnp
from ltc.agents.drl import QNetwork, StochasticVariationalNetwork

import flax.linen as nn

class MultiSVI(nn.Module):
    model: nn.Module
    num_ensembles: int = 16

    @nn.compact
    def __call__(self, x):
        Batched = nn.vmap(
            StochasticVariationalNetwork,
            in_axes=(None,), out_axes=-1,
            variable_axes={'params': None, 'loss': 0},
            split_rngs={'params': False, 'dropout': True, 'posterior': True},
            axis_size=self.num_ensembles
        )
        ensemble = Batched(model=self.model, name='ensemble')

        return ensemble(x)

class BayesTestCase(unittest.TestCase):
    def test_svi(self):
        env_state = jnp.zeros((5, 4))

        qn = QNetwork()
        svi = StochasticVariationalNetwork(model=qn)
        vars = svi.init(jax.random.key(0), env_state)
        y, s = svi.apply(vars, env_state, rngs=jax.random.key(0), mutable=['loss'])

        self.assertTrue('loss' in s)

    def test_multi_svi(self):
        env_state = jnp.zeros((5, 4))

        qn = QNetwork()
        svi = MultiSVI(model=qn)
        vars = svi.init(jax.random.key(0), env_state)
        y, s = svi.apply(vars, env_state, rngs=jax.random.key(0), mutable=['loss'])

        self.assertTrue('loss' in s)
if __name__ == '__main__':
    unittest.main()
