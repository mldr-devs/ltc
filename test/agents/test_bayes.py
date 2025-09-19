import unittest
import jax
import jax.numpy as jnp
from ltc.agents.drl import QNetwork, StochasticVariationalNetwork

import flax.linen as nn

class Softmaxifier(nn.Module):
    """Convert a set of logits into a probability distribution. Optionally can perform Bayesian model aggregation by averaging posterior samples"""
    model: nn.Module
    softmax_axis: int = -2
    mean_axis: int = -1
    @nn.compact
    def __call__(self, x):
        logits = self.model(x)
        x = nn.softmax(logits, axis=self.softmax_axis)
        if self.mean_axis is not None:
            x = x.mean(axis=self.mean_axis)
        return x

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
    def test_bayesian_aggregation(self):
        env_state = jnp.zeros((5, 4))

        qn = QNetwork()
        svi = MultiSVI(model=qn)
        bayes = Softmaxifier(model=svi)
        vars = bayes.init(jax.random.key(0), env_state)
        y, s = bayes.apply(vars, env_state, rngs=jax.random.key(0), mutable=['loss'])

        self.assertTrue('loss' in s)
        self.assertTrue(jnp.allclose(y.sum(axis=-1), jnp.ones(y.shape[:-1])))
if __name__ == '__main__':
    unittest.main()
