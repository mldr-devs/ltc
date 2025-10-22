import unittest

import jax
import jax.numpy as jnp

from ltc.agents.svi import StochasticVariationalNetwork, MultiSVI, Softmaxifier
from ltc.agents import QNetwork
from ltc.sim.constants import Actions


class BayesTestCase(unittest.TestCase):
    def test_svi(self):
        env_state = jnp.zeros((5, 4))

        qn = QNetwork(len(Actions), num_layers=2, dim=32, num_heads=2)
        svi = StochasticVariationalNetwork(model=qn)
        vars = svi.init(jax.random.key(0), env_state)
        y, s = svi.apply(vars, env_state, rngs=jax.random.key(0), mutable=['loss'])

        self.assertTrue('loss' in s)

    def test_multi_svi(self):
        env_state = jnp.zeros((5, 4))

        qn = QNetwork(len(Actions), num_layers=2, dim=32, num_heads=2)
        svi = MultiSVI(model=qn)
        vars = svi.init(jax.random.key(0), env_state)
        y, s = svi.apply(vars, env_state, rngs={'params':jax.random.key(0),'dropout':jax.random.key(0),'rlib':jax.random.key(0)}, mutable=['loss'])

        self.assertFalse(jnp.allclose(y[..., 0], y[..., 1]))
        self.assertTrue('loss' in s)

    def test_bayesian_softmaxifier(self):
        env_state = jnp.zeros((5, 4))

        qn = QNetwork(len(Actions), num_layers=2, dim=32, num_heads=2)
        svi = MultiSVI(model=qn)
        bayes = Softmaxifier(model=svi)
        vars = bayes.init(jax.random.key(0), env_state)
        y, s = bayes.apply(vars, env_state, rngs=jax.random.key(0), mutable=['loss'])

        self.assertTrue('loss' in s)
        self.assertTrue(jnp.allclose(y.sum(axis=-1), jnp.ones(y.shape[:-1])))

    def test_bayesian_aggregation(self):
        env_state = jnp.zeros((5, 4))

        qn = QNetwork(len(Actions), num_layers=2, dim=32, num_heads=2)
        svi = StochasticVariationalNetwork(model=qn)
        multi_svi = MultiSVI(model=qn, apply_mean=True)

        vars = svi.init(jax.random.key(0), env_state)
        y_svi, s = svi.apply(vars, env_state, rngs=jax.random.key(0), mutable=['loss'])

        vars = multi_svi.init(jax.random.key(0), env_state)
        y_multi, s = multi_svi.apply(vars, env_state, rngs=jax.random.key(0), mutable=['loss'])

        self.assertTrue(y_svi.shape == y_multi.shape)


if __name__ == '__main__':
    unittest.main()
