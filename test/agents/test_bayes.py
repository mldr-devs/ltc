import unittest
import jax
import jax.numpy as jnp
from ltc.agents.drl import QNetwork, StochasticVariationalNetwork, MultiSVI, Softmaxifier

import flax.linen as nn

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
