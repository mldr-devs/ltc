import unittest
import jax
import jax.numpy as jnp
from ltc.agents.drl import QNetwork, StochasticVariationalNetwork


class BayesTestCase(unittest.TestCase):
    def test_svi(self):
        env_state = jnp.zeros((5, 4))

        qn = QNetwork()
        svi = StochasticVariationalNetwork(model=qn)
        vars = svi.init(jax.random.key(0), env_state)
        y, s = svi.apply(vars, env_state, rngs=jax.random.key(0), mutable=['loss'])

        self.assertTrue('loss' in s)


if __name__ == '__main__':
    unittest.main()
