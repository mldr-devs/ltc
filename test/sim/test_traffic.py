import unittest

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

from ltc.sim.constants import TAU
from ltc.sim.traffic import _ss_step, cox_traffic


class TrafficTestCase(unittest.TestCase):
    def test_ss(self):
        dt = 1e-3
        fs = 1 / dt  # Sampling frequency in Hz
        cutoff = 1 / (10 * dt)  # Desired 3dB cutoff frequency in Hz
        order = 2  # Filter order

        nyquist = fs / 2
        normalized_cutoff = cutoff / nyquist

        b, a = signal.butter(order, normalized_cutoff, btype="low", analog=False)

        A, B, C, D = signal.tf2ss(b, a)
        A, B, C, D = jax.tree.map(jnp.squeeze, (A, B, C, D))

        u = np.random.randn(6)
        y, _ = signal.lfilter(b, a, u, zi=np.zeros((order,)))

        x = np.zeros((A.shape[0],))

        def step(x, u):
            y, x = _ss_step(x, u, A, B, C, D)
            return x, y.squeeze()

        _, yjax = jax.lax.scan(step, x, u)

        self.assertTrue(np.allclose(y, yjax))

    def test_cycle(self):
        model = cox_traffic(f3dB=1 / (10 * TAU))
        state = model.init(jax.random.key(4))
        state, frames = model.sample(state, key=jax.random.key(44))

    def test_stats(self):
        model = cox_traffic(f3dB=1 / (10 * TAU), loc=0.1)
        state = model.init(jax.random.key(4))
        stats = model.stats()


        keys = jax.random.split(jax.random.PRNGKey(44), 100000)
        _, yjax = jax.lax.scan(model.sample, state, keys)

        # import matplotlib.pyplot as plt
        #
        # l,c,_,_ = plt.acorr(yjax-yjax.mean(),maxlags=4)
        # plt.show()
        # estimated_acf = c[5]
        #
        estimated_acf=np.corrcoef(yjax[:-1]-yjax.mean(), yjax[1:]-yjax.mean())[1,0]


        self.assertAlmostEqual(yjax.mean(), stats.mean, places=2)
        self.assertAlmostEqual(yjax.var(), stats.variance, places=1)
        self.assertAlmostEqual(estimated_acf, stats.acf_lag1, places=2)
        ...




if __name__ == "__main__":
    unittest.main()
