import unittest
import scipy.signal as signal
import numpy as np

import jax

import jax.numpy as jnp

from ltc.sim.constants import TAU
from ltc.sim.generate_frames import _ss_step, cox_traffic


class TrafficTestCase(unittest.TestCase):
    def test_ss(self):
        dt = 1e-3
        fs = 1/dt  # Sampling frequency in Hz
        cutoff = 1/(10*dt)  # Desired 3dB cutoff frequency in Hz
        order = 2  # Filter order

        nyquist = fs / 2
        normalized_cutoff = cutoff / nyquist

        b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)

        A, B, C, D = signal.tf2ss(b, a)
        A, B, C, D = jax.tree.map(jnp.squeeze, (A, B, C, D))

        u = np.random.randn(6)
        y,_ = signal.lfilter(b, a, u, zi=np.zeros((order,)))

        x = np.zeros((A.shape[0],))

        def step(x,u):
            y, x = _ss_step(x, u, A, B, C, D)

            return x,y.squeeze()

        _,yjax = jax.lax.scan(step,x, u)

        self.assertTrue(np.allclose(y, yjax))

    def test_cycle(self):
        model = cox_traffic(f3dB=1/(10*TAU))

        state = model.init(jax.random.key(4))

        state, frames = model.sample(state, key=jax.random.key(44))


if __name__ == "__main__":
    unittest.main()
