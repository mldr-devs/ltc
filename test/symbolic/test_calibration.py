import unittest

import jax.numpy as jnp
import numpy as np

from ltc.symbolic.calibration import ScaleCalibrator, decode_probs, simplex_code_np
from ltc.symbolic.util import SimplexCode, simplex_code


class TestDecodeProbs(unittest.TestCase):

    def test_matches_jax_codes(self):
        for T in (2, 3, 5):
            np.testing.assert_allclose(simplex_code_np(T), np.asarray(simplex_code(T)), atol=1e-6)

    def test_round_trip(self):
        """rho recovered from f = sum_y rho_y c_y is rho itself (spec, test 2)."""
        rng = np.random.default_rng(0)
        for T in (2, 3, 4):
            C = simplex_code_np(T)
            p = rng.dirichlet(np.ones(T), size=64)
            np.testing.assert_allclose(decode_probs(p @ C.T, C), p, atol=1e-9)

    def test_zero_is_uniform(self):
        for T in (2, 3, 6):
            p = decode_probs(np.zeros((3, T - 1)), simplex_code_np(T))
            np.testing.assert_allclose(p, np.full((3, T), 1 / T), atol=1e-12)

    def test_sums_to_one(self):
        rng = np.random.default_rng(1)
        C = simplex_code_np(4)
        p = decode_probs(rng.normal(scale=3.0, size=(128, 3)), C)  # well outside conv(C)
        np.testing.assert_allclose(p.sum(axis=1), np.ones(128), atol=1e-12)


class TestScaleCalibrator(unittest.TestCase):

    def _flat_data(self, T=3, n=2000, shrink=0.2, seed=0):
        """Codewords shrunk towards 0: exactly the under-fitted SR case."""
        rng = np.random.default_rng(seed)
        C = simplex_code_np(T)
        y = rng.integers(0, T, size=n)
        f = shrink * C[:, y].T
        return f, y, C

    def test_scale_sharpens_a_shrunk_fit(self):
        f, y, C = self._flat_data()
        cal = ScaleCalibrator(T=3).fit(f, y)
        self.assertGreater(cal.scale, 1.0)
        before = decode_probs(f, C)
        after = cal.predict_proba(f)
        self.assertGreater(after.max(axis=1).mean(), before.max(axis=1).mean())

    def test_scale_improves_nll(self):
        f, y, C = self._flat_data()
        rows = np.arange(len(y))
        nll = lambda p: -np.log(np.clip(p[rows, y], 1e-12, None)).mean()
        cal = ScaleCalibrator(T=3).fit(f, y)
        self.assertLess(nll(cal.predict_proba(f)), nll(decode_probs(f, C)))

    def test_calibration_does_not_change_argmax(self):
        """Spec test 7: a > 0 rescales every score alike, so the policy's argmax is fixed."""
        rng = np.random.default_rng(2)
        for T in (2, 3, 5):
            C = simplex_code_np(T)
            f = rng.normal(size=(256, T - 1))
            base = decode_probs(f, C).argmax(axis=1)
            for a in (0.01, 0.5, 1.0, 4.0, 100.0):
                np.testing.assert_array_equal(decode_probs(f, C, a).argmax(axis=1), base)

    def test_clipped_fraction_reported(self):
        f, y, _ = self._flat_data()
        cal = ScaleCalibrator(T=3).fit(f, y)
        self.assertGreaterEqual(cal.clipped_fraction_, 0.0)
        self.assertLessEqual(cal.clipped_fraction_, 1.0)


class TestSimplexCodeScale(unittest.TestCase):
    """The agent-side decoder must agree with the numpy one it is calibrated with."""

    def test_probs_matches_numpy(self):
        rng = np.random.default_rng(3)
        for T in (2, 3, 4):
            f = rng.normal(scale=0.4, size=(32, T - 1))
            for a in (1.0, 2.5):
                got = np.asarray(SimplexCode(T=T, scale=a).probs(jnp.asarray(f)))
                np.testing.assert_allclose(got, decode_probs(f, simplex_code_np(T), a), atol=1e-5)

    def test_scale_leaves_decode_alone(self):
        rng = np.random.default_rng(4)
        f = jnp.asarray(rng.normal(size=(64, 2)))
        np.testing.assert_array_equal(
            np.asarray(SimplexCode(T=3, scale=7.0).probs(f).argmax(axis=1)),
            np.asarray(SimplexCode(T=3).decode(f)),
        )


if __name__ == "__main__":
    unittest.main()
