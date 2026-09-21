"""Logit coding: dummy-coded labels, T-1 logits, a reference class pinned to 0."""

import unittest

import jax.numpy as jnp
import numpy as np

from ltc.agents.sr_jax import template_callable
from ltc.symbolic.util import LogitCode


def softmax_reference(logits):
    """softmax over [logits..., 0], the reference class's hard 1 written out."""
    full = np.concatenate([logits, np.zeros((len(logits), 1))], axis=1)
    e = np.exp(full - full.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class TestLogitCode(unittest.TestCase):
    def test_encode_is_dummy_coding(self):
        for T in (2, 3, 5):
            labels = np.arange(T)
            codes = np.asarray(LogitCode(T=T).encode(labels))
            self.assertEqual(codes.shape, (T, T - 1))
            # one 1 per row except the reference class, which is all zeros
            np.testing.assert_array_equal(codes.sum(axis=1), [1.0] * (T - 1) + [0.0])
            np.testing.assert_array_equal(np.asarray(codes[:-1]), np.eye(T - 1))

    def test_probs_match_softmax_with_zero_reference(self):
        rng = np.random.default_rng(0)
        for T in (2, 3, 5):
            logits = rng.normal(size=(7, T - 1)) * 3
            got = np.asarray(LogitCode(T=T).probs(jnp.asarray(logits)))
            self.assertEqual(got.shape, (7, T))
            np.testing.assert_allclose(got.sum(axis=1), 1.0, rtol=1e-6)
            np.testing.assert_allclose(got, softmax_reference(logits), rtol=1e-5)

    def test_probs_survive_large_logits(self):
        # A saturating expression must not decode to NaN.
        got = np.asarray(LogitCode(T=3).probs(jnp.array([[300.0, -300.0]])))
        self.assertTrue(np.isfinite(got).all())
        np.testing.assert_allclose(got.sum(), 1.0, rtol=1e-6)

    def test_decode_is_argmax_over_the_completed_logits(self):
        codec = LogitCode(T=3)
        # last column is the reference 0: it wins when both logits are negative
        got = np.asarray(codec.decode(jnp.array([[1.0, 0.5], [0.5, 1.0], [-1.0, -2.0]])))
        np.testing.assert_array_equal(got, [0, 1, 2])


class TestTemplateCallable(unittest.TestCase):
    """The bridge standing in for PySR's .jax(), which refuses templates."""

    class FakeModel:
        def __init__(self, equation):
            self.equation = equation

        def get_best(self, index=None):
            return {'equation': self.equation}

    def test_parses_subexpressions_and_stacks_logits(self):
        model = self.FakeModel('f1 = #1 * 2.0; f2 = #2 - 1.0')
        fn, params = template_callable(model, None, n_features=2)
        x = jnp.array([[1.0, 2.0], [0.5, -1.0]])
        got = np.asarray(fn(x, params))
        self.assertEqual(got.shape, (2, 2))
        np.testing.assert_allclose(got, [[2.0, 1.0], [1.0, -2.0]], rtol=1e-6)

    def test_constant_subexpression_broadcasts(self):
        # sympy2jax returns a scalar for a constant; stacking it with a per-row
        # sub-expression must still give one row per input.
        fn, params = template_callable(self.FakeModel('f1 = #1; f2 = -0.5'), None, n_features=2)
        got = np.asarray(fn(jnp.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]), params))
        self.assertEqual(got.shape, (3, 2))
        np.testing.assert_allclose(got, [[1.0, -0.5], [2.0, -0.5], [3.0, -0.5]], rtol=1e-6)

    def test_placeholders_are_one_based(self):
        # #1 is the first argument of the sub-expression, i.e. column 0.
        fn, params = template_callable(self.FakeModel('f1 = #2'), None, n_features=2)
        np.testing.assert_allclose(np.asarray(fn(jnp.array([[7.0, 9.0]]), params)), [[9.0]])


if __name__ == '__main__':
    unittest.main()
