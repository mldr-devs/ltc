import unittest

import jax.numpy as jnp
import numpy as np

from ltc.symbolic.util import history_reshape


class TestHistoryReshape(unittest.TestCase):
    N_STEPS = 7
    N_AGENTS = 3
    WINDOW = 4
    N_FEATURES = 6

    def setUp(self):
        shape = (self.N_STEPS, self.N_AGENTS, self.WINDOW, self.N_FEATURES)
        # distinct values so any mis-ordering is detectable
        self.obs = jnp.arange(np.prod(shape), dtype=jnp.int32).reshape(shape)

    def test_shape(self):
        XX = history_reshape(self.obs)
        self.assertEqual(
            XX.shape,
            (self.N_AGENTS * self.N_STEPS, self.WINDOW * self.N_FEATURES),
        )

    def test_rows_are_agent_major(self):
        """Row a*n_steps + t is agent a at step t, flattened."""
        XX = np.asarray(history_reshape(self.obs))
        obs = np.asarray(self.obs)
        for a in range(self.N_AGENTS):
            for t in range(self.N_STEPS):
                np.testing.assert_array_equal(
                    XX[a * self.N_STEPS + t], obs[t, a].reshape(-1)
                )

    def test_matches_per_agent_concatenation(self):
        obs = np.asarray(self.obs)
        expected = np.concatenate(
            [obs[:, a].reshape(self.N_STEPS, -1) for a in range(self.N_AGENTS)], axis=0
        )
        np.testing.assert_array_equal(np.asarray(history_reshape(self.obs)), expected)

    def test_columns_are_window_step_major(self):
        """Within a row, features are grouped by window step (matches build_column_names)."""
        XX = np.asarray(history_reshape(self.obs))
        obs = np.asarray(self.obs)
        row = XX[0]  # agent 0, step 0
        for w in range(self.WINDOW):
            lo, hi = w * self.N_FEATURES, (w + 1) * self.N_FEATURES
            np.testing.assert_array_equal(row[lo:hi], obs[0, 0, w])

    def test_single_observation_via_newaxis(self):
        """The SRJaxAgent path: one [w, f] obs promoted to a [1, 1, w, f] history."""
        single = self.obs[0, 0]
        x = history_reshape(single[jnp.newaxis, jnp.newaxis])
        self.assertEqual(x.shape, (1, self.WINDOW * self.N_FEATURES))
        np.testing.assert_array_equal(np.asarray(x)[0], np.asarray(single).reshape(-1))

    def test_rejects_non_history_rank(self):
        with self.assertRaises(Exception):
            history_reshape(jnp.zeros((self.WINDOW, self.N_FEATURES)))

    def test_preserves_dtype(self):
        self.assertEqual(history_reshape(self.obs).dtype, self.obs.dtype)
        f = self.obs.astype(jnp.float32)
        self.assertEqual(history_reshape(f).dtype, jnp.float32)


if __name__ == "__main__":
    unittest.main()
