import unittest

import jax
import jax.numpy as jnp

from ltc.run import decode_drl_actions, decode_legacy_actions
from ltc.sim.constants import Actions


class ActionSpaceDecodeTestCase(unittest.TestCase):
    def test_duration_set_decode(self):
        raw = jnp.array([0, 4, 5, 9], dtype=jnp.int32)
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        action_type, duration, staged = decode_drl_actions(
            raw, jnp.zeros((4,), dtype=jnp.int32), keys, "txcs_duration_set", 5
        )
        self.assertTrue(jnp.array_equal(action_type, jnp.array([Actions.TX.value, Actions.TX.value, Actions.CS.value, Actions.CS.value])))
        self.assertTrue(jnp.array_equal(duration, jnp.array([1, 5, 1, 5])))
        self.assertTrue(jnp.array_equal(staged, jnp.zeros((4,), dtype=jnp.int32)))

    def test_stage_commit_decode(self):
        raw = jnp.array([1, 2, 2], dtype=jnp.int32)  # stage, commit, commit
        staged = jnp.array([0, 2, 1], dtype=jnp.int32)
        keys = jax.random.split(jax.random.PRNGKey(1), 3)
        action_type, duration, staged_next = decode_drl_actions(
            raw, staged, keys, "tx_stage_commit", 5
        )
        self.assertTrue(jnp.array_equal(action_type, jnp.array([Actions.CS.value, Actions.TX.value, Actions.TX.value])))
        self.assertTrue(jnp.array_equal(duration, jnp.array([1, 2, 1])))
        self.assertTrue(jnp.array_equal(staged_next, jnp.array([1, 0, 0])))

    def test_geometric_decode_range(self):
        raw = jnp.array([0, 9], dtype=jnp.int32)
        keys = jax.random.split(jax.random.PRNGKey(2), 2)
        action_type, duration, _ = decode_drl_actions(
            raw, jnp.zeros((2,), dtype=jnp.int32), keys, "txcs_to_geometric_duration", 5
        )
        self.assertTrue(jnp.array_equal(action_type, jnp.array([Actions.TX.value, Actions.CS.value])))
        self.assertTrue(jnp.all(duration >= 1))
        self.assertTrue(jnp.all(duration <= 5))

    def test_legacy_decode(self):
        raw = jnp.array([Actions.TX.value, Actions.CS.value, Actions.IDLE.value], dtype=jnp.int32)
        action_type, duration = decode_legacy_actions(raw, legacy_tx_duration=5)
        self.assertTrue(jnp.array_equal(action_type, raw))
        self.assertTrue(jnp.array_equal(duration, jnp.array([5, 1, 1], dtype=jnp.int32)))


if __name__ == "__main__":
    unittest.main()
