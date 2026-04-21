import unittest

import jax
import jax.numpy as jnp

from ltc.run import decode_drl_actions, decode_legacy_actions, upgraded_tx_slot_reward
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

    def test_upgraded_tx_slot_reward_branches(self):
        slot_actions = jnp.array([Actions.TX.value, Actions.TX.value, Actions.TX.value, Actions.TX.value], dtype=jnp.int32)
        channel_state = jnp.array([1, -1, -1, 1], dtype=jnp.int32)
        successful_packets = jnp.array([1, 0, 0, 0], dtype=jnp.int32)
        prev_ret_c = jnp.array([0, 0, 8, 0], dtype=jnp.int32)
        tx_collision_other_now = jnp.array([0.0, 1.0, 0.0, 0.0], dtype=jnp.float32)
        rewards, k_tx, k_coll, tx_exec = upgraded_tx_slot_reward(
            slot_actions, channel_state, successful_packets, prev_ret_c, tx_collision_other_now
        )
        self.assertTrue(jnp.array_equal(tx_exec, jnp.ones((4,), dtype=bool)))
        self.assertTrue(jnp.array_equal(k_tx, jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)))
        self.assertTrue(jnp.array_equal(k_coll, jnp.array([0.0, 1.0, 1.0, 0.0], dtype=jnp.float32)))
        # success, coex collision, max retransmission, empty-buffer TX
        self.assertAlmostEqual(float(rewards[0]), 0.6, places=6)
        self.assertAlmostEqual(float(rewards[1]), -5.0, places=6)
        self.assertAlmostEqual(float(rewards[2]), -1.0, places=6)
        self.assertAlmostEqual(float(rewards[3]), -0.5, places=6)


if __name__ == "__main__":
    unittest.main()
