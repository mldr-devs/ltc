import unittest

import jax.numpy as jnp

from ltc.run import decode_drl_actions, decode_legacy_actions
from ltc.sim.constants import Actions, TX_REWARD, TX_SLOTS, TX_LTC_COLLISION_PENALTY, TX_COEX_COLLISION_PENALTY, TX_MAX_RETRANSMISSION_PENALTY, TX_EMPTY_BUFFER_PENALTY, MAX_RETRANSMISSION, TX_SIZE_THRESHOLD, TX_SIZE_PENALTY, TX_SIZE_PENALTY_WINDOW
from ltc.sim.process_output import tx_macro_reward


class ActionSpaceDecodeTestCase(unittest.TestCase):
    def test_duration_set_decode(self):
        raw = jnp.array([0, 4, 5, 9], dtype=jnp.int32)
        action_type, duration, staged = decode_drl_actions(
            raw, jnp.zeros((4,), dtype=jnp.int32), "txcs_duration_set", 5
        )
        self.assertTrue(jnp.array_equal(action_type, jnp.array([Actions.TX.value, Actions.TX.value, Actions.CS.value, Actions.CS.value])))
        self.assertTrue(jnp.array_equal(duration, jnp.array([1, 5, 1, 5])))
        self.assertTrue(jnp.array_equal(staged, jnp.zeros((4,), dtype=jnp.int32)))

    def test_stage_commit_decode(self):
        raw = jnp.array([1, 2, 2], dtype=jnp.int32)  # stage, commit, commit
        staged = jnp.array([0, 2, 1], dtype=jnp.int32)
        action_type, duration, staged_next = decode_drl_actions(
            raw, staged, "tx_stage_commit", 5
        )
        self.assertTrue(jnp.array_equal(action_type, jnp.array([Actions.CS.value, Actions.TX.value, Actions.TX.value])))
        self.assertTrue(jnp.array_equal(duration, jnp.array([1, 2, 1])))
        self.assertTrue(jnp.array_equal(staged_next, jnp.array([1, 0, 0])))

    def test_legacy_decode(self):
        raw = jnp.array([Actions.TX.value, Actions.CS.value, Actions.IDLE.value], dtype=jnp.int32)
        action_type, duration = decode_legacy_actions(raw, legacy_tx_duration=5)
        self.assertTrue(jnp.array_equal(action_type, raw))
        self.assertTrue(jnp.array_equal(duration, jnp.array([5, 1, 1], dtype=jnp.int32)))

    def _macro_reward(self, **kwargs):
        defaults = dict(
            k_tx=jnp.array(0.0), k_tx_planned=jnp.array(0.0),
            k_coll_ltc=jnp.array(0.0), k_coll_coex=jnp.array(0.0),
            tx_started_empty=jnp.array(False), initial_ret_c=jnp.array(0),
            header_collision=jnp.array(False), ack_collision=jnp.array(False),
            header_collision_other=jnp.array(False), ack_collision_other=jnp.array(False),
        )
        defaults.update(kwargs)
        return tx_macro_reward(**defaults)

    def test_tx_macro_reward_branches(self):
        # success: 1 sub-window, no collision
        r = self._macro_reward(k_tx=jnp.array(1.0), k_tx_planned=jnp.array(1.0))
        expected = TX_REWARD * (1.0 * TX_SLOTS - 2.0) / TX_SLOTS
        self.assertAlmostEqual(float(r), expected, places=6)

        # coex collision, ret_c < MAX: 1 collided sub-window → full coex penalty
        r = self._macro_reward(k_coll_coex=jnp.array(1.0), k_tx_planned=jnp.array(1.0))
        self.assertAlmostEqual(float(r), TX_COEX_COLLISION_PENALTY * 1.0, places=6)

        # LTC collision + max retransmission: collision penalty + max-retry penalty (additive)
        r = self._macro_reward(
            k_coll_ltc=jnp.array(1.0), k_tx_planned=jnp.array(1.0),
            initial_ret_c=jnp.array(MAX_RETRANSMISSION),
        )
        expected = TX_LTC_COLLISION_PENALTY * 1.0 + TX_MAX_RETRANSMISSION_PENALTY
        self.assertAlmostEqual(float(r), expected, places=6)

        # empty buffer, no collision: penalty scales with planned duration
        r = self._macro_reward(tx_started_empty=jnp.array(True), k_tx_planned=jnp.array(2.0))
        self.assertAlmostEqual(float(r), TX_EMPTY_BUFFER_PENALTY * 2.0, places=6)

        # header collision (LTC): all 3 planned sub-windows → 3 LTC collisions
        # size penalty applies: k_tx_planned=3, ramp=(3-2)/2=0.5
        r = self._macro_reward(
            k_tx=jnp.array(3.0), k_tx_planned=jnp.array(3.0),
            header_collision=jnp.array(True), header_collision_other=jnp.array(False),
        )
        expected = TX_LTC_COLLISION_PENALTY * 3.0 + TX_SIZE_PENALTY * 0.5
        self.assertAlmostEqual(float(r), expected, places=6)

        # header collision (coex): all 2 planned sub-windows → 2 coex collisions
        r = self._macro_reward(
            k_tx=jnp.array(2.0), k_tx_planned=jnp.array(2.0),
            header_collision=jnp.array(True), header_collision_other=jnp.array(True),
        )
        self.assertAlmostEqual(float(r), TX_COEX_COLLISION_PENALTY * 2.0, places=6)

        # ACK collision (LTC), 2 planned sub-windows: all flagged → 2 LTC collisions
        r = self._macro_reward(
            k_tx=jnp.array(1.0), k_coll_ltc=jnp.array(1.0), k_tx_planned=jnp.array(2.0),
            ack_collision=jnp.array(True), ack_collision_other=jnp.array(False),
        )
        self.assertAlmostEqual(float(r), TX_LTC_COLLISION_PENALTY * 2.0, places=6)


if __name__ == "__main__":
    unittest.main()
