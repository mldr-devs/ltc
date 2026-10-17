import itertools
import unittest

import jax
import jax.numpy as jnp

from ltc.run import update_packet_tracking
from ltc.sim.constants import *
from ltc.sim.process_output import build_observation_entry, normalize_obs, process_output, tx_macro_reward
from ltc.sim.sim import simulate
from ltc.utils.structs import MacroActionState, ObsFeatureConfig, ObsFeatureInputs

EMPTY = EMPTY_PACKET_ID


def _macro_state(n, **kwargs):
    base = dict(
        remaining=jnp.zeros(n, dtype=jnp.int32),
        action_types=jnp.full(n, Actions.TX.value, dtype=jnp.int32),
        durations=jnp.zeros(n, dtype=jnp.int32),
        tx_success_accum=jnp.zeros(n, dtype=jnp.float32),
        tx_planned_accum=jnp.zeros(n, dtype=jnp.float32),
        k_coll_ltc=jnp.zeros(n, dtype=jnp.float32),
        k_coll_coex=jnp.zeros(n, dtype=jnp.float32),
        sub_collision_flag=jnp.zeros(n, dtype=bool),
        sub_collision_other_flag=jnp.zeros(n, dtype=bool),
        header_collision=jnp.zeros(n, dtype=bool),
        header_collision_other=jnp.zeros(n, dtype=bool),
        initial_ret_c=jnp.zeros(n, dtype=jnp.int32),
        tx_start_occupancy=jnp.zeros(n, dtype=jnp.int32),
    )
    base.update(kwargs)
    return MacroActionState(**base)


class MacroTxSettlementTestCase(unittest.TestCase):
    """A macro TX only touches the queue when it ends, with the count the reward is built from."""

    QUEUE = 8
    DURATION = 3

    def _run_macro(self, collide_slots):
        """Drive one station through a DURATION-subframe TX; collide_slots are 0-based mini-slots."""
        n, total = 1, self.DURATION * TX_SLOTS
        macro = _macro_state(n)
        buffers = jnp.array([[10, 11, 12, 13, EMPTY, EMPTY, EMPTY, EMPTY]])
        slot_actions = jnp.array([Actions.TX.value])
        delivered_at_end, drop_at_end, hdr, ack = None, None, None, None

        for i in range(total):
            remaining = total - i
            # a collision shows up as a second transmitter in the slot
            actions = jnp.array([Actions.TX.value, Actions.TX.value]) if i in collide_slots \
                else jnp.array([Actions.TX.value, Actions.IDLE.value])
            pad = lambda x, fill: jnp.concatenate([x, jnp.asarray([fill], dtype=x.dtype)])
            macro2 = jax.tree.map(lambda x: pad(x, 0) if x.dtype != jnp.bool_ else pad(x, False), macro)
            (
                sub_coll, sub_coll_other, tx_packet_mask, tx_success_mask,
                header_collision, _, done_now, ack_collision, _,
                initial_ret_c, tx_start_occupancy,
            ) = update_packet_tracking(
                macro2, jnp.array([True, True]), actions,
                jnp.array([remaining, remaining], dtype=jnp.int32),
                jnp.zeros(2), jnp.array([i == 0, i == 0]), jnp.zeros(2, dtype=jnp.int32),
                jnp.concatenate([buffers, jnp.full((1, self.QUEUE), EMPTY)]), TX_SLOTS,
            )
            take = lambda x: x[:1]
            macro = _macro_state(
                n,
                sub_collision_flag=take(sub_coll), sub_collision_other_flag=take(sub_coll_other),
                header_collision=take(header_collision), initial_ret_c=take(initial_ret_c),
                tx_start_occupancy=take(tx_start_occupancy),
                tx_success_accum=macro.tx_success_accum + take(tx_success_mask).astype(jnp.float32),
                tx_planned_accum=macro.tx_planned_accum + take(tx_packet_mask).astype(jnp.float32),
            )
            hdr, ack = take(header_collision), take(ack_collision)
            if bool(take(done_now)[0]):
                delivered_at_end = jnp.minimum(
                    jnp.where(~(hdr | ack), macro.tx_success_accum, 0.0).astype(jnp.int32),
                    macro.tx_start_occupancy,
                )
                drop_at_end = jnp.zeros(n, dtype=jnp.int32)

        new_buffers, _, _, _, _, successful = simulate(
            buffers, jnp.full((n, self.QUEUE), -1, dtype=jnp.int32), jnp.zeros(n, dtype=jnp.int32),
            slot_actions, jnp.zeros(n, dtype=jnp.int32), 0,
            tx_packet_mask=jnp.zeros(n, dtype=bool), dequeue_count=delivered_at_end,
            drop_count=drop_at_end,
        )
        return int(successful[0]), int(jnp.sum(new_buffers[0] != EMPTY)), bool(hdr[0]), bool(ack[0])

    def test_clean_macro_delivers_every_subframe(self):
        delivered, left, hdr, ack = self._run_macro(collide_slots=set())
        self.assertEqual((delivered, left, hdr, ack), (self.DURATION, 4 - self.DURATION, False, False))

    def test_header_collision_voids_the_whole_macro(self):
        delivered, left, hdr, _ = self._run_macro(collide_slots={0})
        self.assertTrue(hdr)
        self.assertEqual(delivered, 0)
        self.assertEqual(left, 4)

    def test_ack_collision_voids_the_whole_macro(self):
        delivered, left, _, ack = self._run_macro(collide_slots={self.DURATION * TX_SLOTS - 1})
        self.assertTrue(ack)
        self.assertEqual(delivered, 0)
        self.assertEqual(left, 4)

    def test_middle_subframe_collision_loses_only_that_subframe(self):
        delivered, left, hdr, ack = self._run_macro(collide_slots={TX_SLOTS})
        self.assertFalse(hdr or ack)
        self.assertEqual(delivered, self.DURATION - 1)
        self.assertEqual(left, 4 - (self.DURATION - 1))


class RetryExhaustionTestCase(unittest.TestCase):
    def test_exhausted_macro_drops_the_head_packet(self):
        buffers = jnp.array([[10, 11, EMPTY, EMPTY]])
        new_buffers, _, _, _, _, successful = simulate(
            buffers, jnp.full((1, 4), -1, dtype=jnp.int32), jnp.zeros(1, dtype=jnp.int32),
            jnp.array([Actions.TX.value]), jnp.zeros(1, dtype=jnp.int32), 0,
            tx_packet_mask=jnp.zeros(1, dtype=bool),
            dequeue_count=jnp.zeros(1, dtype=jnp.int32), drop_count=jnp.ones(1, dtype=jnp.int32),
        )
        self.assertEqual(int(successful[0]), 0)
        self.assertEqual(int(new_buffers[0][0]), 11)
        self.assertEqual(int(jnp.sum(new_buffers[0] != EMPTY)), 1)

    def test_exhausted_macro_reward_is_the_max_retry_penalty_alone(self):
        r = tx_macro_reward(
            k_tx=jnp.array(0.0), k_tx_planned=jnp.array(1.0),
            k_coll_ltc=jnp.array(1.0), k_coll_coex=jnp.array(0.0),
            tx_started_empty=jnp.array(False), retry_exhausted=jnp.array(True),
            header_collision=jnp.array(False), ack_collision=jnp.array(False),
            header_collision_other=jnp.array(False), ack_collision_other=jnp.array(False), tx_slots=TX_SLOTS,
        )
        self.assertAlmostEqual(float(r), TX_MAX_RETRANSMISSION_PENALTY, places=6)


class ObservationWindowTestCase(unittest.TestCase):
    """The window must retain each macro's settled end state, one slot per macro."""

    WINDOW = 4
    MACROS = [3, 2, 2]          # mini-slots per macro; their last ticks are 3, 5, 7

    def _window_tags(self):
        """Tag every mini-slot with its own tick and return the window after the last one."""
        obs = jnp.zeros((1, self.WINDOW, OBS_SIZE), dtype=jnp.float32)
        tick = 0
        for dur in self.MACROS:
            for i in range(dur):
                tick += 1
                obs, _, _ = process_output(
                    buffer_states=jnp.array([[EMPTY, EMPTY]]),
                    new_buffer_states=jnp.array([[EMPTY, EMPTY]]),
                    new_buffer_birth_states=jnp.array([[-1, -1]]),
                    power_states=jnp.array([100]), channel_state=0, obs=obs,
                    actions=jnp.array([Actions.CS.value]), terminals=jnp.array([False]),
                    key=jax.random.PRNGKey(0), current_step=tick,
                    obs_features=ObsFeatureInputs(*[jnp.array([0.0])] * 6),
                    obs_config=ObsFeatureConfig(enable_cs_tx_same_type=False,
                                                enable_tx_collision_other=False),
                    action_durations=jnp.array([tick], dtype=jnp.int32),   # stands in for the tick
                    roll_obs=jnp.array([i == 0]),                          # roll at macro start
                    done_now=jnp.array([i == dur - 1]),
                )
        return [int(v) for v in obs[0, :, ObsIdx.ACTION_DURATION]]

    def test_window_holds_macro_end_states_one_per_macro(self):
        ends = list(itertools.accumulate(self.MACROS))
        expected = ([0] * (self.WINDOW - len(ends)) + ends)[-self.WINDOW:]
        self.assertEqual(self._window_tags(), expected)


class EmptyBufferTxTestCase(unittest.TestCase):
    def test_five_subframe_tx_on_an_empty_buffer_costs_the_empty_penalty_alone(self):
        r = tx_macro_reward(
            k_tx=jnp.array(0.0), k_tx_planned=jnp.array(5.0),
            k_coll_ltc=jnp.array(0.0), k_coll_coex=jnp.array(0.0),
            tx_started_empty=jnp.array(True), retry_exhausted=jnp.array(False),
            header_collision=jnp.array(False), ack_collision=jnp.array(False),
            header_collision_other=jnp.array(False), ack_collision_other=jnp.array(False), tx_slots=TX_SLOTS,
        )
        self.assertAlmostEqual(float(r), TX_EMPTY_BUFFER_PENALTY * 5.0, places=6)  # -2.5, no size penalty

    def test_size_penalty_still_applies_with_a_non_empty_buffer(self):
        r = tx_macro_reward(
            k_tx=jnp.array(0.0), k_tx_planned=jnp.array(5.0),
            k_coll_ltc=jnp.array(0.0), k_coll_coex=jnp.array(0.0),
            tx_started_empty=jnp.array(False), retry_exhausted=jnp.array(False),
            header_collision=jnp.array(False), ack_collision=jnp.array(False),
            header_collision_other=jnp.array(False), ack_collision_other=jnp.array(False), tx_slots=TX_SLOTS,
        )
        self.assertAlmostEqual(float(r), TX_SIZE_PENALTY, places=6)


class CoexCollisionTestCase(unittest.TestCase):
    def test_coex_collision_is_penalised_regardless_of_the_observation_flag(self):
        """The reward reads the true collision type; obs_config only shapes what the agent sees."""
        rewards = []
        for flag in (False, True):
            _, r, _ = process_output(
                buffer_states=jnp.array([[7, EMPTY]]), new_buffer_states=jnp.array([[7, EMPTY]]),
                new_buffer_birth_states=jnp.array([[0, -1]]), power_states=jnp.array([100]),
                channel_state=-1, obs=jnp.zeros((1, 1, OBS_SIZE), dtype=jnp.float32),
                actions=jnp.array([Actions.TX.value]), terminals=jnp.array([False]),
                key=jax.random.PRNGKey(0), current_step=0,
                obs_features=ObsFeatureInputs(*[jnp.array([0.0])] * 6),
                obs_config=ObsFeatureConfig(enable_cs_tx_same_type=False, enable_tx_collision_other=flag),
                done_now=jnp.array([True]), k_tx_planned=jnp.array([1.0]),
                k_coll_coex=jnp.array([1.0]),
            )
            rewards.append(float(r[0]))
        self.assertEqual(rewards[0], rewards[1])
        self.assertAlmostEqual(rewards[0], TX_COEX_COLLISION_PENALTY, places=6)


class TxCollisionOtherFeatureTestCase(unittest.TestCase):
    """The feature must separate 'no collision' from 'collided with an LTC node'."""

    def _feature(self, channel_state, collision_with_other):
        features = ObsFeatureInputs(
            traffic_mean_arrival_rate=jnp.array(0.0),
            sensed_busy_pct_window=jnp.array(0.0),
            back_pct=jnp.array(0.0),
            unique_ltc_tx_window=jnp.array(0.0),
            cs_tx_same_type_now=jnp.array(0.0),
            tx_collision_other_now=jnp.array(collision_with_other),
        )
        entry = build_observation_entry(
            jnp.array([EMPTY, EMPTY]), jnp.array([EMPTY, EMPTY]), jnp.array([-1, -1]),
            jnp.array(Actions.TX.value), jnp.array(2, dtype=jnp.int32), jnp.array(channel_state),
            0, features, ObsFeatureConfig(enable_cs_tx_same_type=False, enable_tx_collision_other=True),
            jnp.array(0, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32),
        )
        return float(entry[ObsIdx.CHANNEL_LAST_TX_COLLISION_OTHER])

    def test_no_collision(self):
        self.assertEqual(self._feature(channel_state=1, collision_with_other=0.0), -1.0)

    def test_collision_with_ltc(self):
        self.assertEqual(self._feature(channel_state=-1, collision_with_other=0.0), 0.0)

    def test_collision_with_coex(self):
        self.assertEqual(self._feature(channel_state=-1, collision_with_other=1.0), 1.0)


class DurationObservationTestCase(unittest.TestCase):
    def test_chosen_duration_is_observed_and_normalized(self):
        features = ObsFeatureInputs(*[jnp.array(0.0)] * 6)
        entry = build_observation_entry(
            jnp.array([EMPTY, EMPTY]), jnp.array([EMPTY, EMPTY]), jnp.array([-1, -1]),
            jnp.array(Actions.TX.value), jnp.array(3, dtype=jnp.int32), jnp.array(1),
            0, features, ObsFeatureConfig(enable_cs_tx_same_type=False, enable_tx_collision_other=False),
            jnp.array(0, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32),
        )
        self.assertEqual(float(entry[ObsIdx.ACTION_DURATION]), 3.0)
        normalized = normalize_obs(entry[None, None, :], queue_size=2, tx_slots=TX_SLOTS)
        self.assertAlmostEqual(
            float(normalized[0, 0, ObsIdx.ACTION_DURATION]), 3.0 / OBS_DURATION_NORM, places=6
        )


if __name__ == '__main__':
    unittest.main()
