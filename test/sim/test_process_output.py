import unittest

import jax
import jax.numpy as jnp

from ltc.sim.constants import *
from ltc.sim.process_output import *
from ltc.utils.structs import ObsFeatureInputs, ObsFeatureConfig

KEY = jax.random.PRNGKey(0)
EMPTY = EMPTY_PACKET_ID


class ProcessOutputTestCase(unittest.TestCase):
    def test_no_transmission(self):
        args = (Actions.IDLE.value, jnp.array([1, EMPTY, EMPTY]), 5, 0, 1, KEY)
        reward, r, no_tx = no_transmission(args)
        self.assertEqual(reward, 0.0)
        self.assertEqual(r, 5)

    def _make_tx_process_output(self, ret_c_init, channel_state, tx_packet_mask, tx_success_mask):
        """Helper: one TX agent, one step, returns obs ret_c."""
        buf = jnp.array([[5, EMPTY, EMPTY]])
        obs = jnp.zeros((1, 3, OBS_SIZE), dtype=jnp.float32)
        obs = obs.at[0, -1, ObsIdx.STATUS_RETRY_COUNTER].set(ret_c_init)
        obs_features = ObsFeatureInputs(
            traffic_mean_arrival_rate=jnp.array([1.0]),
            channel_occupancy_pct_window=jnp.array([0.0]),
            channel_collisions_pct_window=jnp.array([0.0]),
            back_pct=jnp.array([0.0]),
            unique_ltc_tx_window=jnp.array([0.0]),
            cs_tx_same_type_now=jnp.array([0.0]),
            tx_collision_other_now=jnp.array([0.0]),
        )
        obs_config = ObsFeatureConfig(enable_cs_tx_same_type=False, enable_tx_collision_other=False)
        result_obs, _, _ = process_output(
            buffer_states=buf, new_buffer_states=buf, new_buffer_birth_states=jnp.array([[-1,-1,-1]]),
            power_states=jnp.array([100]), channel_state=channel_state,
            obs=obs, actions=jnp.array([Actions.TX.value]), terminals=jnp.array([False]),
            key=KEY, current_step=0, obs_features=obs_features, obs_config=obs_config,
            tx_packet_mask=jnp.array([tx_packet_mask]),
            tx_success_mask=jnp.array([tx_success_mask]),
        )
        return int(result_obs[0, -1, ObsIdx.STATUS_RETRY_COUNTER])

    def test_tx_sub_window_collision_increments_ret_c(self):
        r = self._make_tx_process_output(ret_c_init=2, channel_state=-1, tx_packet_mask=True, tx_success_mask=False)
        self.assertEqual(r, 3)

    def test_tx_sub_window_collision_at_max_resets_ret_c(self):
        r = self._make_tx_process_output(ret_c_init=MAX_RETRANSMISSION, channel_state=-1, tx_packet_mask=True, tx_success_mask=False)
        self.assertEqual(r, 0)

    def test_tx_sub_window_collision_below_max(self):
        r = self._make_tx_process_output(ret_c_init=7, channel_state=-1, tx_packet_mask=True, tx_success_mask=False)
        self.assertEqual(r, 8)

    def test_tx_sub_window_success_resets_ret_c(self):
        r = self._make_tx_process_output(ret_c_init=3, channel_state=1, tx_packet_mask=True, tx_success_mask=True)
        self.assertEqual(r, 0)

    def test_tx_mid_macro_ret_c_unchanged(self):
        r = self._make_tx_process_output(ret_c_init=5, channel_state=-1, tx_packet_mask=False, tx_success_mask=False)
        self.assertEqual(r, 5)

    def test_process_rl_output(self):
        buffer_states = jnp.array([
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [9, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
        ])
        new_buffer_birth_states = jnp.array([
            [-1, -1, -1],
            [-1, -1, -1],
            [3, -1, -1],
            [-1, -1, -1],
        ], dtype=jnp.int32)
        new_buffer_states = jnp.array([
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [9, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
        ])
        power_states = jnp.array([100, 100, 100, 100])
        actions = jnp.array([Actions.TX.value, Actions.CS.value, Actions.TX.value, Actions.TX.value])
        terminals = jnp.array([False, False, False, False])
        channel_state = -1
        obs = jnp.zeros((4, 3, OBS_SIZE), dtype=jnp.float32)
        obs = obs.at[:, -1, ObsIdx.STATUS_RETRY_COUNTER].set(8)
        expected_R_0 = TX_COEX_COLLISION_PENALTY * 1.0 + TX_MAX_RETRANSMISSION_PENALTY  # -5.0 + -1.0
        expected_power = power_states + jnp.array([-TX_CONSUMPTION, -CS_CONSUMPTION, -TX_CONSUMPTION, -TX_CONSUMPTION])

        obs_features = ObsFeatureInputs(
            traffic_mean_arrival_rate=jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32),
            channel_occupancy_pct_window=jnp.full((4,), 0.25, dtype=jnp.float32),
            channel_collisions_pct_window=jnp.full((4,), 0.5, dtype=jnp.float32),
            back_pct=jnp.array([0.0, 0.6, 1.0, 0.0], dtype=jnp.float32),
            unique_ltc_tx_window=jnp.full((4,), 2.0, dtype=jnp.float32),
            cs_tx_same_type_now=jnp.array([0.0, 1.0, 0.0, 0.0], dtype=jnp.float32),
            tx_collision_other_now=jnp.array([1.0, 0.0, 1.0, 1.0], dtype=jnp.float32),
        )
        obs_config = ObsFeatureConfig(
            enable_cs_tx_same_type=True,
            enable_tx_collision_other=True,
        )

        # Agent 0: TX, collision, ret_c=8 >= MAX_RETRANSMISSION, coex collision → coex penalty + max-retry penalty
        done_now = jnp.array([True, True, True, True])
        k_tx = jnp.zeros(4, dtype=jnp.float32)
        k_coll_ltc = jnp.zeros(4, dtype=jnp.float32)
        k_coll_coex = jnp.array([1.0, 0.0, 1.0, 1.0], dtype=jnp.float32)  # coex collision for TX agents
        header_collision = jnp.zeros(4, dtype=bool)
        header_collision_other = jnp.zeros(4, dtype=bool)
        ack_collision = jnp.zeros(4, dtype=bool)
        ack_collision_other = jnp.zeros(4, dtype=bool)
        initial_ret_c = jnp.array([8, 0, 8, 8], dtype=jnp.int32)
        tx_started_empty = jnp.array([True, False, False, True], dtype=bool)

        result_obs, result_R, power = process_output(
            buffer_states=buffer_states,
            new_buffer_states=new_buffer_states,
            new_buffer_birth_states=new_buffer_birth_states,
            power_states=power_states,
            channel_state=channel_state,
            obs=obs,
            actions=actions,
            terminals=terminals,
            key=KEY,
            current_step=10,
            obs_features=obs_features,
            obs_config=obs_config,
            done_now=done_now,
            k_tx=k_tx,
            k_coll_ltc=k_coll_ltc,
            k_coll_coex=k_coll_coex,
            header_collision=header_collision,
            header_collision_other=header_collision_other,
            ack_collision=ack_collision,
            ack_collision_other=ack_collision_other,
            initial_ret_c=initial_ret_c,
            tx_started_empty=tx_started_empty,
        )

        self.assertEqual(result_obs.shape[-1], OBS_SIZE)
        self.assertEqual(result_obs[1, -1, ObsIdx.CHANNEL_LAST_CS_BUSY], 1.0)
        self.assertEqual(result_obs[1, -1, ObsIdx.CHANNEL_LAST_CS_TX_SAME_TYPE], -1.0)
        self.assertEqual(result_obs[0, -1, ObsIdx.CHANNEL_LAST_TX_COLLISION_OTHER], 1.0)
        self.assertEqual(result_obs[2, -1, ObsIdx.BUFFER_PACKET_COUNT], 1.0)
        self.assertEqual(result_obs[2, -1, ObsIdx.ACTION_BACK_PCT], 1.0)
        self.assertEqual(result_R[0], expected_R_0)
        self.assertTrue(jnp.array_equal(power, expected_power))

if __name__ == '__main__':
    unittest.main()
