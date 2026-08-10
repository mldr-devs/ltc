import unittest

import jax.numpy as jnp

from ltc.sim.constants import Actions, Features
from ltc.sim.features import raw_action, select_features


def make_obs(rows):
    return jnp.array(rows)


class SelectFeaturesTestCase(unittest.TestCase):
    def test_selects_in_the_requested_order(self):
        obs = make_obs([[1, -1, 3, 4, 1, 0]])
        selected = select_features(obs, (Features.CHANNEL, Features.BUFFER))
        self.assertTrue(jnp.array_equal(selected, jnp.array([[-1, 1]])))

    def test_full_selection_is_the_identity(self):
        obs = make_obs([[1, -1, 3, 4, 1, 0], [0, 0, 0, 0, 0, 1]])
        self.assertTrue(jnp.array_equal(select_features(obs, tuple(Features)), obs))

    def test_keeps_the_window_dimension(self):
        obs = jnp.zeros((7, len(Features)), dtype=int)
        self.assertEqual(select_features(obs, (Features.BUFFER,)).shape, (7, 1))


class RawActionTestCase(unittest.TestCase):
    def test_rebuilds_the_action_index(self):
        obs = make_obs([
            [0, 0, 0, 0, 1, 0],  # action_tx
            [0, 0, 0, 0, 0, 1],  # action_cs
            [0, 0, 0, 0, 0, 0],  # neither -> idle
        ])
        expected = jnp.array([Actions.TX.value, Actions.CS.value, Actions.IDLE.value])
        self.assertTrue(jnp.array_equal(raw_action(obs), expected))


class AgentFeatureContractTestCase(unittest.TestCase):
    """Each ported agent must unpack exactly as many features as it declares."""

    def test_declared_widths_match_the_unpacking(self):
        from ltc.agents import SRJaxAgent
        from ltc.baselines import (
            ALOHAQTF, DCF, DLMANetwork, DOS, EBALOHA, FWALOHA, Idle_sense, QALOHA, StatelessQLearning, TDMA
        )

        expected_widths = {
            DCF: 3, Idle_sense: 3, DOS: 2, ALOHAQTF: 2,
            QALOHA: 1, EBALOHA: 2, FWALOHA: 1, TDMA: 1, StatelessQLearning: 1,
            SRJaxAgent: len(Features), DLMANetwork: 4,
        }
        obs = jnp.arange(len(Features))[None, :]

        for agent, width in expected_widths.items():
            with self.subTest(agent=agent.__name__):
                self.assertEqual(len(agent.FEATURES), width)
                self.assertEqual(select_features(obs, agent.FEATURES).shape[-1], width)
                for feature in agent.FEATURES:
                    self.assertIn(feature, tuple(Features))


class DLMAObservationTestCase(unittest.TestCase):
    def test_produces_the_channel_action_buffer_triple(self):
        from ltc.baselines.dlma import dlma_observation

        obs = make_obs([[1, -1, 5, 6, 1, 0]])  # buffer=1, channel=-1, action=TX
        expected = jnp.array([[-1, Actions.TX.value, 1]])
        self.assertTrue(jnp.array_equal(dlma_observation(obs), expected))


if __name__ == '__main__':
    unittest.main()
