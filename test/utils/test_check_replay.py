"""The replay guard: reject a policy that collided its way to a standstill."""

import unittest
from types import SimpleNamespace

import numpy as np

from ltc.utils.check_replay import COLLISION, collision_share


def history(channel):
    return SimpleNamespace(channel_state=np.asarray(channel))


class TestCollisionShare(unittest.TestCase):
    def test_measures_the_tail_not_the_whole_run(self):
        # Congested start, clean second half: a replay settling must not read as dead.
        channel = np.concatenate([np.full(100, COLLISION), np.ones(100, dtype=int)])
        self.assertAlmostEqual(collision_share(history(channel), 0.5), 0.0)
        self.assertAlmostEqual(collision_share(history(channel), 1.0), 0.5)

    def test_deadlock_reads_as_one(self):
        self.assertAlmostEqual(collision_share(history(np.full(200, COLLISION))), 1.0)

    def test_idle_is_not_a_collision(self):
        self.assertAlmostEqual(collision_share(history(np.zeros(200, dtype=int))), 0.0)

    def test_accepts_the_recorded_epoch_layout(self):
        # Histories arrive as [n_epochs, n_steps]; the share is over the flat timeline.
        channel = np.full((4, 50), COLLISION)
        channel[:, :25] = 1
        self.assertAlmostEqual(collision_share(history(channel), 1.0), 0.5)

    def test_empty_history_does_not_divide_by_zero(self):
        self.assertEqual(collision_share(history(np.zeros(0, dtype=int))), 0.0)


if __name__ == '__main__':
    unittest.main()
