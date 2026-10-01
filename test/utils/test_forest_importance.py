"""The importance grid: features on y, window slots on x, newest slot last."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np

from ltc.utils.forest_importance import DYNAMIC_RANGE_DECADES, importance_grid, shared_scale
from ltc.symbolic.history2csv import FEATURE_NAMES, build_column_names


def fake_forest(importances, path):
    joblib.dump(SimpleNamespace(feature_importances_=np.asarray(importances)), path)
    return path


class TestImportanceGrid(unittest.TestCase):
    def test_orientation_follows_the_column_names(self):
        # Mark one column and check it lands where its name says it should. A
        # transposed reshape still produces a plausible-looking heatmap.
        window = 10
        columns = build_column_names(window)
        for name in ('buffer_9', 'no_tx_0', 'action_cs_4'):
            importances = np.zeros(len(columns))
            importances[columns.index(name)] = 1.0
            with tempfile.TemporaryDirectory() as d:
                grid, got_window = importance_grid(fake_forest(importances, Path(d) / 'f.pkl'))
            feature, slot = name.rsplit('_', 1)
            self.assertEqual(got_window, window)
            self.assertEqual(grid.shape, (len(FEATURE_NAMES), window))
            self.assertEqual(grid[FEATURE_NAMES.index(feature), int(slot)], 1.0)
            self.assertEqual(grid.sum(), 1.0)


class TestSharedScale(unittest.TestCase):
    def test_floor_is_a_fixed_number_of_decades_below_the_peak(self):
        # One noise cell must not stretch the ramp over the whole figure.
        grids = [np.array([[1.0, 1e-12]]), np.array([[0.5, 0.25]])]
        norm = shared_scale(grids)
        self.assertEqual(norm.vmax, 1.0)
        self.assertAlmostEqual(np.log10(norm.vmax / norm.vmin), DYNAMIC_RANGE_DECADES)

    def test_a_narrow_range_is_not_widened(self):
        norm = shared_scale([np.array([[1.0, 0.1]])])
        self.assertAlmostEqual(norm.vmin, 0.1)

    def test_scale_is_shared_across_panels(self):
        norm = shared_scale([np.array([[1e-3]]), np.array([[1.0]])])
        self.assertEqual(norm.vmax, 1.0)


if __name__ == '__main__':
    unittest.main()
