"""Guard the stand position inputs and the hidden-stand test's selection and blanking."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2] / 'pipeline')]
import stand_features as sf
import stand_geometry as sg
import stand_coords_run as run


def tables():
    coords = pd.DataFrame({'airport': ['AAAA', 'AAAA', 'BBBB'], 'stand': ['1', '2', '1'],
                           'lat': [50.001, 50.0, 40.0], 'lon': [8.0, 8.001, 3.0]})
    ends = pd.DataFrame({'airport': ['AAAA', 'AAAA', 'BBBB', 'BBBB'], 'runway': ['09', '27', '18', '36'],
                         'lat': [50.0, 50.0, 40.01, 39.99], 'lon': [7.99, 8.01, 3.0, 3.0],
                         'displaced_m': [0.0] * 4})
    return coords, ends, ends.groupby('airport')[['lat', 'lon']].mean()


class StandFeatureTests(unittest.TestCase):
    raw = pd.DataFrame({'ADEP_mvt': ['AAAA', 'AAAA', 'AAAA', 'BBBB', 'AAAA'],
                        'STAND_mvt': ['1', '2', '9', '1', None],
                        'RUNWAY_mvt': ['09', '27', '09', 'NA', '27']}, index=[10, 11, 12, 13, 14])

    def features(self):
        X = pd.DataFrame({'a': range(5)}, index=[5, 6, 7, 8, 9])
        return sf.add_features(X, self.raw, tables())

    def test_offsets_from_the_runway_end_centre(self):
        out = self.features()
        self.assertAlmostEqual(out.sc_x_m.iloc[0], 0.0, places=6)
        self.assertAlmostEqual(out.sc_y_m.iloc[0], 0.001 * sg.M_PER_DEG_LAT, places=6)
        self.assertGreater(out.sc_x_m.iloc[1], 0)
        self.assertAlmostEqual(out.sc_y_m.iloc[1], 0.0, places=6)

    def test_distance_to_the_named_runway_end(self):
        out = self.features()
        self.assertAlmostEqual(out.sc_rwy_end_m.iloc[0], float(sg.metres(50.001, 8.0, 50.0, 7.99)))
        self.assertAlmostEqual(out.sc_rwy_end_m.iloc[1], float(sg.metres(50.0, 8.001, 50.0, 8.01)))

    def test_missing_stand_or_runway_gives_missing_inputs(self):
        out = self.features()
        self.assertTrue(out.iloc[2][sf.FEATURES].isna().all())      # stand not in the table
        self.assertTrue(out.iloc[4][sf.FEATURES].isna().all())      # no stand recorded
        self.assertTrue(np.isfinite(out.sc_x_m.iloc[3]))            # stand known ...
        self.assertTrue(np.isnan(out.sc_rwy_end_m.iloc[3]))         # ... runway 'NA' is not

    def test_rows_stay_aligned_and_inputs_untouched(self):
        out = self.features()
        self.assertEqual(out.index.tolist(), [5, 6, 7, 8, 9])
        self.assertEqual(out.a.tolist(), list(range(5)))
        with self.assertRaises(ValueError):
            sf.add_features(pd.DataFrame({'a': [1]}), self.raw, tables())


class HiddenStandTests(unittest.TestCase):
    keys = pd.Series([f'AP{i % 10}:{i}' for i in range(5000)] + [pd.NA], dtype='string')

    def test_share_is_about_ten_percent_and_deterministic(self):
        a = run.is_hidden(self.keys, 'v3_2025-06')
        self.assertTrue(np.array_equal(a, run.is_hidden(self.keys, 'v3_2025-06')))
        self.assertTrue(0.08 < a[:-1].mean() < 0.12)
        self.assertFalse(a[-1])

    def test_each_window_hides_different_stands(self):
        a, b = run.is_hidden(self.keys, 'v3_2025-06'), run.is_hidden(self.keys, 'v3_2025-07')
        self.assertLess((a & b).sum(), 0.3 * a.sum())

    def test_a_stand_is_hidden_on_every_row_or_none(self):
        keys = pd.Series(['X:1'] * 3 + ['X:2'] * 3, dtype='string')
        mask = run.is_hidden(pd.concat([keys, self.keys], ignore_index=True), 'w')[:6]
        self.assertEqual(len(set(mask[:3])), 1)
        self.assertEqual(len(set(mask[3:])), 1)

    def test_blanking_touches_only_hidden_rows_and_the_two_inputs(self):
        V = pd.DataFrame({'stand': pd.Categorical(['A:1', 'A:2', 'A:3']),
                          'stand_taxi_90d': [600.0, 700.0, 800.0], 'sc_x_m': [1.0, 2.0, 3.0]})
        out = run.blank_hidden(V, np.array([False, True, False]))
        self.assertTrue(pd.isna(out.stand.iloc[1]) and np.isnan(out.stand_taxi_90d.iloc[1]))
        self.assertEqual(out.stand.iloc[[0, 2]].tolist(), ['A:1', 'A:3'])
        self.assertEqual(out.stand_taxi_90d.iloc[[0, 2]].tolist(), [600.0, 800.0])
        self.assertEqual(out.sc_x_m.tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(list(out.stand.cat.categories), ['A:1', 'A:2', 'A:3'])
        self.assertEqual(V.stand.iloc[1], 'A:2')                    # input frame unchanged

    def test_arms_differ_only_by_the_stand_inputs(self):
        self.assertEqual(run.CANDIDATE[:len(run.CONTROL)], run.CONTROL)
        self.assertEqual(run.CANDIDATE[len(run.CONTROL):], sf.FEATURES)
        self.assertFalse(set(sf.FEATURES) & set(run.CLASSIFIER))


@unittest.skipUnless(sf.COORDS.exists() and sf.ENDS.exists(), 'stand tables not built')
class RealTableTests(unittest.TestCase):
    def test_munich_new_stand_is_between_its_runways(self):
        raw = pd.DataFrame({'ADEP_mvt': ['EDDM', 'EDDM'], 'STAND_mvt': ['105', '105'],
                            'RUNWAY_mvt': ['26R', '08R']})
        out = sf.add_features(pd.DataFrame(index=range(2)), raw)
        self.assertTrue(((out.sc_rwy_end_m > 500) & (out.sc_rwy_end_m < 5000)).all())
        self.assertLess(abs(out.sc_y_m.iloc[0]), 1500)            # the terminal sits between the runways


if __name__ == '__main__':
    unittest.main()
