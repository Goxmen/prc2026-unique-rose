"""Checks for METAR parsing and the takeoff-time weather join."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import weather as w


def metar_rows(rows):
    base = dict(station='LFPG', tmpf=50.0, dwpf=40.0, relh=np.nan, drct=180.0, sknt=10.0, gust=np.nan,
                vsby=6.21, alti=np.nan, skyc1=pd.NA, skyc2=pd.NA, skyc3=pd.NA, skyc4=pd.NA,
                skyl1=np.nan, skyl2=np.nan, skyl3=np.nan, skyl4=np.nan, wxcodes=pd.NA, metar='', report_type='routine')
    frame = pd.DataFrame([{**base, **row} for row in rows])
    frame['valid'] = pd.to_datetime(frame['valid'], utc=True)
    return frame


class ReportTableTests(unittest.TestCase):
    def test_units_ceiling_and_weather_flags(self):
        table = w.report_table(metar_rows([
            dict(valid='2025-11-22 07:00', tmpf=28.4, dwpf=26.6, vsby=0.155, skyc1='VV ', wxcodes='FZFG', metar='x'),
            dict(valid='2025-11-22 07:30', vsby=np.nan, metar='LFPG 220730Z CAVOK', gust=25.0),
            dict(valid='2025-11-22 08:00', skyc1='FEW', skyl1=4000.0, skyc2='BKN', skyl2=1500.0,
                 wxcodes='-SN VCTS', metar='y'),
            dict(valid='2025-11-22 08:30', wxcodes='+RA BCFG', metar='z'),
        ]))
        np.testing.assert_allclose(table.wx_temp_c, [-2.0, 10.0, 10.0, 10.0], atol=1e-9)
        np.testing.assert_allclose(table.wx_visibility_m, [249.45, 10_000, 9994.03, 9994.03], atol=0.01)
        np.testing.assert_allclose(table.wx_ceiling_ft, [0.0, w.NO_CEILING_FT, 1500.0, np.nan])
        np.testing.assert_allclose(table.wx_gust_kt, [10.0, 25.0, 10.0, 10.0])
        self.assertEqual(table.wx_fog.tolist(), [1, 0, 0, 0])
        self.assertEqual(table.wx_freezing.tolist(), [1, 0, 0, 0])
        self.assertEqual(table.wx_snow.tolist(), [0, 0, 1, 0])
        self.assertEqual(table.wx_thunder.tolist(), [0, 0, 1, 0])
        self.assertEqual(table.wx_rain.tolist(), [0, 0, 0, 1])
        self.assertEqual(table.wx_heavy.tolist(), [0, 0, 0, 1])
        self.assertEqual(table.wx_deicing.tolist(), [1, 0, 0, 0])


class WeatherJoinTests(unittest.TestCase):
    def setUp(self):
        self.reports = w._with_window(w.report_table(metar_rows([
            dict(valid='2025-07-01 09:00', vsby=0.5, metar='a'),
            dict(valid='2025-07-01 10:00', vsby=3.0, metar='b'),
            dict(valid='2025-07-01 10:30', vsby=6.21, metar='c'),
            dict(valid='2025-07-01 11:00', vsby=0.1, metar='d'),
            dict(station='EGLL', valid='2025-07-01 10:50', tmpf=95.0, metar='e'),
        ])))

    def test_latest_report_at_or_before_takeoff_same_airport_only(self):
        raw = pd.DataFrame({
            'ADEP_mvt': ['LFPG', 'LFPG', 'LFPG', 'EGLL', 'LEMD'],
            'MVT_TIME_UTC_mvt': pd.to_datetime(['2025-07-01 10:45', '2025-07-01 10:30', '2025-07-01 12:31',
                                                '2025-07-01 10:40', '2025-07-01 10:45'], utc=True),
        }, index=[10, 11, 12, 13, 14])
        out = w.weather_features(raw, self.reports)
        self.assertEqual(out.index.tolist(), [10, 11, 12, 13, 14])
        np.testing.assert_allclose(out.wx_visibility_m.iloc[:2], [9994.03, 9994.03], atol=0.01)
        np.testing.assert_allclose(out.wx_age_min.iloc[:2], [15.0, 0.0])
        # 10:45 sees the 09:00 report inside its two-hour window; the 11:00 report is after takeoff
        self.assertAlmostEqual(float(out.wx_2h_min_visibility_m.iloc[0]), 804.67, places=1)
        self.assertTrue(out.iloc[2].isna().all(), 'report older than 90 minutes must not be used')
        self.assertTrue(out.iloc[3].isna().all(), 'EGLL report after takeoff must not be used')
        self.assertTrue(out.iloc[4].isna().all(), 'no report for that airport')

    def test_add_weather_requires_aligned_rows(self):
        raw = pd.DataFrame({'ADEP_mvt': ['LFPG'], 'MVT_TIME_UTC_mvt': pd.to_datetime(['2025-07-01 10:45'], utc=True)},
                           index=[5])
        with self.assertRaises(ValueError):
            w.add_weather(pd.DataFrame({'a': [1.0]}, index=[6]), raw, self.reports)
        X = w.add_weather(pd.DataFrame({'a': [1.0]}, index=[5]), raw, self.reports)
        self.assertEqual(X.columns.tolist(), ['a'] + w.WEATHER_FEATURES)


if __name__ == '__main__':
    unittest.main()
