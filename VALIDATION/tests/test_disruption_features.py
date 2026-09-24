"""Information-time, airport, coverage and boundary checks for the new inputs."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import disruption_features as f


class DisruptionTests(unittest.TestCase):
    def departures(self):
        return pd.DataFrame({'ADEP_mvt': ['EHAM', 'EDDF'],
                             'MVT_TIME_UTC_mvt': pd.to_datetime(['2025-01-02 12:00'] * 2, utc=True)}, index=[8, 3])

    def arrivals(self):
        return pd.DataFrame({'mvt_id': range(7), 'ADES_mvt': ['EHAM'] * 6 + ['EDDF'],
            'MVT_TIME_UTC_mvt': pd.to_datetime(['2025-01-02 ' + x for x in
                ['10:50', '10:29', '11:55', '10:00', '09:00', '11:00', '11:10']], utc=True),
            'taxi_sec': [600., 3600., 600., 7200., np.nan, -1., 1200.]})

    def test_completion_strict_boundary_airport_and_empty(self):
        out = f.arrival_features(self.departures(), self.arrivals())
        # 11:00 and 11:29 are included; 12:00 and 12:05 are not completed yet.
        self.assertEqual(out.loc[8, 'dis_arr_60m_count'], 2)
        self.assertEqual(out.loc[8, 'dis_arr_60m_mean_sec'], 2100)
        self.assertEqual(out.loc[8, 'dis_arr_60m_over30_share'], .5)
        self.assertEqual(out.loc[8, 'dis_arr_60m_over60_share'], 0)
        self.assertEqual(out.loc[3, 'dis_arr_60m_mean_sec'], 1200)
        empty = f.arrival_features(self.departures(), self.arrivals().iloc[:0])
        self.assertTrue(empty.dis_arr_60m_count.eq(0).all())
        self.assertTrue(empty.dis_arr_60m_mean_sec.isna().all())

    def test_future_completions_and_departure_labels_cannot_leak(self):
        raw, arr = self.departures(), self.arrivals()
        before = f.arrival_features(raw, arr)
        arr.loc[[2, 3], 'taxi_sec'] = [10000, 20000]
        raw['taxi_sec'] = [1e8, 1e9]
        pd.testing.assert_frame_equal(before, f.arrival_features(raw, arr))

    def test_source_scope_and_duplicate_guard(self):
        raw, arr = self.departures(), self.arrivals()
        limited = f.arrival_features(raw, arr.loc[arr.mvt_id.ne(1)])
        self.assertEqual(limited.loc[8, 'dis_arr_60m_count'], 1)
        with self.assertRaises(ValueError):
            f.arrival_features(raw, pd.concat([arr, arr.iloc[:1]]))

    def test_mean_cap_does_not_cap_tail_flag_or_availability(self):
        a = self.arrivals().iloc[:1].copy()
        a['MVT_TIME_UTC_mvt'] = pd.Timestamp('2025-01-02 08:30', tz='UTC')
        a['taxi_sec'] = 10800
        out = f.arrival_features(self.departures(), a)
        self.assertEqual(out.loc[8, 'dis_arr_60m_mean_sec'], 7200)
        self.assertEqual(out.loc[8, 'dis_arr_60m_over60_share'], 1)

    def metar(self):
        times = pd.date_range('2025-01-01', '2025-01-03', freq='30min', tz='UTC')
        return pd.concat([pd.DataFrame({'station': ap, 'valid': times, 'wxcodes': code})
                          for ap, code in [('EHAM', '-SN'), ('EDDF', 'GS')]], ignore_index=True)

    def test_snow_codes_coverage_and_no_future_report(self):
        metar, raw = self.metar(), self.departures()
        before = f.snow_features(raw, f.snow_grid(metar))
        self.assertEqual(before.loc[8, 'dis_snow_24h_fraction'], 1)
        self.assertEqual(before.loc[3, 'dis_snow_24h_fraction'], 0)
        self.assertTrue(before.dis_snow_24h_coverage.eq(1).all())
        metar.loc[metar.valid.gt(raw.MVT_TIME_UTC_mvt.iloc[0]), 'wxcodes'] = 'TSRA'
        pd.testing.assert_frame_equal(before, f.snow_features(raw, f.snow_grid(metar)))

    def test_missing_weather_and_simultaneous_reports(self):
        metar = self.metar()
        duplicate = metar.iloc[[0]].assign(wxcodes='')
        pd.testing.assert_frame_equal(f.snow_grid(metar), f.snow_grid(pd.concat([duplicate, metar])))
        sparse = metar.loc[metar.valid.dt.hour.eq(0)]
        out = f.snow_features(self.departures(), f.snow_grid(sparse))
        self.assertTrue(out.dis_snow_24h_fraction.isna().all())
        self.assertTrue(out.dis_snow_24h_coverage.lt(1).all())


if __name__ == '__main__':
    unittest.main()
