"""Time alignment and accumulation checks for the ERA5 candidate."""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import external_conditions as wx


class WeatherTests(unittest.TestCase):
    def setUp(self):
        self.times = pd.date_range('2025-01-01', periods=60, freq='h', tz='UTC')
        self.raw = pd.DataFrame({'time': self.times, **{v: np.arange(60, dtype=float) for v in wx.VARIABLES}})

    def test_accumulations_end_at_timestamp_and_require_full_history(self):
        f = wx.hourly_features(self.raw)
        self.assertTrue(np.isnan(f.om_precipitation_48h.iloc[46]))
        self.assertEqual(f.om_precipitation_48h.iloc[47], sum(range(48)))
        self.assertEqual(f.om_snowfall_6h.iloc[50], sum(range(45, 51)))
        self.raw.loc[51:, 'snowfall'] = 9999
        self.assertEqual(wx.hourly_features(self.raw).om_snowfall_6h.iloc[50], sum(range(45, 51)))

    def test_join_airport_time_order_missing_and_stale(self):
        table = wx.hourly_features(self.raw).assign(airport='EHAM')
        raw = pd.DataFrame(dict(ADEP_mvt=['EHAM','EGLL','EHAM','EHAM','EHAM'],
            MVT_TIME_UTC_mvt=[self.times[50]+pd.Timedelta(minutes=59), self.times[50],
                              self.times[49], pd.NaT, self.times[-1]+pd.Timedelta(hours=2)]), index=[7,2,9,4,5])
        f = wx.weather_join(raw, table)
        self.assertTrue(f.index.equals(raw.index))
        self.assertEqual(f.om_temperature_2m.iloc[0], 50)
        self.assertEqual(f.om_temperature_2m.iloc[2], 49)
        self.assertTrue(f.iloc[[1,3,4]].isna().all().all())

    def test_missing_hour_rejected(self):
        with self.assertRaises(AssertionError): wx.hourly_features(self.raw.drop(index=20))

    def test_local_timezone_does_not_shift_weather_hour(self):
        table = wx.hourly_features(self.raw).assign(airport='EHAM')
        utc = self.times[47] + pd.Timedelta(minutes=30)
        local = utc.tz_convert('Europe/Amsterdam')
        self.assertNotEqual(local.day, utc.day)
        raw = pd.DataFrame(dict(ADEP_mvt=['EHAM'], MVT_TIME_UTC_mvt=[local]))
        self.assertEqual(wx.weather_join(raw, table).om_temperature_2m.iloc[0], 47)


if __name__ == '__main__': unittest.main()
