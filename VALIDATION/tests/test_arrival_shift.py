"""Guard arrival clock direction, weather location and conditional denominators."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import investigate_arrival_shift as audit
from check_arrival_snow_severity import snowfall_grid


class ArrivalShiftTests(unittest.TestCase):
    def test_traffic_excludes_self_and_simultaneous_events(self):
        events=pd.to_datetime(['2025-01-01 11:29','2025-01-01 11:30',
                               '2025-01-01 11:59','2025-01-01 12:00','2025-01-01 12:00'])
        self.assertEqual(audit.preceding_counts(pd.to_datetime(['2025-01-01 12:00']),events)[0],2)

    def populations(self):
        def make(a0,a1,b0,b1):
            groups=['A']*(a0+a1)+['B']*(b0+b1)
            y=[600]*a0+[2400]*a1+[600]*b0+[2400]*b1
            return pd.DataFrame({'group':groups,'taxi_sec':y,
                                 'date':pd.Timestamp('2025-01-01')+pd.to_timedelta(np.arange(len(y))%4,unit='D')})
        return make(81,9,2,8),make(45,5,10,40)

    def test_standardization_removes_pure_composition_change(self):
        s,t=self.populations()
        r,_,_=audit.standardize(s,t,['group'],1800,min_source=1,min_target=1)
        self.assertAlmostEqual(r['source_all_tail_pct'],17)
        self.assertAlmostEqual(r['target_all_tail_pct'],45)
        self.assertAlmostEqual(r['expected_target_tail_pct'],45)
        self.assertAlmostEqual(r['residual_pp'],0)
        self.assertAlmostEqual(r['composition_pp'],28)

    def test_unsupported_target_rows_remain_in_coverage_denominator(self):
        s,t=self.populations();t.loc[0:24,'group']='unseen'
        r,_,_=audit.standardize(s,t,['group'],1800,min_source=1,min_target=1)
        self.assertEqual(r['target_n'],100)
        self.assertEqual(r['target_supported_n'],75)
        self.assertEqual(r['target_coverage_pct'],75)

    def test_block_resampling_preserves_calendar_day_count(self):
        w=audit.day_weights(31,100,3,np.random.default_rng(4))
        np.testing.assert_array_equal(w.sum(axis=1),31)

    def test_traffic_bins_do_not_depend_on_other_rows_missingness(self):
        f=pd.DataFrame({'traffic_prev30':[1.,5.,10.,np.nan], 'origin':['A']*4,
            'snow_prev24h':[0.]*4,'wx_temp_c':[5.]*4,'wx_visibility_m':[10000.]*4,
            'wx_gust_kt':[5.]*4})
        s,t,_=audit.bins(f,f.iloc[:3].copy())
        self.assertEqual(s.traffic.iloc[:3].tolist(),t.traffic.tolist())

    def test_weather_uses_arrival_airport_and_no_future_report(self):
        rows=[]
        for station,valid,temp,code in [('EHAM','2025-01-01 12:00',32,'-SN'),
                                       ('EHAM','2025-01-01 13:00',90,'GS'),
                                       ('EDDF','2025-01-01 12:00',86,'')]:
            row=dict(station=station,valid=pd.Timestamp(valid,tz='UTC'),tmpf=temp,
                dwpf=30,sknt=5,gust=np.nan,drct=180,vsby=10,metar=station+' CAVOK',wxcodes=code)
            row.update({f'skyc{i}':None for i in range(1,5)})
            row.update({f'skyl{i}':np.nan for i in range(1,5)})
            rows.append(row)
        arr=pd.DataFrame({'mvt_id':[1.],'airport':['EHAM'],'origin':['EDDF'],
                          'landing':pd.to_datetime(['2025-01-01 12:30'])})
        f=audit.attach_weather(arr,pd.DataFrame(rows))
        self.assertEqual(f.wx_temp_c.iloc[0],0)
        self.assertEqual(f.strict_snow.iloc[0],1)
        self.assertEqual(f.ice_or_small_hail.iloc[0],0)
        self.assertEqual(f.wx_age_min.iloc[0],30)

    def test_snow_duration_is_a_past_window_not_a_future_weather_summary(self):
        times=pd.date_range('2025-01-01',periods=60,freq='30min',tz='UTC')
        metar=pd.DataFrame({'station':'EHAM','valid':times,
            'wxcodes':['SN' if 8<=i<20 else '' for i in range(60)]})
        before=snowfall_grid(metar)
        metar.loc[49:,'wxcodes']='SN'
        after=snowfall_grid(metar)
        self.assertEqual(before.snow_fraction24h.iloc[48],.25)
        np.testing.assert_allclose(before.snow_fraction24h.iloc[:49],after.snow_fraction24h.iloc[:49],equal_nan=True)

    def test_missing_reports_do_not_become_snow_free_hours(self):
        metar=pd.DataFrame({'station':['EHAM']*2,
            'valid':pd.to_datetime(['2025-01-01','2025-01-03'],utc=True),
            'wxcodes':['SN','']})
        g=snowfall_grid(metar)
        self.assertTrue(pd.isna(g.snow_fraction24h.iloc[48]))


if __name__=='__main__':unittest.main()
