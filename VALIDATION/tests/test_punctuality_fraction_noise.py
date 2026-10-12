"""Bounded synthetic checks; these never read competition rows or labels."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import punctuality_fraction_noise as pilot


class PunctualityNoiseTests(unittest.TestCase):
    def test_bias_excludes_entire_query_month(self):
        f=pd.DataFrame([dict(month=m,day=m+f'-{d:02d}',own_otp=.5,
            public_otp=.5+offset) for m,offset in zip(pilot.COMMON,[.1,.2,.3,.4]) for d in [1,2]])
        before,records=pilot.crossfit(f)
        changed=f.copy();changed.loc[changed.month.eq(pilot.COMMON[0]),'public_otp']+=10
        after,new=pilot.crossfit(changed)
        for key in ['bias','reference_days','reference_months','reference_mad']:
            self.assertEqual(records[0][key],new[0][key])
        self.assertAlmostEqual(records[0]['bias'],.3)
        self.assertEqual(records[0]['reference_months'],pilot.COMMON[1:])
        self.assertAlmostEqual(after.loc[0,'lomo_error']-before.loc[0,'lomo_error'],10)

    def test_exact_support_and_normal_probability(self):
        def support(*args):
            return np.array([2999.,3000.,3900.]),np.array([.2,.3,.5]),None
        with patch.object(pilot.retained,'support',side_effect=support) as mocked:
            p,load=pilot.prior_otp([1900.,3900.],[1000.,1500.],['a','a'],{'a':{'support':{}}})
        np.testing.assert_array_equal(p,[.5,.8])
        self.assertGreater(load[0],0);self.assertEqual(load[1],0)
        self.assertEqual(mocked.call_count,1)

    def test_projection_never_admits_held_labels_and_unknown_rules_retained(self):
        with tempfile.TemporaryDirectory() as folder,duckdb.connect(config={'threads':1}) as con:
            con.execute("SET TimeZone='UTC'")
            labels=[];inputs=[]
            for i,m in enumerate(pilot.COMMON+pilot.HELD):
                for j,rule in enumerate(['I',None,'V']):
                    mid=10*i+j;held=m in pilot.HELD
                    labels.append(dict(mvt_id=mid,airport='LIRF',day=m+'-01',month=m,
                        gap=1000.,y=np.nan if held else 700.,fltrule=rule,takeoff_epoch=1000.,true_day=m+'-01'))
                    inputs.append(dict(mvt_id=mid,airport='LIRF',day=m+'-01',gap=1000.,clock=600.,fltrule=rule))
            lp=Path(folder)/'label.parquet';ip=Path(folder)/'input.parquet'
            pd.DataFrame(labels).to_parquet(lp);pd.DataFrame(inputs).to_parquet(ip)
            with patch.object(pilot,'LABELS',lp),patch.object(pilot,'PUBLIC_INPUTS',ip):
                kept,raw=pilot.projected(con)
            self.assertEqual(len(kept),8);self.assertEqual(set(kept.month),set(pilot.COMMON))
            self.assertEqual(kept.fltrule.isna().sum(),4);self.assertEqual(len(raw),18)

    def test_arrival_day_label_scope_missingness_and_no_identity_gate(self):
        with tempfile.TemporaryDirectory() as folder,duckdb.connect(config={'threads':1}) as con:
            con.execute("SET TimeZone='UTC'")
            rows=[];days=[]
            for i,m in enumerate(pilot.COMMON):
                t=pd.Timestamp(m+'-01T23:55Z')
                for j,seconds in enumerate([900,901,None]):
                    rows.append(dict(mvt_id=i*10+j,phase='ARR',airport='LIRF',t=t,
                        inblock=t+pd.Timedelta(seconds=600) if seconds is not None else pd.NaT,
                        arrival_schedule=t+pd.Timedelta(seconds=600-seconds) if seconds is not None else t,
                        hex=None))
                # Poison departure fields and a held-month ARR must never matter.
                rows.append(dict(mvt_id=i*10+8,phase='DEP',airport='LIRF',t=t,
                    inblock=t,arrival_schedule=t,hex=1))
                days.append(dict(day=m+'-01',month=m,public_arrival_otp=.7,error=.1*i,lomo_error=.01*i))
            rows.append(dict(mvt_id=999,phase='ARR',airport='LIRF',t=pd.Timestamp('2025-01-01T12:00Z'),
                inblock=pd.NaT,arrival_schedule=pd.NaT,hex=None))
            path=Path(folder)/'arr.parquet';pd.DataFrame(rows).to_parquet(path)
            with patch.object(pilot,'ARRIVAL',path):
                result,stats=pilot.arrival_witness(con,pd.DataFrame(days))
            np.testing.assert_array_equal(result.arrivals,3)
            np.testing.assert_array_equal(result.valid_arrivals,2)
            np.testing.assert_array_equal(result.own_arrival_otp,.5)
            np.testing.assert_allclose(result.arrival_fraction_error,.2)
            self.assertEqual(stats[0]['days'],4)
            self.assertIsNone(stats[0]['statistics']['error']['correlation'])

    def test_overlap_calendars_retained_and_first_local_day_not_projected(self):
        with tempfile.TemporaryDirectory() as folder,duckdb.connect(config={'threads':1}) as con:
            con.execute("SET TimeZone='UTC'")
            folder=Path(folder)
            taxi=pd.DataFrame(dict(mvt_id=[1.,2.],airport='LIRF',runway='16R',stand='A',
                takeoff=pd.to_datetime(['2025-03-01T23:30Z','2025-03-01T00:00Z']),
                local_takeoff=['2025-03-02','2025-03-01'],y=[np.nan,np.nan]))
            cal=pd.DataFrame(dict(airport='LIRF',fold='2025-01',local_day=['2025-03-01','2025-03-02'],
                runway='16R',n=[10,20],valid=[9,18],total=100.,source_n=[11,21],source_mean=10.,
                omitted_count=[999.,-2.],omitted_sum=[99999.,-600.]))
            delay=pd.DataFrame(dict(APT_ICAO=['LIRF'],FLT_DATE=['2025-03-01'],FLT_DEP_IFR_2=[2],DLY_ALL_PRE_2=[30.]))
            tp=folder/'taxi.parquet';cp=folder/'cal.parquet';dp=folder/'delay.csv'
            taxi.to_parquet(tp);cal.to_parquet(cp);delay.to_csv(dp,index=False)
            train=pd.DataFrame(dict(mvt_id=[1.,2.],day='2025-03-01',true_day='2025-03-01',
                gap=[2000.,1000.],y=[1000.,500.]))
            days=pd.DataFrame(dict(day=['2025-03-01'],lomo_error=[.2]))
            with patch.object(pilot,'TAXI_ROWS',tp),patch.object(pilot,'TAXI_CAL',cp),patch.object(pilot,'DELAY',dp):
                te,de=pilot.overlap_diagnostics(con,train,days)
            first=te.loc[te.local_day.eq('2025-03-01')].iloc[0]
            second=te.loc[te.local_day.eq('2025-03-02')].iloc[0]
            self.assertTrue(np.isnan(first.source_count_error_per_row))
            self.assertEqual(second.source_count_error_per_row,.1)
            self.assertEqual(second.source_sum_error_per_row,30.)
            self.assertEqual(de.source_error_per_row.iloc[0],150.)
            self.assertTrue(de.boundary_day.iloc[0])


if __name__=='__main__':unittest.main()
