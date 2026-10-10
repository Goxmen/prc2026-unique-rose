"""Source aggregation aliases and fold-dependent arrival comparisons."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import callsign_clocks as cc


def ms(text):
    return pd.Timestamp(text).value//1_000_000


class CallsignClockTests(unittest.TestCase):
    def test_midnight_bridge_preserves_previous_retained_start(self):
        current=[(ms('2025-01-01T23:50Z'),ms('2025-01-01T23:59:50Z'),1,'2025-01-01'),
                 (ms('2025-01-02T00:00Z'),ms('2025-01-02T00:20Z'),1,'2025-01-02')]
        row=cc.row_clocks(ms('2025-01-02T00:10Z'),'2025-01-02',current,[])
        self.assertEqual(row[0],ms('2025-01-01T23:50Z'))
        self.assertEqual(row[1],row[0])
        self.assertEqual(row[4],0)

    def test_midnight_gap_is_not_bridged_and_is_censored(self):
        current=[(ms('2025-01-01T20:00Z'),ms('2025-01-01T21:00Z'),1,'2025-01-01'),
                 (ms('2025-01-02T00:00Z'),ms('2025-01-02T00:20Z'),1,'2025-01-02')]
        row=cc.row_clocks(ms('2025-01-02T00:10Z'),'2025-01-02',current,[])
        self.assertEqual(row[0],ms('2025-01-02T00:00Z'))
        self.assertEqual(row[4],1)

    def test_squawk_reset_is_separate_and_overlap_is_flagged(self):
        t=ms('2025-01-02T12:20Z')
        current=[(t-1200000,t-20000,1,'2025-01-02'),
                 (t-100000,t+60000,2,'2025-01-02')]
        row=cc.row_clocks(t,'2025-01-02',current,[(t-1000000,t-500000)])
        self.assertEqual(row[0],t-1200000)
        self.assertEqual(row[1],t-100000)
        self.assertEqual(row[2],t-500000)
        self.assertEqual(row[3],1)

    def test_same_day_repeated_departure_exposes_collapsed_episode(self):
        t=ms('2025-01-02T12:20Z');start=t-5000000
        row=cc.row_clocks(t,'2025-01-02',[(start,t+60000,1,'2025-01-02')],[],t-2000000)
        self.assertEqual(row[3],1)
        missing=cc.row_clocks(t,'2025-01-02',[],[])
        self.assertTrue(np.isnan(missing).all())

    def test_other_call_overlap_extending_after_takeoff_is_ambiguous(self):
        t=ms('2025-01-02T12:20Z')
        row=cc.row_clocks(t,'2025-01-02',[(t-600000,t+60000,1,'2025-01-02')],
                          [(t-1200000,t+100000)])
        self.assertTrue(np.isnan(row[2]))
        self.assertEqual(row[3],1)

    def test_stability_filter_and_fold_arrival_missingness(self):
        t=ms('2025-01-02T12:20Z')
        q=pd.DataFrame({'mvt_id':[1,2],'hex':[100,200],'cs':['TEST','SHORT'],
                        'to_ms':[t,t],'previous_departure_ms':[np.nan,np.nan]})
        c=pd.DataFrame({'hex':[100,200],'callsign':['TEST','SHORT'],'records':[3,2],
            'first_ms':[t-1000000]*2,'last_ms':[t+100000]*2,'squawk':[1,1],
            'source_day':['2025-01-02']*2})
        raw=cc.day_clocks(q,c,'2025-01-02').set_index('mvt_id')
        self.assertEqual(raw.loc[1,'cc_call_start_sec'],1000)
        self.assertTrue(np.isnan(raw.loc[2,'cc_call_start_sec']))
        scoped=pd.DataFrame({'ar_prev_inblock_age_sec':[500,np.nan]},index=pd.Index([1,2],name='mvt_id'))
        got=cc.fold_features(raw,scoped)
        self.assertEqual(got.loc[1,'cc_before_own_arrival'],1)
        self.assertTrue(np.isnan(got.loc[2,'cc_before_own_arrival']))
        scoped.loc[1,'ar_prev_inblock_age_sec']=1500
        self.assertEqual(cc.fold_features(raw,scoped).loc[1,'cc_before_own_arrival'],0)

    def test_missing_identity_keeps_all_six_features_missing(self):
        t=ms('2025-01-02T12:20Z')
        q=pd.DataFrame({'mvt_id':[1],'hex':[np.nan],'cs':['TEST'],
                        'to_ms':[t],'previous_departure_ms':[np.nan]})
        c=pd.DataFrame({'hex':[100],'callsign':['TEST'],'records':[3],
            'first_ms':[t-1000000],'last_ms':[t+100000],'squawk':[1],
            'source_day':['2025-01-02']})
        raw=cc.day_clocks(q,c,'2025-01-02').set_index('mvt_id')
        scoped=pd.DataFrame({'ar_prev_inblock_age_sec':[500]},index=pd.Index([1],name='mvt_id'))
        self.assertTrue(cc.fold_features(raw,scoped).iloc[0].isna().all())


if __name__ == '__main__':
    unittest.main()
