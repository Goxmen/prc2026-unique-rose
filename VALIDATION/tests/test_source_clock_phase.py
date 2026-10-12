import sys
from pathlib import Path
import unittest
import tempfile

import duckdb

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import source_clock_phase as phase


class SourceClockPhaseTest(unittest.TestCase):
    def test_missing_midnight_and_no_target_dependency(self):
        raw=pd.DataFrame({column:pd.to_datetime(['2025-03-01T23:59:59Z','2025-04-01T00:00:00Z',None])
                          for column in phase.CLOCKS.values()},index=[9,4,5])
        got=phase.transform(raw)
        self.assertEqual(list(got),phase.FEATURES)
        self.assertEqual(got.loc[9,'scp_aobt3_second'],59)
        self.assertEqual(got.loc[9,'scp_aobt3_minute_mod5'],4)
        self.assertEqual(got.loc[9,'scp_aobt3_minute_mod15'],14)
        self.assertTrue(got.loc[4].eq(0).all())
        self.assertTrue(got.loc[5].isna().all())
        changed=raw.assign(MVT_TIME_UTC_mvt='2026-01-01',BLOCK_TIME_UTC_mvt='2026-01-02',TAXITIME_SEC_mvt=999999)
        pd.testing.assert_frame_equal(got,phase.transform(changed),check_exact=True)
        self.assertEqual(set(str(d) for d in got.dtypes),{'float32'})

    def test_independent_row_order_and_original_missing(self):
        raw=pd.DataFrame({column:pd.to_datetime(['2025-03-01T06:20:17Z',None]) for column in phase.CLOCKS.values()})
        raw['SCHED_TIME_UTC_mvt']=pd.to_datetime(['2025-03-01T06:00:00Z']*2)
        expected=phase.transform(raw)
        pd.testing.assert_frame_equal(expected.iloc[::-1],phase.transform(raw.iloc[::-1]))
        self.assertTrue(expected.iloc[1].isna().all())
        self.assertEqual(expected.iloc[0].scp_aobt3_minute_mod15,5)

    def test_sql_matches_independent_utc_scalar_at_boundaries(self):
        with tempfile.TemporaryDirectory() as temp:
            clock=pd.to_datetime(['2025-03-31T23:59:59Z','2025-04-01T00:00:00Z',None],utc=True)
            raw=pd.DataFrame({c:clock for c in phase.CLOCKS.values()})
            raw['MVT_ID_mvt']=[19,11,15];raw['ADEP_mvt']='EHAM';raw['PHASE_mvt']='DEP'
            raw['MVT_TIME_UTC_mvt']=pd.to_datetime(['2025-03-31T23:59:59Z','2025-04-01T00:00:00Z','2025-04-02T00:00:00Z'])
            raw['SCHED_TIME_UTC_mvt']=raw['MVT_TIME_UTC_mvt']
            path=Path(temp)/'source.parquet';raw.to_parquet(path,index=False)
            with duckdb.connect(config={'threads':1}) as con:
                con.execute("SET TimeZone='UTC'")
                got=phase.read_view(con,[path]).set_index('mvt_id')
            expected=phase.transform(raw).set_axis(raw.MVT_ID_mvt).sort_index()
            expected.index.name='mvt_id'
            pd.testing.assert_frame_equal(got[phase.FEATURES],expected,check_exact=True)
            self.assertEqual(got.loc[19,'month'],'2025-03')
            self.assertEqual(got.loc[11,'month'],'2025-04')


if __name__=='__main__':unittest.main()
