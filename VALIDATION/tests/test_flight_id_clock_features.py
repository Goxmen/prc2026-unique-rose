import sys
from pathlib import Path
import unittest
import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import flight_id_clock_features as features


class PeerClock(unittest.TestCase):
    def test_strict_leave_one_out_missing_and_month_scope(self):
        rows=pd.DataFrame(dict(dep_id=range(1,8),month_key=['2025-01']*5+['2025-02']*2,
            flight_id=[10,20,30,40,50,15,25],iobt=[100.,500.,np.nan,400.,800.,10000.,20000.]))
        with duckdb.connect(config={'threads':1}) as con:
            con.register('eligible',rows)
            f=features.table(con).set_index('mvt_id')
        # Query20 cannot reuse its own500 clock; peers10/40 interpolate to200.
        self.assertEqual(f.loc[2,'idclock_iobt_residual'],-300.)
        self.assertEqual(f.loc[2,'idclock_neighbor_span'],300.)
        self.assertEqual(f.loc[3,'idclock_neighbor_span'],-100.)
        self.assertTrue(np.isnan(f.loc[3,'idclock_iobt_residual']))
        self.assertTrue(f.loc[[1,5,6,7],features.FEATURES].isna().all().all())
        self.assertEqual(f.loc[4,'lo_id'],20)
        self.assertEqual(f.loc[4,'hi_id'],50)

    def test_all_missing_clock_never_imputed(self):
        rows=pd.DataFrame(dict(dep_id=[1,2],month_key=['2025-01']*2,flight_id=[10,20],iobt=[np.nan,np.nan]))
        with duckdb.connect(config={'threads':1}) as con:
            con.register('eligible',rows)
            f=features.table(con)
        self.assertTrue(f[features.FEATURES].isna().all().all())


if __name__=='__main__':unittest.main()
