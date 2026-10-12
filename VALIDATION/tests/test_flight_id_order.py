"""Tiny synthetic contract checks; no competition rows are read."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import duckdb
import pandas as pd

SOURCE=Path(__file__).resolve().parents[1]/'tools/research_flight_id_order_20261012.py'
spec=importlib.util.spec_from_file_location('id_order',SOURCE)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class IdentityOrderChecks(unittest.TestCase):
    def test_own_clock_null_and_id_exclusions(self):
        rows=[]
        def row(mid,fid,phase,iobt,eobt):
            rows.append(dict(MVT_ID_mvt=float(mid),FLIGHT_ID_mvt=fid,PHASE_mvt=phase,
                ADEP_mvt='EDDF',ADES_mvt='EGLL',MVT_TIME_UTC_mvt=pd.Timestamp('2025-01-05T12:00Z'),
                IOBT_flt=pd.Timestamp(iobt) if iobt else pd.NaT,
                EOBT_1_flt=pd.Timestamp(eobt) if eobt else pd.NaT))
        row(1,100.,'DEP',None,'2025-01-05T10:00Z')
        row(2,100.,'ARR','2025-01-05T09:00Z','2025-01-05T10:00Z')
        row(3,200.,'DEP','2025-01-05T09:00Z','2025-01-05T10:00Z')
        row(4,200.,'ARR','2025-01-05T09:01Z','2025-01-05T10:00Z')
        row(5,300.,'DEP',None,None);row(6,300.,'DEP',None,None)
        row(7,400.5,'DEP',None,None);row(8,None,'DEP',None,None)
        with tempfile.TemporaryDirectory() as folder,duckdb.connect(config={'threads':1}) as con:
            con.execute("SET TimeZone='UTC'")
            p=Path(folder)/'inputs.parquet';pd.DataFrame(rows).to_parquet(p,index=False)
            module.make_source(con,p,['2025-01'])
            kept=con.execute('SELECT flight_id,iobt,eobt FROM eligible ORDER BY flight_id').fetchall()
            self.assertEqual([r[0] for r in kept],[100]);self.assertIsNone(kept[0][1])
            self.assertEqual(kept[0][2],pd.Timestamp('2025-01-05T10:00Z').timestamp())
            self.assertEqual(con.execute('SELECT count(*) FROM raw WHERE raw_id IS NOT NULL AND flight_id IS NULL').fetchone()[0],1)
            self.assertEqual(con.execute('SELECT iobt_nvalues FROM ids WHERE flight_id=200').fetchone()[0],2)

    def test_fixed_label_free_projection(self):
        self.assertEqual(module.DATES,[f'{m}-{d:02d}' for m in ['2025-01','2025-07','2026-01','2026-07'] for d in [5,15,25]])
        self.assertFalse({'BLOCK_TIME_UTC_mvt','TAXITIME_SEC_mvt'} & set(module.COLUMNS))
        self.assertEqual(len(module.COLUMNS),8)


if __name__=='__main__':unittest.main()
