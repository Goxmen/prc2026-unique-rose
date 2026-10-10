"""Independent auditor catches corrupt bounds/metadata and raw selection."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import audit_sparse_submit_outputs_20261010 as audit


class AuditSequenceOutputTests(unittest.TestCase):
    def test_raw_reconstruction_dedup_window_and_subsampling(self):
        q=SimpleNamespace(to_ms=20000000,hex=123)
        times=[q.to_ms-200000+i*i*7 for i in range(150)]
        raw=[(t,50000000+i,4000000,-123 if i%2 else 1,50) for i,t in enumerate(times)]
        raw.extend([(times[0],49999999,4000000,1,51),
                    (q.to_ms-7200001,1,1,1,1),(q.to_ms+60001,1,1,1,1)])
        points,meta=audit.independent_rebuild(q,list(reversed(raw)))
        self.assertEqual(meta,dict(raw_count=150,count=128,raw_ground_count=75,span_ms=times[-1]-times[0]))
        self.assertEqual(points[0,1],49999999)
        self.assertEqual(points[0,5],-1)
        ranks=[i*149//127 for i in range(128)]
        np.testing.assert_array_equal(points[:,0],[q.to_ms-times[i] for i in ranks])
        np.testing.assert_array_equal(points[1:,5],[times[i]-times[i-1] for i in ranks[1:]])
        empty,summary=audit.independent_rebuild(SimpleNamespace(to_ms=q.to_ms,hex=np.nan),raw)
        self.assertEqual(empty.shape,(0,6));self.assertEqual(summary['raw_count'],0)

    def test_whole_index_and_cross_chunk_corruption(self):
        stamp=int(pd.Timestamp('2026-01-01T00:01:00Z').timestamp()*1000)
        query=pd.DataFrame([dict(mvt_id=i,airport='TEST',stand='A',runway='01',to_ms=stamp,
            clock_ms=np.nan,hex=123 if i!=2 else np.nan,day='2026-01-01',stand_lat=np.nan,
            stand_lon=np.nan,runway_lat=50.,runway_lon=4.,origin_lat=50.,origin_lon=4.) for i in [1,2,3]])
        index=query.assign(start=[0,2,2],count=[2,0,1],raw_count=[2,0,1],
            raw_ground_count=[2,0,1],span_ms=[1000,0,0],left_source_missing=1,right_source_missing=0)
        points=np.array([[1000,50000000,4000000,-123,50,-1],
                         [0,50000000,4000000,-123,50,1000],
                         [-60000,50000000,4000000,-123,50,-1]],dtype='<i4')
        report=audit.check_layout(query,index,points,{'2026-01-01'},chunk=1)
        self.assertEqual(report['stored_points'],3)
        for row,col,value in [(1,5,999),(1,0,1001),(2,0,-60001),(0,1,90000001),(2,5,1)]:
            corrupt=points.copy();corrupt[row,col]=value
            with self.assertRaises(AssertionError):
                audit.check_layout(query,index,corrupt,{'2026-01-01'},chunk=1)
        wrong=index.copy();wrong.loc[2,'to_ms']+=12000
        with self.assertRaises(AssertionError):audit.check_layout(query,wrong,points,{'2026-01-01'})


if __name__=='__main__':unittest.main()
