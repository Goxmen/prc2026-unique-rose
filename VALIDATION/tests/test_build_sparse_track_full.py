"""Small resumability, byte layout and assembly checks; no external-cache scan."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import build_sparse_track_full as b
import sparse_track_sequence as s


def query(mid,day,hex=123):
    t=int(pd.Timestamp(day+'T00:01:00Z').timestamp()*1000)
    return pd.DataFrame([dict(mvt_id=mid,airport='TEST',stand='A',runway='01',
        to_ms=t,clock_ms=np.nan,hex=hex,day=day,stand_lat=np.nan,stand_lon=np.nan,
        runway_lat=50.,runway_lon=4.,origin_lat=50.,origin_lon=4.)])


class TestFullSequence(unittest.TestCase):
    def test_chunk_resume_and_stream_assembly(self):
        with tempfile.TemporaryDirectory(dir=b.ROOT/'tmp',prefix='sequence_test_') as root:
            root=Path(root);out=root/'out';out.mkdir();heat=root/'heat';(heat/'positions').mkdir(parents=True)
            b.write_json(out/'full_design.json',{'synthetic':True})
            pilot=out/'pilot_points.npy';pilot.write_bytes(b'untouched pilot')
            day='2025-01-10';q=query(1,day)
            t=q.to_ms.iloc[0]
            pd.DataFrame(dict(airport=['TEST']*2,hex=[123]*2,
                ts=pd.to_datetime([t-1000,t],unit='ms',utc=True),lat=[50000000]*2,
                lon=[4000000]*2,alt=[-123]*2,gs=[50]*2)).to_parquet(heat/'positions'/f'{day}.parquet')
            con=b.seq.rotation.connection()
            try:
                with patch.object(s,'HEATMAP',heat):
                    first=b.build_chunk(con,out,day,q)
                    mtime=b.chunk_paths(out,day)[1].stat().st_mtime_ns
                    second=b.build_chunk(con,out,day,q)
                    self.assertEqual(first,second)
                    self.assertEqual(mtime,b.chunk_paths(out,day)[1].stat().st_mtime_ns)
                    day2='2025-01-11';second=b.build_chunk(con,out,day2,query(2,day2,hex=np.nan))
                final=b.assemble(out,np.array([2,1]),{day:first,day2:second},{'synthetic':True},0.)
                self.assertEqual(final['rows'],2);self.assertEqual(final['stored_points'],2)
                raw=np.fromfile(out/'full_points.bin',dtype='<i4').reshape(-1,6)
                np.testing.assert_array_equal(raw[:,0],[1000,0])
                np.testing.assert_array_equal(raw[:,5],[-1,1000])
                ix=pd.read_parquet(out/'full_index.parquet')
                np.testing.assert_array_equal(ix.start,[0,2]);np.testing.assert_array_equal(ix['count'],[2,0])
                self.assertEqual(pilot.read_bytes(),b'untouched pilot')
                with b.chunk_paths(out,day)[1].open('r+b') as f:f.write(b'xxxx')
                with self.assertRaises(ValueError):b.existing_chunk(out,day,np.array([1]))
            finally:con.close()

    def test_identity_changes_and_unknown_artifacts_rejected(self):
        with tempfile.TemporaryDirectory(dir=b.ROOT/'tmp',prefix='sequence_guard_') as root:
            out=Path(root)/'out'
            with patch.object(b,'source_identity',return_value={'a':1}):b.guard(out)
            with patch.object(b,'source_identity',return_value={'a':2}):
                with self.assertRaises(ValueError):b.guard(out)
            another=Path(root)/'other';another.mkdir();(another/'full_points.bin').touch()
            with patch.object(b,'source_identity',return_value={'a':1}):
                with self.assertRaises(ValueError):b.guard(another)

    def test_index_alignment_and_bin_boundaries(self):
        good=pd.DataFrame(dict(mvt_id=[1,2,3],start=[0,2,2],count=[2,0,1],raw_count=[2,0,1]))
        b.validate_index(good,3)
        for col,value in [('start',[0,3,3]),('mvt_id',[1,1,3]),('count',[129,0,1])]:
            bad=good.copy();bad[col]=value
            with self.assertRaises((ValueError,AssertionError)):b.validate_index(bad,3)


if __name__=='__main__':unittest.main()
