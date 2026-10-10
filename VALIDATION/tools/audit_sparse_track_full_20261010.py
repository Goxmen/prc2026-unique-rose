"""Check streamed full cache against all preserved pilot points and bounds."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'VALIDATION'))
import sparse_track_sequence as seq
import build_sparse_track_full as full
from aircraft_rotation import sha256,write_json


def run():
    manifest=json.loads((seq.OUT/'full.json').read_text())
    assert manifest['source_sha256']==sha256(Path(full.__file__))
    for name,digest in manifest['outputs'].items(): assert sha256(seq.OUT/name)==digest
    pilot=pd.read_parquet(seq.OUT/'pilot_index.parquet')
    pilot_points=np.load(seq.OUT/'pilot_points.npy',mmap_mode='r',allow_pickle=False)
    full_index=pd.read_parquet(seq.OUT/'full_index.parquet')
    full_points=np.memmap(seq.OUT/'full_points.bin',mode='r',dtype='<i4').reshape(-1,6)
    full.validate_index(full_index,len(full_points))
    checked=0
    for day in seq.PILOT_DAYS:
        expected=pilot.loc[pilot.day.eq(day)].copy()
        actual=full_index.loc[full_index.day.eq(day)].copy()
        e0,a0=int(expected.start.iloc[0]),int(actual.start.iloc[0])
        n=int(expected['count'].sum())
        np.testing.assert_array_equal(pilot_points[e0:e0+n],full_points[a0:a0+n])
        expected['start']-=e0;actual['start']-=a0
        # Global year read can widen nullable dtypes; value and row identities
        # still must agree exactly with the original fixed-date pilot.
        pd.testing.assert_frame_equal(expected.reset_index(drop=True),actual.reset_index(drop=True),check_dtype=False)
        checked+=n
    starts=full_index.start.to_numpy(np.int64)
    for first in range(0,len(full_points),1000000):
        last=min(first+1000000,len(full_points))
        data=full_points[first:last]
        assert (data[:,0]>=-60000).all() and (data[:,0]<=7200000).all()
        assert ((data[:,5]==-1)|((data[:,5]>0)&(data[:,5]<=7260000))).all()
        differences=np.diff(full_points[max(0,first-1):last,0].astype(np.int64))
        origin=max(1,first)
        boundaries=starts[(starts>=origin)&(starts<last)]-origin
        valid=np.ones(len(differences),dtype=bool);valid[boundaries]=False
        assert (differences[valid]<0).all(), 'Within-sequence chronology changed'
    used=full_index['count'].gt(0).to_numpy()
    assert (full_points[starts[used],5]==-1).all()
    result=dict(status='passed',rows=len(full_index),stored_points=len(full_points),
        pilot_rows=len(pilot),exact_pilot_points=checked,exact_pilot_raw_values=checked*6,
        all_sequence_time_bounds_and_order=True,full_manifest_sha256=sha256(seq.OUT/'full.json'),
        audit_source_sha256=sha256(Path(__file__)))
    write_json(ROOT/'tmp/research_20261010/model_diversity/sparse_track_full_audit.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':run()
