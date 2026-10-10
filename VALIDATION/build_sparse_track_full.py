"""Stream the frozen sparse-track representation for all2025 departures.

Per-day index/raw-int32 chunks are committed independently and hash-checked on
resume. Final assembly copies raw bytes to a memory-mappable .bin file; it never
concatenates all raw point arrays in RAM or materializes full dense encodings.
The original four-day pilot is read-only. No departure label/block is selected.
This is a heavy build and must run only in its allocated desktop slot.
"""
import argparse
from datetime import date,timedelta
import gc
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import pyarrow as pa

import sparse_track_sequence as seq

ROOT=Path(__file__).resolve().parents[1]
OUT=seq.OUT
DAYS=[str(date(2025,1,1)+timedelta(days=i)) for i in range(365)]
SETTINGS=dict(original_sequence=seq.SETTINGS,full_days=DAYS,
    outputs='full_index.parquet plus C-order little-endian int32 full_points.bin, six frozen raw columns',
    chunks='one committed index/point chunk per UTC takeoff day; immutable hash-checked resume',
    labels_used=False,full_dense_encoding=False)
DTYPE=np.dtype('<i4')
sha=seq.rotation.sha256


def read(path):
    return json.loads(Path(path).read_text())


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    seq.rotation.write_json(temp,value)
    temp.replace(path)


def path_record(path):
    return dict(path=str(Path(path).relative_to(ROOT)),sha256=sha(path),bytes=Path(path).stat().st_size)


def source_identity():
    identity_manifest=seq.IDENTITY.with_suffix('.json')
    im=read(identity_manifest)
    if sha(seq.IDENTITY)!=im['identity_sha256']:
        raise ValueError('Audited aircraft identity changed')
    if im['source_sha256']!=sha(Path(seq.rotation.__file__)):
        raise ValueError('Audited identity inference source changed')
    helpers={str(Path(m.__file__).relative_to(ROOT)):sha(Path(m.__file__))
             for m in [seq,seq.rotation,seq.sf,seq.sg]}
    dates=sorted({str(date.fromisoformat(d)+timedelta(days=k)) for d in DAYS for k in [-1,0,1]})
    paths=[seq.HEATMAP/'positions'/f'{d}.parquet' for d in dates]
    present=[p for p in paths if p.exists()]
    movements=sorted(ROOT.glob('data/training_2025-*.parquet'))
    if len(movements)!=12:raise ValueError('Expected twelve training movement sources')
    movement_hashes={str(p.relative_to(ROOT)):sha(p) for p in movements}
    identity_movement_hashes={str(Path(r['path']).relative_to(ROOT)):r['sha256']
                              for r in im['movement_sources']}
    if movement_hashes!=identity_movement_hashes:
        raise ValueError('Raw movement sources differ from audited identities')
    return dict(settings=SETTINGS,source_sha256=sha(Path(__file__)),helpers=helpers,
        identity_sha256=im['identity_sha256'],identity_manifest_sha256=sha(identity_manifest),
        stand_map_sha256=sha(seq.sf.COORDS),runway_map_sha256=sha(seq.sf.ENDS),
        movement_sources=movement_hashes,
        positions={str(p.relative_to(ROOT)):sha(p) for p in present},
        missing_position_days=[p.stem for p in paths if not p.exists()],
        versions=dict(numpy=np.__version__,pandas=pd.__version__,pyarrow=pa.__version__))


def guard(out):
    out.mkdir(parents=True,exist_ok=True)
    identity=source_identity()
    path=out/'full_design.json'
    if path.exists():
        if read(path)!=identity:
            raise ValueError('Full sequence sources/settings changed; do not mix cached chunks')
    else:
        if (out/'full_chunks').exists() or any((out/n).exists() for n in ['full_index.parquet','full_points.bin','full.json']):
            raise ValueError('Unrecorded full-cache artifacts require inspection')
        write_json(path,identity)
    return identity


def publish_identical(temp,path):
    """An interrupted deterministic stage may reuse identical committed bytes."""
    if path.exists():
        if sha(path)!=sha(temp):
            raise ValueError(f'Refusing to overwrite a different committed artifact: {path}')
        temp.unlink()
    else:temp.replace(path)


def validate_index(index,stored_points):
    if index.mvt_id.isna().any() or index.mvt_id.duplicated().any():
        raise ValueError('Missing/duplicate full sequence movement IDs')
    count=index['count'].to_numpy(np.int64)
    if (count<0).any() or (count>seq.MAX_POINTS).any() or (index.raw_count.to_numpy()<count).any():
        raise ValueError('Invalid retained point counts')
    starts=np.r_[0,np.cumsum(count)[:-1]] if len(count) else np.empty(0,dtype=np.int64)
    np.testing.assert_array_equal(index.start.to_numpy(np.int64),starts)
    if int(count.sum())!=stored_points:raise ValueError('Index/point length mismatch')


def chunk_paths(out,day):
    folder=out/'full_chunks'
    return folder/f'{day}.parquet',folder/f'{day}.bin',folder/f'{day}.json'


def existing_chunk(out,day,query_ids):
    ip,pp,mp=chunk_paths(out,day)
    if not mp.exists():return None
    record=read(mp)
    if record['design_sha256']!=sha(out/'full_design.json') or record['day']!=day:
        raise ValueError('Daily chunk design changed')
    for key,path in [('index',ip),('points',pp)]:
        if record[key]!=path_record(path):raise ValueError('Daily chunk changed')
    index=pd.read_parquet(ip)
    validate_index(index,record['stored_points'])
    np.testing.assert_array_equal(index.mvt_id.to_numpy(),query_ids)
    if pp.stat().st_size!=record['stored_points']*len(seq.RAW_NAMES)*DTYPE.itemsize:
        raise ValueError('Daily binary size does not match point shape')
    return record


def build_chunk(con,out,day,query):
    current=existing_chunk(out,day,query.mvt_id.to_numpy())
    if current is not None:return current
    stamp=date.fromisoformat(day)
    paths=[seq.HEATMAP/'positions'/f'{stamp+timedelta(days=k)}.parquet' for k in [-1,0,1]]
    present=[p for p in paths if p.exists()]
    positions=con.execute('''SELECT airport,hex,epoch_ms(ts)::BIGINT ts_ms,lat,lon,alt,gs
        FROM read_parquet(?) WHERE ts>=?::TIMESTAMPTZ-INTERVAL 2 HOUR
        AND ts<?::TIMESTAMPTZ+INTERVAL 1 DAY+INTERVAL 60 SECOND''',
        [[str(p) for p in present],day,day]).df() if present else pd.DataFrame()
    index,points=seq.select_sequences(query,positions,{p.stem for p in present})
    validate_index(index,len(points))
    np.testing.assert_array_equal(index.mvt_id.to_numpy(),query.mvt_id.to_numpy())
    if points.ndim!=2 or points.shape[1]!=len(seq.RAW_NAMES) or points.dtype!=np.dtype('int32'):
        raise ValueError('Frozen raw sequence format changed')
    ip,pp,mp=chunk_paths(out,day);ip.parent.mkdir(parents=True,exist_ok=True)
    itmp=ip.with_suffix('.parquet.tmp');ptmp=pp.with_suffix('.bin.tmp')
    index.to_parquet(itmp,index=False)
    np.asarray(points,dtype=DTYPE,order='C').tofile(ptmp)
    publish_identical(itmp,ip);publish_identical(ptmp,pp)
    record=dict(day=day,design_sha256=sha(out/'full_design.json'),rows=len(index),
        stored_points=len(points),raw_points=int(index.raw_count.sum()),
        index=path_record(ip),points=path_record(pp),source_days=[p.stem for p in present],
        identified=int(index.hex.notna().sum()),usable_2plus=int(index['count'].ge(2).sum()),
        missing_original_clock_usable=int((index['count'].ge(2)&index.clock_ms.isna()).sum()),
        left_source_missing=int(index.left_source_missing.sum()),right_source_missing=int(index.right_source_missing.sum()))
    write_json(mp,record)
    return record


def assemble(out,query_ids,records,identity,elapsed):
    ip,pp=out/'full_index.parquet',out/'full_points.bin'
    ptmp=pp.with_suffix('.bin.tmp');itmp=ip.with_suffix('.parquet.tmp')
    indices=[];cursor=0
    with ptmp.open('wb') as target:
        for day,record in records.items():
            cip,cpp,_=chunk_paths(out,day)
            if path_record(cip)!=record['index'] or path_record(cpp)!=record['points']:
                raise ValueError('A committed chunk changed during assembly')
            frame=pd.read_parquet(cip)
            validate_index(frame,record['stored_points'])
            frame['start']=frame.start.to_numpy(np.int64)+cursor
            indices.append(frame)
            with cpp.open('rb') as source:shutil.copyfileobj(source,target,length=8*1024*1024)
            cursor+=record['stored_points']
    index=pd.concat(indices,ignore_index=True)
    validate_index(index,cursor)
    np.testing.assert_array_equal(np.sort(index.mvt_id.to_numpy()),np.sort(query_ids))
    if ptmp.stat().st_size!=cursor*len(seq.RAW_NAMES)*DTYPE.itemsize:
        raise ValueError('Assembled raw binary has wrong length')
    index.to_parquet(itmp,index=False)
    publish_identical(itmp,ip);publish_identical(ptmp,pp)
    summary={day:{k:v for k,v in r.items() if k not in ['index','points','design_sha256']} for day,r in records.items()}
    manifest=dict(**identity,design_sha256=sha(out/'full_design.json'),rows=len(index),stored_points=cursor,
        raw_names=seq.RAW_NAMES,point_dtype=DTYPE.str,point_shape=[cursor,len(seq.RAW_NAMES)],
        raw_array_bytes=pp.stat().st_size,by_day=summary,
        chunks={day:sha(chunk_paths(out,day)[2]) for day in records},
        outputs={ip.name:sha(ip),pp.name:sha(pp)},elapsed_sec=time.monotonic()-elapsed)
    write_json(out/'full.json',manifest)
    return manifest


def build(out=OUT):
    started=time.monotonic();pa.set_cpu_count(2)
    identity=guard(out)
    if (out/'full.json').exists():
        final=read(out/'full.json')
        if any(final[k]!=v for k,v in identity.items()):raise ValueError('Completed full cache identity changed')
        for name,digest in final['outputs'].items():
            if sha(out/name)!=digest:raise ValueError('Completed full cache bytes changed')
        print('Full cache already complete and hash checked',flush=True)
        return final
    con=seq.rotation.connection()
    try:
        query=seq.add_geometry(seq.inputs(con,DAYS))
        if query.mvt_id.duplicated().any() or set(query.day)!=set(DAYS):
            raise ValueError('Full-year departure day/ID population changed')
        raw_n=con.execute("SELECT count(*) FROM read_parquet(?) WHERE PHASE_mvt='DEP'",
                          [str(ROOT/'data/training_2025-*.parquet')]).fetchone()[0]
        if len(query)!=raw_n:raise ValueError('Date-filtered queries did not cover every supplied departure')
        records={}
        for i,(day,own) in enumerate(query.groupby('day',sort=True)):
            records[day]=build_chunk(con,out,day,own)
            if (i+1)%30==0:print(f'{i+1}/365 days committed in {time.monotonic()-started:.1f}s',flush=True)
        ids=query.mvt_id.to_numpy().copy()
        del query;gc.collect()
        result=assemble(out,ids,records,identity,started)
        print(json.dumps({k:result[k] for k in ['rows','stored_points','raw_array_bytes','elapsed_sec']},indent=2),flush=True)
        return result
    finally:con.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=OUT)
    a=p.parse_args()
    build(a.out)
