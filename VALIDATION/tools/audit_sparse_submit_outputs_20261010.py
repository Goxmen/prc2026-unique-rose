"""Independent production sequence audit; run only after its cache build.

No selector/model helper is imported. Check every index/point value in bounded
chunks, then reconstruct fixed input-selected cases directly from raw positions.
Only compact query packets, source positions and completed caches are read.
"""
import argparse
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import time

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT/'data/derived/sparse_track_submit_20261010'
QUERY = ['mvt_id','airport','stand','runway','to_ms','clock_ms','hex','day',
         'stand_lat','stand_lon','runway_lat','runway_lon','origin_lat','origin_lon']
RAW = ['takeoff_minus_position_ms','lat_microdeg','lon_microdeg',
       'alt_code','gs_deciknot','preceding_raw_gap_ms']


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def artifact(path):
    return dict(path=str(Path(path).relative_to(ROOT)),sha256=sha(path))


def verify(record):
    path=ROOT/record['path']
    assert path.is_file() and sha(path)==record['sha256'], str(path)
    if 'bytes' in record: assert path.stat().st_size==record['bytes'],str(path)
    return path


def independent_rebuild(query, rows):
    """Raw rows: ts_ms,lat,lon,alt,gs; lexicographic dedup and integer ranks."""
    if pd.isna(query.hex): rows=[]
    ordered=sorted(tuple(int(value) for value in row) for row in rows)
    unique=[]
    for row in ordered:
        if not (int(query.to_ms)-7200000 <= row[0] <= int(query.to_ms)+60000):
            continue
        if not unique or row[0]!=unique[-1][0]: unique.append(row)
    count=len(unique)
    ranks=list(range(count)) if count<=128 else [(i*(count-1))//127 for i in range(128)]
    result=[]
    for i in ranks:
        t,lat,lon,alt,gs=unique[i]
        gap=-1 if i==0 else t-unique[i-1][0]
        result.append((int(query.to_ms)-t,lat,lon,alt,gs,gap))
    points=np.asarray(result,dtype=np.int32).reshape(-1,6)
    return points,dict(raw_count=count,count=min(count,128),
        raw_ground_count=sum(row[3]==-123 for row in unique),
        span_ms=unique[-1][0]-unique[0][0] if unique else 0)


def check_layout(query,index,points,available_days,chunk=262144):
    assert list(query)==QUERY and not query.mvt_id.duplicated().any()
    assert len(query)==len(index) and not index.mvt_id.duplicated().any()
    expected=query.sort_values(['day','mvt_id']).reset_index(drop=True)
    pd.testing.assert_frame_equal(index[QUERY].reset_index(drop=True),expected,check_dtype=False)
    counts=index['count'].to_numpy(np.int64)
    raw=index.raw_count.to_numpy(np.int64)
    starts=index.start.to_numpy(np.int64)
    assert (raw>=0).all()
    np.testing.assert_array_equal(counts,np.minimum(raw,128))
    np.testing.assert_array_equal(starts,np.r_[0,np.cumsum(counts)[:-1]])
    assert int(counts.sum())==len(points)
    assert (index.loc[index.hex.isna(),'count']==0).all()
    assert index.raw_ground_count.between(0,index.raw_count).all()
    assert index.span_ms.between(0,7260000).all()
    assert index.loc[index.raw_count.lt(2),'span_ms'].eq(0).all()
    for field,shift in [('left_source_missing',-7200000),('right_source_missing',60000)]:
        days=pd.to_datetime(index.to_ms+shift,unit='ms',utc=True).dt.strftime('%Y-%m-%d')
        np.testing.assert_array_equal(index[field],~days.isin(available_days))
    used=counts>0
    used_starts,used_raw=starts[used],raw[used]
    if used.any():
        np.testing.assert_array_equal(points[used_starts,5],-np.ones(used.sum(),dtype=np.int32))
        np.testing.assert_array_equal(points[used_starts,0]-points[(starts+counts-1)[used],0],
                                      index.loc[used,'span_ms'])
    for first in range(0,len(points),chunk):
        last=min(len(points),first+chunk)
        block=points[first:last]
        assert block.dtype==np.dtype('<i4') and block.shape[1]==6
        assert ((block[:,0]>=-60000)&(block[:,0]<=7200000)).all()
        assert ((block[:,1]>=-90000000)&(block[:,1]<=90000000)).all()
        assert ((block[:,2]>=-180000000)&(block[:,2]<=180000000)).all()
        positions=np.arange(first,last,dtype=np.int64)
        owner=np.searchsorted(used_starts,positions,side='right')-1
        beginning=positions==used_starts[owner]
        previous=np.r_[points[first-1,0] if first else 0,block[:-1,0]]
        selected_gap=previous.astype(np.int64)-block[:,0]
        assert (selected_gap[~beginning]>0).all()
        assert (block[beginning,5]==-1).all()
        assert ((block[~beginning,5]>0)&(block[~beginning,5]<=selected_gap[~beginning])).all()
        uncompressed=(used_raw[owner]<=128)&~beginning
        np.testing.assert_array_equal(block[uncompressed,5],selected_gap[uncompressed])
    return dict(rows=len(index),stored_points=len(points),point_chunks=(len(points)+chunk-1)//chunk,
                empty=int((counts==0).sum()),single=int((counts==1).sum()),
                subsampled=int((raw>128).sum()))


def choose_sample(index,common_ids):
    pieces=[index.sample(n=min(16,len(index)),random_state=20261010)]
    for _,month in index.groupby(index.day.str[:7]):
        ordered=month.sort_values('to_ms')
        pieces.extend([ordered.head(2),ordered.tail(2)])
    masks=[index['count'].eq(0),index['count'].eq(1),index.raw_count.gt(128),
           index.hex.isna(),index.clock_ms.isna(),index.raw_ground_count.gt(0),
           index.left_source_missing.eq(1),index.right_source_missing.eq(1)]
    for mask in masks:
        frame=index.loc[mask]
        pieces.append(frame.sample(n=min(2,len(frame)),random_state=20261010))
    pieces.append(index.loc[index.mvt_id.isin(common_ids)])
    return pd.concat(pieces).drop_duplicates('mvt_id').sort_values(['day','mvt_id'])


def check_samples(con,index,points,position_paths,common_ids):
    sample=choose_sample(index,common_ids)
    checked=[]
    for day,own in sample.groupby('day',sort=True):
        stamp=date.fromisoformat(day)
        days={str(stamp+timedelta(days=k)) for k in [-1,0,1]}
        sources=[str(path) for key,path in position_paths.items() if key in days]
        keys=own[['airport','hex']].dropna().drop_duplicates()
        con.register('audit_aircraft',keys)
        rows=con.execute('''SELECT p.airport,p.hex,epoch_ms(p.ts)::BIGINT ts_ms,
            p.lat,p.lon,p.alt,p.gs FROM read_parquet(?) p
            JOIN audit_aircraft a USING(airport,hex)
            WHERE epoch_ms(p.ts)>=? AND epoch_ms(p.ts)<=?''',
            [sources,int(own.to_ms.min())-7200000,int(own.to_ms.max())+60000]).df() if sources and len(keys) else pd.DataFrame()
        grouped={} if rows.empty else {key:frame[['ts_ms','lat','lon','alt','gs']].to_numpy()
            for key,frame in rows.groupby(['airport','hex'],observed=True)}
        for query in own.itertuples(index=False):
            raw=grouped.get((query.airport,query.hex),[]) if pd.notna(query.hex) else []
            expected,metadata=independent_rebuild(query,raw)
            actual=points[int(query.start):int(query.start)+int(query.count)]
            np.testing.assert_array_equal(actual,expected)
            for key,value in metadata.items(): assert getattr(query,key)==value,(query.mvt_id,key)
            checked.append(dict(mvt_id=int(query.mvt_id),day=day,**metadata,
                missing_clock=pd.isna(query.clock_ms),missing_identity=pd.isna(query.hex)))
    return checked


def run(out=DEFAULT):
    started=time.monotonic();pa.set_cpu_count(2)
    prepared=read(out/'manifest.json')
    assert sha(ROOT/'VALIDATION/build_sparse_track_submit.py')==prepared['source_sha256']
    for name,digest in prepared['helpers'].items(): assert sha(ROOT/name)==digest,name
    for item in [*prepared['maps'],prepared['training_contract']['manifest'],prepared['training_contract']['design']]: verify(item)
    position_paths={Path(item['path']).stem:verify(item) for item in prepared['positions']}
    for missing in prepared['missing_position_days']:
        assert not (ROOT/'data/external/adsblol_heatmap/positions'/f'{missing}.parquet').exists()
    query_frames={view:pd.read_parquet(verify(item['query'])) for view,item in prepared['views'].items()}
    common=query_frames['old'][['mvt_id','to_ms']].merge(query_frames['final'][['mvt_id','to_ms']],
        on='mvt_id',suffixes=('_old','_final'),validate='one_to_one')
    shifted=common.loc[common.to_ms_old.ne(common.to_ms_final)]
    common_ids=shifted.sample(n=min(8,len(shifted)),random_state=20261010).mvt_id.tolist()
    con=duckdb.connect();con.execute("SET threads=2; SET TimeZone='UTC'; SET memory_limit='1GB'")
    result={}
    for view in ['old','final']:
        target=out/view;manifest=read(target/'full.json');design=read(target/'full_design.json')
        assert all(manifest[key]==value for key,value in design.items())
        assert manifest['settings']['ranking_view']==view and manifest['raw_names']==RAW
        assert set(manifest['outputs'])=={'full_index.parquet','full_points.bin'}
        assert manifest['source_sha256']==prepared['source_sha256'] and manifest['helpers']==prepared['helpers']
        assert manifest['identity_sha256']==prepared['views'][view]['identity']['sha256']
        assert manifest['identity_manifest_sha256']==prepared['views'][view]['identity_manifest']['sha256']
        assert manifest['raw_ranking']==prepared['views'][view]['raw_ranking']
        assert manifest['query']==prepared['views'][view]['query']
        assert manifest['prepared_manifest']==dict(**artifact(out/'manifest.json'),bytes=(out/'manifest.json').stat().st_size)
        assert manifest['design_sha256']==sha(target/'full_design.json')
        assert manifest['point_dtype']=='<i4' and manifest['point_shape']==[manifest['stored_points'],6]
        for name,digest in manifest['outputs'].items(): assert sha(target/name)==digest,name
        index=pd.read_parquet(target/'full_index.parquet')
        p=target/'full_points.bin'; assert p.stat().st_size==manifest['stored_points']*24
        points=np.memmap(p,mode='r',dtype='<i4').reshape(-1,6) if p.stat().st_size else np.empty((0,6),dtype='<i4')
        summary=check_layout(query_frames[view],index,points,set(position_paths))
        assert summary['rows']==manifest['rows']
        samples=check_samples(con,index,points,position_paths,common_ids)
        result[view]=dict(**summary,manifest=artifact(target/'full.json'),
                         sampled_rows=len(samples),sampled_points=sum(row['count'] for row in samples),samples=samples)
        del index,points
    report=dict(passed=True,auditor=artifact(__file__),prepared_manifest=artifact(out/'manifest.json'),
        views=result,common_jitter_sample_ids=common_ids,elapsed_seconds=time.monotonic()-started,
        departure_labels_or_block_read=False,selector_or_model_helper_used=False,
        original_ranking_scope='raw query packets independently audited on desktop; no raw ranking read here')
    destination=out/'output_audit.json';destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='views'},indent=2))
    print(json.dumps({view:{k:v for k,v in row.items() if k!='samples'} for view,row in result.items()},indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=DEFAULT)
    run(parser.parse_args().out)
