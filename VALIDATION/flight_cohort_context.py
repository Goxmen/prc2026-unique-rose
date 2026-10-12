"""Paired input-only flight-context bank and exact bounded neighbor selection.

No labels, predicted taxi values or neighboring raw point tensors are loaded.
The same movement can have different old/final timestamps: views never share a
bank. Calendar membership is an explicit set, not the range between endpoints.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'data/derived/flight_cohort_context_20261012'
K, WINDOW_MS, ALIAS_MS = 16, 1800000, 120000
FOLDS = {'2025-01':[f'2025-{m:02d}' for m in range(3,12)],
         '2025-07':['2025-01','2025-02','2025-03','2025-09','2025-10','2025-11','2025-12']}
META = ['mvt_id','to_ms','airport','stand','runway','hex']
VALUES = ['actual_clock','op_ground','op_stand','ab_stand',
          'actual_valid','op_ground_valid','op_stand_valid','ab_stand_valid',
          'actual_present','stand_x','stand_y','stand_known',
          'log_raw_count','ground_fraction','ground_known']
SOURCE_COLUMNS = ['mvt_id','airport','stand','runway','to_ms','clock_ms','hex',
    'stand_lat','stand_lon','origin_lat','origin_lon','raw_count','raw_ground_count']
SETTINGS = dict(k=K,window_ms=WINDOW_MS,alias_ms=ALIAS_MS,
    order='absolute takeoff distance, takeoff timestamp, movement ID',
    aliases='same movement ID; known same aircraft within120seconds; exact known takeoff/AOBT/stand/runway tuple',
    source_rows='all supplied departure inputs, regardless ordinary-model training routing',
    values=VALUES,meta=META,clock_range=[0,10800],
    physical_gates='OPDI matched1; ADSB matched1 and stand_n>0; finite clock in[0,10800]',
    geometry='R6371000 equirectangular airport-origin meters; missing explicit',
    count='log1p(min(raw_count,4096))/log4097',
    contexts='train only exact allowed months; held allowed train plus own month',
    views='independent train/old/final source banks and neighbor construction',
    no_departure_labels=True,no_peer_predictions=True,no_arrival_labels=True,
    no_sensor_quality_neighbor_selection=True,no_raw_neighbor_points=True)
SOURCES = {
    'train':['data/derived/sparse_track_sequence_20261010/full_index.parquet',
             'data/external/opdi/opdi_features.parquet',
             'data/external/adsblol_heatmap/adsb_features.parquet',
             'data/external/adsblol_heatmap/adsb_stand_features.parquet'],
    'old':['data/derived/sparse_track_submit_20261010/old/full_index.parquet',
           'data/derived/final_tables/old/opdi.parquet',
           'data/derived/final_tables/old/adsb.parquet',
           'data/derived/final_tables/old/adsb_stand.parquet'],
    'final':['data/derived/sparse_track_submit_20261010/final/full_index.parquet',
             'data/derived/final_tables/final/opdi.parquet',
             'data/derived/final_tables/final/adsb.parquet',
             'data/derived/final_tables/final/adsb_stand.parquet']}


def art(path):
    path=Path(path);h=hashlib.sha256()
    with path.open('rb')as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return dict(path=str(path.relative_to(ROOT)),sha256=h.hexdigest(),bytes=path.stat().st_size)


def verify(record):
    path=ROOT/record['path']
    if art(path)!=record:raise ValueError('Changed cohort artifact: '+str(path))
    return path


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def codes(series):
    text=series.astype('string').fillna('')
    names=sorted(set(text)-{''});lookup={name:i+1 for i,name in enumerate(names)}
    return text.map(lookup).fillna(0).to_numpy(np.int64),names


def arrays(frame):
    f=frame.sort_values('mvt_id').reset_index(drop=True)
    assert f.mvt_id.is_unique and len(f)>0
    meta=np.empty((len(f),len(META)),np.int64);vocab={}
    meta[:,0]=f.mvt_id;meta[:,1]=f.to_ms
    for j,name in enumerate(['airport','stand','runway'],2):meta[:,j],vocab[name]=codes(f[name])
    assert (meta[:,2]>0).all()
    meta[:,5]=f.hex.fillna(-1).to_numpy(np.int64)
    rawclock=f.clock_ms.to_numpy(float,na_value=np.nan)
    clock=(meta[:,1]-rawclock)/1000
    value=np.column_stack([clock,f.op_ground_sec,f.op_own_stand_sec,f.ab_stand_exit_sec]).astype(float)
    valid=np.isfinite(value)&(value>=0)&(value<=10800)
    valid[:,1:3]&=f.op_matched.eq(1).to_numpy()[:,None]
    valid[:,3]&=(f.ab_matched.eq(1)&f.ab_stand_n.gt(0)).to_numpy()
    value=np.where(valid,value,0)
    geometry=f[['stand_lat','stand_lon','origin_lat','origin_lon']].to_numpy(float)
    known=np.isfinite(geometry).all(1)
    x=6371000*np.deg2rad(geometry[:,1]-geometry[:,3])*np.cos(np.deg2rad(geometry[:,2]))
    y=6371000*np.deg2rad(geometry[:,0]-geometry[:,2])
    n=f.raw_count.to_numpy(float);ground=f.raw_ground_count.to_numpy(float)
    assert np.isfinite(n).all() and (n>=0).all() and ((ground>=0)&(ground<=n)).all()
    values=np.column_stack([value,valid,np.isfinite(rawclock),np.where(known,x,0),np.where(known,y,0),
        known,np.log1p(np.minimum(n,4096))/np.log(4097),ground/np.maximum(n,1),n>0]).astype('float32')
    assert values.shape==(len(f),len(VALUES)) and np.isfinite(values).all()
    return meta,values,rawclock,vocab


def valid_candidates(meta,clock,q,donors):
    q=np.asarray(q);d=np.asarray(donors)
    qid,qt,_,qs,qr,qh=(meta[q,j,None] for j in range(6))
    dt=meta[d,1]-qt
    keep=(meta[d,0]!=qid)&(abs(dt)<=WINDOW_MS)&(meta[d,2]==meta[q,2,None])
    alias=(qh>=0)&(meta[d,5]==qh)&(abs(dt)<=ALIAS_MS)
    signature=(qs>0)&(qr>0)&np.isfinite(clock[q,None])
    alias|=signature&(meta[d,3]==qs)&(meta[d,4]==qr)&(dt==0)&(clock[d]==clock[q,None])
    return keep&~alias


def select(meta,clock,queries,references,chunk=8192):
    """Vectorized nearby candidates, with exact full-window fallback if needed.

    Thirty-two positions on each temporal side suffice unless rejected aliases
    exhaust that side. The first unseen position proves completeness; any
    possible tie at the selected cutoff invokes the exact full-window fallback.
    """
    queries=np.asarray(queries,bool);references=np.asarray(references,bool)
    assert queries.shape==references.shape==(len(meta),)
    output=np.full((len(meta),K),-1,np.int32);fallback=0
    for airport in np.unique(meta[queries,2]):
        donor=np.flatnonzero(references&(meta[:,2]==airport))
        donor=donor[np.lexsort((meta[donor,0],meta[donor,1]))]
        if not len(donor):continue
        times=meta[donor,1];qq=np.flatnonzero(queries&(meta[:,2]==airport))
        for start in range(0,len(qq),chunk):
            q=qq[start:start+chunk];qt=meta[q,1]
            pos=np.searchsorted(times,qt,side='left')
            raw=pos[:,None]+np.arange(-32,32)
            inside=(raw>=0)&(raw<len(donor));d=donor[np.clip(raw,0,len(donor)-1)]
            valid=inside&valid_candidates(meta,clock,q,d)
            distance=np.where(valid,abs(meta[d,1]-qt[:,None]),np.iinfo(np.int64).max)
            order=np.lexsort((meta[d,0],meta[d,1],distance),axis=1)[:,:K]
            chosen=np.take_along_axis(d,order,1);good=np.take_along_axis(valid,order,1)
            output[q]=np.where(good,chosen,-1)
            worst=np.take_along_axis(distance,order,1)[:,-1]
            must=np.zeros(len(q),bool)
            for nextpos in [pos-33,pos+32]:
                ok=(nextpos>=0)&(nextpos<len(donor));nexttime=times[np.clip(nextpos,0,len(donor)-1)]
                dd=abs(nexttime-qt);must|=ok&(dd<=WINDOW_MS)&(dd<=worst)
            for local in np.flatnonzero(must):
                qi=q[local];lo=np.searchsorted(times,qt[local]-WINDOW_MS,'left')
                hi=np.searchsorted(times,qt[local]+WINDOW_MS,'right');d=donor[lo:hi]
                good=valid_candidates(meta,clock,np.array([qi]),d[None,:])[0];d=d[good]
                rank=np.lexsort((meta[d,0],meta[d,1],abs(meta[d,1]-qt[local])))[:K]
                output[qi]=-1;output[qi,:len(rank)]=d[rank];fallback+=1
    return output,dict(queries=int(queries.sum()),references=int(references.sum()),exact_fallback_queries=fallback)


def scoped(meta,clock,month):
    months=pd.to_datetime(meta[:,1],unit='ms',utc=True).strftime('%Y-%m').to_numpy()
    allowed=np.isin(months,FOLDS[month]);held=months==month
    training,a=select(meta,clock,allowed,allowed)
    heldout,b=select(meta,clock,held,allowed|held)
    training[held]=heldout[held]
    return training,dict(training=a,held=b,reference_months=FOLDS[month],held_reference_months=FOLDS[month]+[month])


def source_frame(view):
    paths=[ROOT/p for p in SOURCES[view]]
    with duckdb.connect(config={'threads':1,'memory_limit':'2GB'})as con:
        con.execute("SET TimeZone='UTC'")
        columns=','.join('q.'+c for c in SOURCE_COLUMNS)
        frame=con.execute(f'''SELECT {columns},op.op_matched,op.op_ground_sec,
          op.op_own_stand_sec,ab.ab_matched,st.ab_stand_n,st.ab_stand_exit_sec
          FROM read_parquet(?) q LEFT JOIN read_parquet(?) op USING(mvt_id)
          LEFT JOIN read_parquet(?) ab USING(mvt_id)
          LEFT JOIN read_parquet(?) st USING(mvt_id) ORDER BY q.mvt_id''',list(map(str,paths))).df()
    assert frame.mvt_id.is_unique
    return frame


def build(view='train'):
    if view!='train':
        raise ValueError('Only the CV train bank is authorized; production needs an explicit train+own-month scoped builder')
    dest=OUT/view;manifest=dest/'manifest.json'
    if manifest.exists():guard(view);return
    if dest.exists() and any(dest.iterdir()):raise ValueError('Unrecorded partial cache; inspect before rebuilding')
    started=time.monotonic();dest.mkdir(parents=True,exist_ok=True)
    f=source_frame(view);meta,values,clock,vocab=arrays(f);del f
    for name,array in [('meta',meta),('values',values),('clock',clock)]:np.save(dest/f'{name}.npy',array)
    scope={}
    if view=='train':
        for month in FOLDS:
            index,scope[month]=scoped(meta,clock,month);np.save(dest/f'neighbors_{month}.npy',index);del index
    else:
        allrows=np.ones(len(meta),bool);index,scope[view]=select(meta,clock,allrows,allrows)
        np.save(dest/'neighbors.npy',index)
    identity=dict(settings=SETTINGS,source=art(__file__),ranking_view=view,rows=len(meta),vocabulary=vocab,
        sources=[art(ROOT/p) for p in SOURCES[view]],outputs=[art(p) for p in sorted(dest.glob('*.npy'))],
        scope=scope,seconds=time.monotonic()-started,no_labels_loaded=True)
    write(manifest,identity);print(dict(view=view,rows=len(meta),seconds=identity['seconds']),flush=True)


def guard(view='train',full=True):
    manifest=OUT/view/'manifest.json';identity=json.loads(manifest.read_text())
    assert identity['settings']==SETTINGS and identity['source']==art(__file__)
    assert identity['ranking_view']==view
    expected_sources=[str(Path(p)) for p in SOURCES[view]]
    assert [r['path'] for r in identity['sources']]==expected_sources
    expected_outputs=['meta.npy','values.npy','clock.npy']
    expected_outputs+=([f'neighbors_{m}.npy' for m in FOLDS] if view=='train' else ['neighbors.npy'])
    assert sorted(r['path'] for r in identity['outputs'])==sorted(str((OUT/view/name).relative_to(ROOT)) for name in expected_outputs)
    if view=='train':
        for month,months in FOLDS.items():
            assert identity['scope'][month]['reference_months']==months
            assert identity['scope'][month]['held_reference_months']==months+[month]
    if full:
        for rec in identity['sources']+identity['outputs']:verify(rec)
    return identity


def checks():
    rng=np.random.default_rng(12);n=230
    meta=np.column_stack([np.arange(n)+100,np.sort(rng.integers(-2000000,2000000,n)),
        rng.integers(1,3,n),rng.integers(0,4,n),rng.integers(0,3,n),np.full(n,-1)]).astype(np.int64)
    clock=meta[:,1].astype(float)-600000;clock[::7]=np.nan
    # Force enough aliases to exercise adaptive exact fallback.
    meta[:90,2]=1;meta[:90,1]=0;meta[:90,5]=7
    meta[91,1]=0;meta[91,2]=1
    queries=np.ones(n,bool);refs=np.ones(n,bool)
    actual,receipt=select(meta,clock,queries,refs,chunk=13)
    assert receipt['exact_fallback_queries']>0
    for i in range(n):
        donors=np.flatnonzero(refs)
        good=valid_candidates(meta,clock,np.array([i]),donors[None,:])[0];d=donors[good]
        order=np.lexsort((meta[d,0],meta[d,1],abs(meta[d,1]-meta[i,1])))[:K]
        expected=np.full(K,-1);expected[:len(order)]=d[order]
        np.testing.assert_array_equal(actual[i],expected)
    permutation=rng.permutation(n);inverse=np.argsort(permutation)
    changed,_=select(meta[permutation],clock[permutation],queries,refs,chunk=19)
    mapped=np.where(changed>=0,permutation[np.maximum(changed,0)],-1)
    np.testing.assert_array_equal(mapped[inverse],actual)
    dates=['2025-03-31T23:55','2025-04-01T00:00','2025-06-30T23:55',
           '2025-07-01T00:00','2025-09-01T00:00','2025-08-31T23:55']
    small=np.column_stack([np.arange(6)+1,pd.to_datetime(dates,utc=True).as_unit('ms').asi8,
        np.ones(6),np.arange(6)+1,np.ones(6),np.full(6,-1)]).astype(np.int64)
    index,_=scoped(small,np.full(6,np.nan),'2025-07')
    assert (index==-1).all(),'Disallowed Apr/Jun/Aug crossed a calendar boundary'
    # The same IDs in separately jittered views must have different distance order.
    a=small[:3].copy();a[:,1]=[0,1000,-2000];a[:,3]=np.arange(3)+1
    b=a.copy();b[1,1]=3000
    ia,_=select(a,np.full(3,np.nan),np.ones(3,bool),np.ones(3,bool))
    ib,_=select(b,np.full(3,np.nan),np.ones(3,bool),np.ones(3,bool))
    assert ia[0,:2].tolist()==[1,2] and ib[0,:2].tolist()==[2,1]
    return dict(vectorized_exact_reference=True,alias_fallback=True,query_exclusion=True,
        missing_identity_not_alias=True,source_permutation=True,calendar_gap_isolation=True,
        separate_jittered_views=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['check','build'])
    p.add_argument('--view',choices=list(SOURCES),default='train');a=p.parse_args()
    if a.stage=='check':print(checks())
    else:build(a.view)
