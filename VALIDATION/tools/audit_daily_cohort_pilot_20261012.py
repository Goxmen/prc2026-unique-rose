"""Separate input-only review of proposed full-day peer information."""
from pathlib import Path
import hashlib
import json
import sys

import duckdb
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'VALIDATION'))
import flight_cohort_context as old
import public_delay_features as public

OUT=ROOT/'VALIDATION/reports/cohort_learning_20261012'


def main():
    result=json.loads((OUT/'pilot_result.json').read_text())
    identity=json.loads((OUT/'pilot_identity.json').read_text())
    for a in identity['sources']+[result['cohort_artifact'],result['identity']]:
        assert hashlib.sha256((ROOT/a['path']).read_bytes()).hexdigest()==a['sha256']
    meta=np.load(old.OUT/'train/meta.npy',mmap_mode='r')
    values=np.load(old.OUT/'train/values.npy',mmap_mode='r')
    raw=pd.read_parquet(public.OUT/'inputs2025.parquet',columns=['mvt_id','airport','day','fltrule'])
    raw=raw.set_index('mvt_id').loc[meta[:,0]].reset_index()
    assert len(raw)==raw.mvt_id.nunique()==2085047
    np.testing.assert_array_equal(raw.day,pd.to_datetime(meta[:,1],unit='ms',utc=True).strftime('%Y-%m-%d'))
    donor=~raw.fltrule.eq('V').to_numpy()
    paired=donor&(values[:,4]>0)&np.any(values[:,5:8]>0,axis=1)
    f=raw.assign(time=meta[:,1],donor=donor,paired=paired,own_absent=~np.any(values[:,5:8]>0,axis=1))
    with duckdb.connect(config={'threads':1,'memory_limit':'512MB'}) as con:
        con.register('f',f)
        g=pd.read_parquet(OUT/'pilot_cohorts.parquet');con.register('g',g)
        assert con.execute('''with a as (select airport,day,count(*) n,sum(donor::int) d,sum(paired::int) p
            from f group by all) select count(*) from a full join g using(airport,day)
            where a.n is distinct from g.rows or a.d is distinct from g.donor_rows or a.p is distinct from g.paired_rows''').fetchone()[0]==0
        # Different fixed sample from producer, independently enumerated SQL.
        query=f.iloc[np.random.default_rng(189).choice(len(f),512,replace=False)]
        con.register('queries',query)
        direct=con.execute('''select q.mvt_id,count(d.mvt_id) as far from queries q left join f d
            on q.airport=d.airport and q.day=d.day and d.paired and abs(q.time-d.time)>1800000
            group by q.mvt_id''').df().set_index('mvt_id')
        for _,q in query.iterrows():
            mask=paired&raw.airport.eq(q.airport).to_numpy()&raw.day.eq(q.day).to_numpy()
            answer=np.sum(np.abs(meta[mask,1]-q.time)>1800000)
            assert answer==direct.loc[q.mvt_id,'far']
    for fold,scope in result['scope'].items():
        allowed=old.FOLDS[fold];assert scope['allowed']==allowed
        train=raw.day.str[:7].isin(allowed);held=raw.day.str[:7].eq(fold)
        assert not (train&held).any()
        assert train.sum()==scope['train_source_rows'] and held.sum()==scope['held_rows']
    # Explicitly instantiate separated temporal donors rather than merely
    # asserting that a swap lies outside the old context window.
    flight=pd.DataFrame(dict(mvt_id=[11,12,13,14],airport='LIRF',day='2025-01-10',
        gap=[6000.,4000.,8000.,1800.],clock=[1000.,1000.,1000.,900.],fltrule='I'))
    times=np.array([12.,6.,18.,12.1])*3600000
    physical_a=np.array([1000.,300.,1700.,900.]);physical_b=physical_a[[0,2,1,3]]
    source=pd.DataFrame(dict(airport=['LIRF'],day=['2025-01-10'],n_public=[4.],total=[15000.]))
    first=public.build(flight.assign(physical=physical_a),source)
    second=public.build(flight.assign(physical=physical_b),source)
    pd.testing.assert_frame_equal(first,second,check_exact=True)
    near=(abs(times-times[0])<=old.WINDOW_MS)&(np.arange(4)!=0)
    np.testing.assert_array_equal(physical_a[near],physical_b[near])
    np.testing.assert_array_equal(np.sort(physical_a),np.sort(physical_b))
    joint=lambda x:float(np.mean(flight.gap.to_numpy()[1:]*(x-flight.clock.to_numpy())[1:]))
    assert joint(physical_a)!=joint(physical_b)
    audit=dict(status='pass_input_information_only',rows=len(raw),cohorts=len(g),
        separately_sampled_temporal_queries=512,all_source_hashes=True,all_UTC_dates_and_counts=True,
        disjoint_exact_allowed_months=True,explicit_fixed_old_summary_collision=True,
        no_departure_labels_or_predictions=True,no_model_fit=True,
        caveat='UTC takeoff grouping follows existing public inputs; public cohort/calendar identity is not established',
        pending='Review new bank/model/trainer, gradient/RNG parity, CUDA capacity and frozen transport before fitting')
    (OUT/'independent_pilot_review.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit))


if __name__=='__main__':main()
