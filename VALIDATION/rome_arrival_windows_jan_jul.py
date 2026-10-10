"""R20 plus four frozen airline arrival-recording window inputs; no uploads.

Prepare only until the desktop heavy slot is allocated. Classifier population,
labels, 68 original features, R20 settings, 1,700 rounds and seed42 stay fixed.
The feature builder and descriptive associations live in
tools/research_arrival_copy_windows_20261010.py; no windows are selected by CV.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import rome_adsblol_jan_jul as shared

ROOT=shared.ROOT
sys.path.insert(0,str(ROOT/'VALIDATION/tools'))
import research_arrival_copy_windows_20261010 as features

ORIGINAL_SOURCE=shared.OUT
REMOVE=list(shared.NEW)
OUT=shared.ts.RUNS/'v3_20261010_rome_arrival_windows_jan_jul'
DESIGN=dict(parent='v58',part='Rome recording classifier',new_features=features.FEATURES,
    retained_classifier_extras=list(shared.ts.CLASSIFIER),seed=42,rounds=1700,trial='R20',
    network_share=.71,heldout_months=shared.ts.SPLITS,
    training_membership='unchanged original R20 after_drop',
    checkpoint_selection=False,exploratory_reused_development_months=True,
    windows='centered ±24h and trailing7days; shares/counts',
    prior_weight=10,context='training arrivals only for training; training+held for held',
    unchanged_monthly_handler=True)


def identity():
    return dict(design=DESIGN, runner_sha256=shared.sha256(Path(__file__)),
                sources={str(Path(m.__file__).relative_to(ROOT)):shared.sha256(Path(m.__file__))
                                     for m in [features,shared]},
                feature_design_sha256=shared.sha256(features.OUT/'design.json'))


def guard():
    OUT.mkdir(parents=True,exist_ok=True)
    current=identity()
    path=OUT/'identity.json'
    if path.exists() and shared.ts.read(path)!=current:
        raise ValueError('Frozen feature/helper sources changed')
    shared.write_json(path,current)
    design=OUT/'design.json'
    if design.exists() and shared.ts.read(design)!=DESIGN:
        raise ValueError('Frozen experiment changed')
    shared.write_json(design,DESIGN)


def prepare():
    (OUT/'inputs').mkdir(exist_ok=True)
    for month in shared.ts.SPLITS:
        source=ORIGINAL_SOURCE/'inputs'/f'{month}.pkl'
        tables={kind:features.OUT/f'{month}_{kind}.parquet' for kind in ['train','held']}
        provenance=dict(original_input_sha256=shared.sha256(source),
                        feature_sha256={kind:shared.sha256(path) for kind,path in tables.items()},
                        identity=identity())
        dest=OUT/'inputs'/f'{month}.pkl'
        if dest.exists():
            old=shared.ts.read(dest.with_suffix('.json'))
            if any(old.get(k)!=v for k,v in provenance.items()) or old['prepared_sha256']!=shared.sha256(dest):
                raise ValueError('Prepared input provenance changed')
            continue
        d=pd.read_pickle(source)
        for part,kind,ids in [('A','train',d['train_ids']),('B','held',d['ids'][d['use']])]:
            original=d[part].drop(columns=REMOVE)
            assert original.shape[1]==68
            table=pd.read_parquet(tables[kind]).set_index('mvt_id')
            assert table.index.is_unique and np.isin(ids,table.index).all()
            added=table.loc[ids,features.FEATURES].set_axis(original.index)
            d[part]=pd.concat([original,added],axis=1)
            assert d[part].index.equals(original.index)
            pd.testing.assert_frame_equal(d[part][original.columns],original)
        assert list(d['A'])==list(d['B']) and d['A'].shape[1]==72
        assert len(d['target'])==len(d['A'])==len(d['train_ids'])
        assert int(d['use'].sum())==len(d['B'])
        tmp=dest.with_suffix('.tmp');pd.to_pickle(d,tmp);tmp.replace(dest)
        shared.write_json(dest.with_suffix('.json'),dict(**provenance,prepared_sha256=shared.sha256(dest),
            rows=len(d['A']),held_rome_rows=len(d['B']),positives=int(d['target'].sum()),
            columns=list(d['A']),source_sha256=shared.sha256(Path(__file__))))
        shared.log(f'{month}: prepared {len(d["A"]):,} / {len(d["B"]):,} Rome rows;72columns')


def checks():
    t=pd.Timestamp('2025-01-10T12:00:00Z')
    times=pd.to_datetime([t+pd.Timedelta(seconds=s) for s in [-7*86400,-86400,-1,0,86400,86401]])
    copied=np.array([True,False,True,False,True,False])
    taxi=np.full(6,300.)
    arrivals=pd.DataFrame(dict(mvt_id=np.arange(6),ADES_mvt='LIRF',FLIGHT_mvt='ITY1',
        MVT_TIME_UTC_mvt=times,taxi_sec=taxi,
        SCHED_TIME_UTC_mvt=times+pd.to_timedelta(taxi+np.where(copied,0,120),unit='s')))
    query=pd.DataFrame(dict(mvt_id=[10,11,12],ADEP_mvt='LIRF',MVT_TIME_UTC_mvt=t,
                            FLIGHT_mvt=['ITY1','XXX1',None]))
    f=features.build(query,arrivals)
    np.testing.assert_allclose(f.loc[:1,'handler_arr_copy_pm24h'],[.5,.5])
    np.testing.assert_allclose(f.loc[:1,'handler_arr_copy_prev7d'],[2/3,2/3])
    np.testing.assert_array_equal(f.loc[:1,'handler_arr_n_pm24h'],[4,0])
    np.testing.assert_array_equal(f.loc[:1,'handler_arr_n_prev7d'],[3,0])
    assert f.loc[2,features.FEATURES].isna().all()
    poisoned=query.assign(taxi_sec=1e12,BLOCK_TIME_UTC_mvt=t+pd.Timedelta(days=100))
    pd.testing.assert_frame_equal(f,features.build(poisoned,arrivals))
    shuffled=arrivals.sample(frac=1,random_state=42)
    pd.testing.assert_frame_equal(f,features.build(query,shuffled))
    shared.log('Synthetic boundaries, missing airline, zero support, order and departure-label isolation pass')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['check','prepare','fit','score'])
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    if args.threads<1: parser.error('threads must be positive')
    with threadpool_limits(2):
        if args.stage=='check':
            checks();return
        guard()
        if args.stage=='prepare':
            prepare();return
        shared.OUT,shared.ARM,shared.DESIGN=OUT,'arrival_windows',DESIGN
        if args.stage=='fit': shared.fit(args.threads)
        else: shared.score()


if __name__=='__main__':
    main()
