"""Matched laptop LB1 control for the selected six-callsign-feature lead.

Reuse the exact locally fingerprinted original 156 matrices, row order and
candidate parameters. Fit seed 42 for 10,000 rounds; change no features. This
control resolves machine numerical variation before interpreting the feature
gain. No full-year fits or uploads. Fits require explicit root allocation.
"""
import argparse
import gc
import json
from pathlib import Path
import platform
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'VALIDATION'), str(ROOT/'pipeline'), str(ROOT/'VALIDATION/tools')]
import callsign_lgb_jan_jul as candidate
from records import sha256, write_json

ts, hpo, ideas = candidate.ts, candidate.hpo, candidate.ideas
OUT = ts.RUNS/'v3_20261010_callsign_lgb_control_jan_jul'
DESIGN = dict(parent='v58 original156', purpose='matched laptop control for callsign feature screen',
              trial='LB1', rounds=10000, seed=42, feature_count=156)


def identity():
    return dict(runner_sha256=sha256(Path(__file__)),
                helpers={str(Path(m.__file__).relative_to(ROOT)):sha256(Path(m.__file__))
                         for m in [candidate, candidate.input_cache, hpo, ideas, candidate.modeling]})


def guard():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, value in [('design', DESIGN), ('code_identity', identity())]:
        path=OUT/f'{name}.json'
        if path.exists() and ts.read(path)!=value:
            raise ValueError(f'Changed {name}; use a separate control run')
        write_json(path,value)


def fit(months, threads):
    for month in months:
        source=candidate.input_cache.input_path(month)
        candidate_record=ts.read(candidate.OUT/'predictions'/f'{month}_s42.json')
        # Read parameters from the actual completed candidate, not a new trial.
        params=dict(candidate_record['params'])
        if params['num_threads']!=threads or candidate_record['rounds']!=10000 or candidate_record['seed']!=42:
            raise ValueError('Control must use the exact candidate threads/rounds/seed')
        if sha256(source)!=candidate_record['inputs_sha256']:
            raise ValueError('Local input pickle differs from the completed candidate')
        provenance=dict(identity=identity(), source_sha256=sha256(source),
                        candidate_record_sha256=sha256(candidate.OUT/'predictions'/f'{month}_s42.json'),
                        params=params, rounds=10000)
        dest=OUT/'predictions'/f'{month}_s42.parquet'
        if dest.exists():
            old=ts.read(dest.with_suffix('.json'))
            if old['provenance']!=provenance or old['prediction_sha256']!=sha256(dest):
                raise ValueError('Saved control changed')
            continue
        data=pd.read_pickle(source)
        names=data['baseline']
        if len(names)!=156 or list(candidate_record['columns'])!=names+candidate.FEATURES:
            raise ValueError('Original input columns differ')
        X,V=data['X'].loc[:,names],data['V'].loc[:,names]
        cats=ideas.signals._categorical(X.columns)+[c for c in ideas.ef.CATEGORICAL if c in X.columns]
        if cats!=candidate_record['categorical']:
            raise ValueError('Categorical order differs from candidate')
        started=time.monotonic()
        candidate.log(f'{month}: matched original156 control, {len(X):,} rows')
        model=lgb.train(params,lgb.Dataset(X,data['y'],categorical_feature=cats),num_boost_round=10000)
        pred=candidate.modeling.predict(model,V,threads=threads)
        if not np.isfinite(pred).all():
            raise ValueError('Nonfinite control prediction')
        dest.parent.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(dict(mvt_id=data['ids'],pred=pred)).to_parquet(dest,index=False)
        write_json(dest.with_suffix('.json'),dict(provenance=provenance,
            prediction_sha256=sha256(dest),columns=names,categorical=cats,
            train_rows=len(X),held_rows=len(V),host=platform.node(),
            minutes=(time.monotonic()-started)/60))
        candidate.log(f'{month}: control saved')
        del model,data,X,V
        gc.collect()


def score():
    sc,rome,pool,trees=candidate.matched.baseline()
    errors={name:{} for name in ['historical','laptop_control','callsign']}
    rows=[]
    for month in ts.SPLITS:
        frames={}
        for name,folder in [('laptop_control',OUT),('callsign',candidate.OUT)]:
            path=folder/'predictions'/f'{month}_s42.parquet'
            record=ts.read(path.with_suffix('.json'))
            if sha256(path)!=record['prediction_sha256']:
                raise ValueError('Prediction hash differs')
            f=pd.read_parquet(path)
            np.testing.assert_array_equal(f.mvt_id,sc.d[month]['ids'])
            frames[name]=f.pred.to_numpy(float)
        base={hpo.MEMBER_OF[p]:trees[p][42][month] for p in trees}
        errors['historical'][month]=sc.se(month,42,nn=pool[month],p=rome[month],**base)
        for name,values in frames.items():
            members=dict(base);members[hpo.MEMBER_OF['lgb']]=values
            errors[name][month]=sc.se(month,42,nn=pool[month],p=rome[month],**members)
        rows.append(pd.DataFrame(dict(mvt_id=sc.d[month]['ids'],month=month,day=sc.d[month]['day'],
            **{f'se_{name}':v[month] for name,v in errors.items()})))
    frame=pd.concat(rows,ignore_index=True)
    report=dict(design=DESIGN,scores={name:sc.cv(v) for name,v in errors.items()},comparisons={})
    for a,b in [('callsign','laptop_control'),('laptop_control','historical')]:
        report['comparisons'][f'{a}_vs_{b}']=ts.paired_bootstrap(frame.day,frame.month,
                                                       frame[f'se_{b}'],frame[f'se_{a}'])
    report['caveat']='Same reused development months; matched machine controls numerical variation, not generalization.'
    frame.to_parquet(OUT/'scored_rows.parquet',index=False)
    write_json(OUT/'result.json',report)
    candidate.log(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['check','fit','score'])
    p.add_argument('--months',nargs='+',choices=ts.SPLITS,default=ts.SPLITS)
    p.add_argument('--threads',type=int,default=8)
    a=p.parse_args()
    guard()
    with threadpool_limits(limits=a.threads):
        if a.stage=='fit':fit(a.months,a.threads)
        elif a.stage=='score':score()
        else:candidate.log('Matched control source identity passes; no fits run')


if __name__=='__main__':
    main()
