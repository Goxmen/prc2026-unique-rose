"""Training-only feasibility of own-aircraft previous-arrival copy persistence."""
import json
import os
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'tmp/research_20261010/error_mechanisms/own_arrival_copy_association.json'
ROT=ROOT/'data/derived/aircraft_rotation_20261010'

def rates(frame):
    return {str(key):dict(rows=len(g),departure_copy_rate=float(g.y.mean()),
        age_median_sec=float(g.age.median()) if g.age.notna().any() else None)
        for key,g in frame.groupby('own_copy',dropna=False,observed=True)}

def conditional(frame,keys):
    group=frame.dropna(subset=['own_copy']).groupby(keys+['own_copy'],observed=True).y.agg(['mean','size']).reset_index()
    p=group.pivot(index=keys,columns='own_copy',values=['mean','size'])
    if 0. not in p['size'] or 1. not in p['size']:return dict(groups=0)
    good=p[p['size'][0.].ge(30)&p['size'][1.].ge(30)]
    if not len(good):return dict(groups=0)
    w=good['size'][0.]*good['size'][1.]/(good['size'][0.]+good['size'][1.])
    diff=good['mean'][1.]-good['mean'][0.]
    return dict(groups=len(good),rows=int((good['size'][0.]+good['size'][1.]).sum()),
        weighted_copy_probability_difference=float(np.average(diff,weights=w)),
        positive_groups=int((diff>0).sum()),negative_groups=int((diff<0).sum()))

def main():
    manifest=json.loads((ROT/'folds.json').read_text());report={}
    for month in ['2025-01','2025-07']:
        parent=json.loads((ROOT/'VALIDATION/runs/v3_20261010_rome_catboost_jan_jul/predictions'/f'{month}_s42.json').read_text())
        d=pd.read_pickle(ROOT/'VALIDATION/runs/v3_20261010_rome_adsblol_jan_jul/inputs'/f'{month}.pkl')
        X=d['A'][parent['columns']]
        assert len(X)==parent['training_rows'] and int(np.sum(d['target']))==parent['positives']
        refs=manifest['folds'][month]['train']['reference_months'];assert month not in refs
        table=pd.read_parquet(ROT/f'{month}_train.parquet',columns=['mvt_id','ar_prev_arrival_delay_sec','ar_prev_inblock_age_sec']).set_index('mvt_id')
        own=table.loc[d['train_ids']]
        offset=own.ar_prev_arrival_delay_sec.to_numpy(float)
        frame=pd.DataFrame(dict(y=np.asarray(d['target'],float),
            own_copy=np.where(np.isfinite(offset),(np.abs(offset)<=60).astype(float),np.nan),
            age=own.ar_prev_inblock_age_sec.to_numpy(float),airline=X.airline.astype(object).fillna('<MISSING>').to_numpy(),
            month=X.month.to_numpy(),noid=X.clock_raw_flight_id_missing.to_numpy()==1,gap=X.mvt_minus_sched_sec.to_numpy(float)))
        slices={'all':np.ones(len(frame),bool),'no_id':frame.noid,'no_id_gap_above1h':frame.noid&(frame.gap>3600)}
        result={name:dict(rows=int(np.sum(mask)),own_available=int(frame.loc[mask].own_copy.notna().sum()),
            rates=rates(frame.loc[mask])) for name,mask in slices.items()}
        result.update(reference_months=refs,query_months=sorted(map(int,frame.month.unique())),
            own_available=int(frame.own_copy.notna().sum()),
            conditional_airline_month=conditional(frame,['airline','month']))
        report[month]=result
    output=dict(design='Training-only original R20 query rows; own arrival features from already audited allowed outer-training references only. Copy tolerance60s; no held-departure labels or threshold selection.',
        existing_feature='Original68 lack own-aircraft previous-arrival copy flag. st_inblock_minus_sched_sec is previous-stand-arrival inblock minus CURRENT DEPARTURE schedule (stand_day.py:21), not arrival schedule, and cannot serve as its copy flag.',
        conditional_support='Each own-copy state must have at least30 rows within a stratum; weight n0*n1/(n0+n1). Descriptive association, not validation.',folds=report)
    OUT.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))

if __name__=='__main__':main()
