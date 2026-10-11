"""R20 with input-only public daily aggregates, fixed two-fold screen.

Twelve features; original68 inputs, population, labels, seed42 and1700 rounds.
Primary comparison replaces R20's half of the selected Cat3/R20 mixture inside
v70's retained ordinary model. Also report a fixed delta transfer to v74, with
its physical constraints restored. No production fit or upload.
"""
import argparse
from pathlib import Path
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
import rome_adsblol_jan_jul as shared
import public_delay_features as features
import physical_candidate as physical
import public_constraints_candidate as constraints
from records import write_json,sha256
ROOT=shared.ROOT
OUT=shared.ts.RUNS/'v3_20261011_rome_public_delay'
DESIGN=dict(parent='v70 half Cat3/R20 Rome classifier',new_features=features.FEATURES,
    retained_classifier_extras=list(shared.ts.CLASSIFIER),seed=42,rounds=1700,trial='R20',
    training_membership='unchanged original R20 after_drop',heldout_months=shared.ts.SPLITS,
    no_hyperparameter_search=True,public_population='IFR and unknown flight rule; explicit VFR excluded',
    feature_targets='none; raw clocks and public airport-day total only',
    secondary='same final-prediction delta added to v74; restore matched and recovered plan bounds')

def guard():
    OUT.mkdir(parents=True,exist_ok=True)
    current=dict(design=DESIGN,source=features.base.artifact(__file__),
        helpers=[features.base.artifact(m.__file__) for m in [shared,features,physical,constraints]],
        features=features.base.artifact(features.OUT/'manifest.json'))
    p=OUT/'identity.json'
    if p.exists():assert shared.ts.read(p)==current
    else:write_json(p,current)

def prepare():
    table=pd.read_parquet(features.OUT/'features2025.parquet').set_index('mvt_id')
    (OUT/'inputs').mkdir(exist_ok=True)
    for month in shared.ts.SPLITS:
        src=shared.OUT/'inputs'/f'{month}.pkl';dest=OUT/'inputs'/f'{month}.pkl'
        assert not dest.exists(),'Preserve prepared matrices'
        d=pd.read_pickle(src)
        for part,ids in [('A',d['train_ids']),('B',d['ids'][d['use']])]:
            original=d[part].drop(columns=shared.NEW);assert original.shape[1]==68
            extra=table.loc[ids,features.FEATURES].set_axis(original.index)
            d[part]=pd.concat([original,extra],axis=1)
            pd.testing.assert_frame_equal(d[part][original.columns],original)
        assert list(d['A'])==list(d['B']) and d['A'].shape[1]==80
        pd.to_pickle(d,dest)
        write_json(dest.with_suffix('.json'),dict(original=features.base.artifact(src),prepared=features.base.artifact(dest),
            features=features.base.artifact(features.OUT/'features2025.parquet'),train_rows=len(d['A']),held_rows=len(d['B']),columns=list(d['A'])))
        print(month,len(d['A']),len(d['B']),flush=True)

def score():
    replay=physical.replay;sc,rome,_,trees=replay.baseline.baseline();frames=[]
    parent_cv=pd.read_parquet(physical.OUT/'pool3/cv/predictions.parquet').set_index('mvt_id')
    best=pd.read_parquet(constraints.OUT/'submission_v74/cv.parquet').set_index('mvt_id')
    raw=pd.read_parquet(features.base.OUT/'rows2025.parquet').set_index('mvt_id')
    matches=pd.read_parquet(constraints.exclusion.OUT/'matches2025.parquet')
    identity=shared.ts.read(physical.production.cv.MANIFEST)
    for month in shared.ts.SPLITS:
        d=sc.d[month];ids=d['ids'];path=OUT/'predictions'/f'{month}_s42.parquet'
        record=shared.ts.read(path.with_suffix('.json'));assert sha256(path)==record['prediction_sha256']
        cand=pd.read_parquet(path);np.testing.assert_array_equal(cand.mvt_id,ids)
        calls=replay.values(replay.combo.sequence.CALLS_PRED,ids,'candidate_nn')
        neural=[replay.values(physical.verify(physical.pool.completed(month,k,identity)['prediction']),ids) for k in [0,1,2]]
        main=replay.values(replay.combo.parent.control.candidate.OUT/'predictions'/f'{month}_s42.parquet',ids,'pred')
        tree=np.mean([main,trees['airport'][42][month],trees['catboost'][42][month]],axis=0)
        ordinary=.71*.5*(calls+np.mean(neural,axis=0))+.29*tree
        cat=replay.combo.parent.cat.probabilities(month,ids)
        control=sc.final(month,ordinary,.5*cat+.5*rome[month])
        np.testing.assert_allclose(control,parent_cv.loc[ids,'pred'],atol=1e-9,rtol=0)
        candidate=sc.final(month,ordinary,.5*cat+.5*cand.p.to_numpy(float))
        v74=best.loc[ids,'pred'].to_numpy(float);transferred=v74+candidate-control
        transferred=constraints.matched_bound(raw.loc[ids].reset_index(),transferred)
        permitted=matches[matches.donor_month.isin(features.base.FOLDS[month]+[month])]
        transferred,_=constraints.exclusion.apply(ids,np.maximum(0,transferred),permitted)
        frames.append(pd.DataFrame(dict(mvt_id=ids,month=month,day=d['day'],airport=raw.loc[ids,'airport'].to_numpy(),y=d['y'],
            control=control,pred=candidate,v74=v74,transferred=transferred,p_old=rome[month],p_new=cand.p.to_numpy(float))))
    f=pd.concat(frames,ignore_index=True);f.to_parquet(OUT/'scored.parquet',index=False)
    scores={m:{n:float(np.sqrt(np.mean((q[n]-q.y)**2))) for n in ['control','pred','v74','transferred']} for m,q in f.groupby('month')}
    paired={f'{new}_vs_{old}':shared.ts.paired_bootstrap(f.day,f.month,(f[old]-f.y)**2,(f[new]-f.y)**2) for old,new in [('control','pred'),('v74','transferred')]}
    write_json(OUT/'result.json',dict(design=DESIGN,scores=scores,cv={n:float(np.mean([q[n] for q in scores.values()])) for n in ['control','pred','v74','transferred']},paired=paired,source=features.base.artifact(__file__)))
    print(scores,paired,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','fit','score']);a=p.parse_args()
    with threadpool_limits(2):
        guard()
        if a.stage=='prepare':prepare()
        elif a.stage=='fit':
            shared.OUT,shared.ARM,shared.DESIGN=OUT,'public_delay',DESIGN
            shared.fit(2)
        else:score()
