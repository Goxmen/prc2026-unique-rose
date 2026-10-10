"""Replay, serialize and submit the physical-scale change to v64.

All retained v64 components are reused. The ordinary neural component alone
changes before the original correction stack. Every build first reproduces v64
exactly. A separate current-fold replay supplies real predictions for subsequent
ensemble scoring; it never reconstructs predictions from squared errors.
"""
import argparse, json, os
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
import nn_track_physical_submit as production
import nn_track_physical_pool3 as pool
import callsign_submit as parent
import sparse_callsign_submit as selected
import modern_prediction_replay as replay
from records import sha256, write_json

ROOT=parent.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261011_physical_candidate'

def verify(r):
    p=ROOT/r['path'];assert sha256(p)==r['sha256'],p;return p

def tag(members):
    assert members in ([0],[0,1,2]);return 'single' if len(members)==1 else 'pool3'

def cv(members):
    dest=OUT/tag(members)/'cv';assert not dest.exists(),'Retain completed CV replay'
    run=production.cv if members==[0] else pool
    result=parent.read(run.OUT/'score.json');audit=parent.read(run.OUT/'audit.json')
    assert audit['status']=='pass' and audit['score']==parent.artifact(run.OUT/'score.json')
    identity=parent.read(production.cv.MANIFEST)
    sc,rome,_,trees=replay.baseline.baseline();rows=[];sources={}
    for m in production.cv.STOP:
        d=sc.d[m];ids=d['ids'];neural=[]
        for k in members:
            r=pool.completed(m,k,identity);assert r is not None
            neural.append(replay.values(verify(r['prediction']),ids))
            sources[str(pool.paths(m,k)[2].relative_to(ROOT))]=sha256(pool.paths(m,k)[2])
        calls=replay.values(replay.combo.sequence.CALLS_PRED,ids,'candidate_nn')
        main=replay.values(replay.combo.parent.control.candidate.OUT/'predictions'/f'{m}_s42.parquet',ids,'pred')
        tree=np.mean([main,trees['airport'][42][m],trees['catboost'][42][m]],axis=0)
        prob=.5*replay.combo.parent.cat.probabilities(m,ids)+.5*rome[m]
        p=sc.final(m,.71*.5*(calls+np.mean(neural,axis=0))+(1-.71)*tree,prob)
        se=(p-d['y'])**2
        score_rows=pd.read_parquet(run.OUT/'scored.parquet').set_index('mvt_id')
        col='physical_scales_single_mix' if members==[0] else 'physical3'
        np.testing.assert_array_equal(se,score_rows.loc[ids,col])
        rows.append(pd.DataFrame(dict(mvt_id=ids,month=m,day=d['day'],y=d['y'],pred=p,
            submitted=np.rint(np.maximum(p,0)),airport=d['held']['raw'].ADEP_mvt.to_numpy())))
    f=pd.concat(rows,ignore_index=True);assert len(f)==344419 and f.mvt_id.is_unique
    scores={m:float(np.sqrt(np.mean((x.pred-x.y)**2))) for m,x in f.groupby('month')}
    expected=result['scores'][col]['cv'];np.testing.assert_allclose(np.mean(list(scores.values())),expected,atol=1e-10,rtol=0)
    sources.update(replay.SOURCES)
    for p in [run.OUT/'score.json',run.OUT/'audit.json',run.OUT/'scored.parquet']:
        sources[str(p.relative_to(ROOT))]=sha256(p)
    dest.mkdir(parents=True);f.to_parquet(dest/'predictions.parquet',index=False)
    write_json(dest/'result.json',dict(members=members,scores=scores,cv=expected,
        rounded_cv=float(np.mean([np.sqrt(np.mean((x.submitted-x.y)**2)) for _,x in f.groupby('month')])),
        prediction=parent.artifact(dest/'predictions.parquet'),sources=sources,source=parent.artifact(__file__),
        note='Existing current-fold predictions; reused development months; all final squared errors reproduce exactly.'))
    print('Current-fold replay complete',scores,flush=True)

def prepare(members,version):
    dest=OUT/f'submission_v{version}';output=ROOT/'submissions'/f'unique-rose_v{version}.parquet'
    assert not dest.exists() and not output.exists(),'Preserve existing candidate'
    cp=OUT/tag(members)/'cv/result.json';c=parent.read(cp)
    assert c['members']==members and c['source']==parent.artifact(__file__)
    verify(c['prediction'])
    for path,digest in c['sources'].items():assert sha256(ROOT/path)==digest,path
    pm=production.guard();sources={str(cp.relative_to(ROOT)):sha256(cp),
        str(production.MANIFEST.relative_to(ROOT)):sha256(production.MANIFEST)}
    mp=selected.OUT/'candidates/old/manifest.json';manifest=parent.read(mp)
    reference=pd.read_parquet(verify(manifest['parts']))
    data=parent.ranking_data('old');ids=data['rank_ids']
    np.testing.assert_array_equal(reference.mvt_id,ids)
    template=pq.read_table(verify(manifest['template']))
    zero,masks=parent.final_frame(data,reference.nn_base.to_numpy(),reference.gbm_base.to_numpy(),reference.p.to_numpy())
    pd.testing.assert_frame_equal(zero,reference[zero.columns],check_exact=True)
    assert masks==manifest['correction_masks']
    parent.assert_parent_output(zero,template,pq.read_table(verify(manifest['output'])),reference)
    values=[]
    for k in members:
        r=production.completed(k,pm);assert r is not None
        f=pd.read_parquet(verify(r['predictions']['old']));np.testing.assert_array_equal(f.mvt_id,ids)
        assert np.isfinite(f.base).all() and f.base.ge(0).all();values.append(f.base.to_numpy(float))
        for p in production.paths(k):sources[str(p.relative_to(ROOT))]=sha256(p)
        for a in r['predictions'].values():sources[a['path']]=a['sha256']
    neural=.5*(reference.calls_nn.to_numpy(float)+np.mean(values,axis=0))
    final,masks=parent.final_frame(data,neural,reference.gbm_base.to_numpy(),reference.p.to_numpy())
    table=selected.serialized(template,ids,final.pred.to_numpy())
    dest.mkdir(parents=True);pq.write_table(table,output);assert pq.read_table(output).equals(table)
    final.to_parquet(dest/'parts.parquet',index=False)
    for p in [mp,verify(manifest['parts']),verify(manifest['output']),verify(manifest['template']),
              parent.fs.WORK/'inputs_old.pkl',parent.fs.OUT/'inputs_old.json',Path(__file__)]:
        sources[str(p.relative_to(ROOT))]=sha256(p)
    write_json(dest/'manifest.json',dict(version=version,state='prepared',file=str(output.relative_to(ROOT)),
        sha256=sha256(output),rows=344841,members=members,jan_jul_cv=c['cv'],rounded_cv=c['rounded_cv'],
        parent='v64',change='physical-scale sparse track encoder',parent_replay_exact=True,
        authorization='October11 user authorized autonomous uploads',sources=sources,
        model_host='laptop RTX5060',cv_host='desktop RTX3080',correction_masks=masks,
        audit=dict(parent_raw_and_serialized_exact=True,schema_and_ids_exact=True,output_readback_exact=True),
        parts=parent.artifact(dest/'parts.parquet')))
    print(f'Prepared v{version} physical {tag(members)}; CV {c["cv"]:.6f}',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['cv','prepare'])
    ap.add_argument('--members',nargs='+',type=int,default=[0]);ap.add_argument('--version',type=int)
    a=ap.parse_args()
    with threadpool_limits(2):
        if a.stage=='cv':cv(a.members)
        else:prepare(a.members,a.version)
