"""Evaluate a frozen, leaderboard-selected affine blend on current Jan/Jul folds.

Weights are chosen from our own aggregate leaderboard scores. This check is
not independent confirmation. Each component and the combined output receive
the same nonnegative integer serialization used by real submission files.
"""
import json,os,sys
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'VALIDATION'),str(ROOT/'pipeline')]
import numpy as np,pandas as pd
from records import sha256,write_json
import tuning_seasonal as ts
RUNS=ROOT/'VALIDATION/runs'
GEOMETRY=RUNS/'v3_20261011_target225/geometry/validation_ready_basis.json'
OUT=RUNS/'v3_20261011_scored_blend_jan_jul'
BASIC=RUNS/'v3_20261011_legacy_basic_jan_jul'
RAW=RUNS/'v3_20261011_legacy_raw_jan_jul'
HYBRID=RUNS/'v3_20261011_legacy_hybrid_jan_jul'
MODERN=RUNS/'v3_20261011_modern_prediction_replay'
INPUTS={}

def retain(path):
    path=Path(path);INPUTS[str(path.relative_to(ROOT))]=sha256(path)
    return path

def load(path,ids,col):
    f=pd.read_parquet(retain(path)).set_index('mvt_id')
    assert f.index.is_unique and set(f.index)==set(ids),str(path)
    v=f.loc[ids,col].to_numpy(float)
    assert np.isfinite(v).all()
    return v

def saved(folder,m,v,ids):
    path=folder/'predictions'/f'{m}_v{v}.parquet'
    record_path=(folder/'records'/f'{m}_v{v}.json') if folder==RAW else path.with_suffix('.json')
    r=ts.read(retain(record_path))
    digest=r.get('prediction_sha256') or r['prediction']['sha256']
    assert sha256(path)==digest
    return load(path,ids,'pred')

def main():
    assert not (OUT/'result.json').exists(),'Retain completed validation'
    geometry=ts.read(retain(GEOMETRY));weights=geometry['weights'];versions=geometry['versions']
    assert set(weights)==set(map(str,versions)) and abs(sum(weights.values())-1)<1e-8
    for directory in [BASIC,RAW,HYBRID,MODERN]:
        record=ts.read(retain(directory/'complete.json'))
        if 'identity_sha256' in record:assert sha256(retain(directory/'identity.json'))==record['identity_sha256']
        elif directory!=MODERN:
            assert record['identity']==ts.read(retain(directory/'identity.json'))
        if directory==MODERN:
            assert record['source_sha256']==sha256(ROOT/'VALIDATION/modern_prediction_replay.py')
            for part in record['parts']:assert sha256(retain(ROOT/part['path']))==part['sha256']
    progress=ts.read(retain(BASIC/'progress.json'))
    for v in progress['predictions'].values():
        for name,digest in v['artifacts'].items():assert sha256(retain(BASIC/name))==digest
    OUT.mkdir(parents=True,exist_ok=True)
    frames=[];component_scores={};preclip_scores={};summary={}
    for m in ts.SPLITS:
        f=pd.read_parquet(retain(MODERN/f'{m}.parquet'))
        ids=f.mvt_id.to_numpy();y=f.y.to_numpy(float)
        assert f.month.eq(m).all() and len(np.unique(ids))==len(ids)
        parts=[]
        for v in versions:
            if f'v{v}' in f: a=f[f'v{v}'].to_numpy(float)
            elif v in [12,13,14,15,16]:
                parent,col={12:('baseline','B0-800'),13:('baseline','Q0'),14:('recovery','Q1'),
                            15:('baseline','Q0-4h'),16:('recovery','Q1-4h')}[v]
                path=BASIC/f'janjul_{m}_{parent}_{m}_42.parquet'
                np.testing.assert_array_equal(load(path,ids,'y'),y)
                a=load(path,ids,col)
            else:a=saved(RAW if v<22 else HYBRID,m,v,ids)
            a=np.rint(np.maximum(a,0))
            component_scores.setdefault(str(v),{})[m]=float(np.sqrt(np.mean((a-y)**2)))
            parts.append(a)
        matrix=np.column_stack(parts);w=np.array([weights[str(v)] for v in versions])
        raw=matrix@w;pred=np.rint(np.maximum(raw,0))
        f=f[['mvt_id','month','day','airport','y']].copy()
        f['pred']=pred;f['raw_pred']=raw;f['v64']=matrix[:,versions.index(64)]
        for v,a in zip(versions,parts):f[f'v{v}']=a
        f['se_control']=(f.v64-y)**2;f['se_candidate']=(pred-y)**2
        preclip_scores[m]=float(np.sqrt(np.mean((raw-y)**2)))
        summary[m]=dict(rows=len(f),candidate=float(np.sqrt(f.se_candidate.mean())),
            control=float(np.sqrt(f.se_control.mean())),negative_before_clipping=int((raw<0).sum()),
            mean_error_sec=float(np.mean(pred-y)))
        frames.append(f)
    allrows=pd.concat(frames,ignore_index=True)
    assert len(allrows)==344419 and allrows.mvt_id.is_unique
    path=OUT/'scored.parquet';allrows.to_parquet(path,index=False)
    for scores in component_scores.values():scores['cv']=float(np.mean(list(scores.values())))
    paired=ts.paired_bootstrap(allrows.day,allrows.month,allrows.se_control,allrows.se_candidate)
    report=dict(selection='own leaderboard scores; not independent confirmation',
        source_sha256=sha256(__file__),inputs=INPUTS,geometry_sha256=sha256(GEOMETRY),weights=weights,
        versions=versions,folds=ts.FOLDS,rows=len(allrows),component_scores=component_scores,months=summary,
        candidate_cv=float(np.mean([v['candidate'] for v in summary.values()])),
        control_cv=float(np.mean([v['control'] for v in summary.values()])),
        unclipped_cv=float(np.mean(list(preclip_scores.values()))),paired=paired,
        scored_sha256=sha256(path),serialization='round(max(pred,0)) per component, then after weighted sum',
        reconstruction_caveat='v3-v9 use an archived original27 implementation; missing exact historical source is explicit in raw runner identity',
        machine='legacy raw/recovery laptop; saved modern controls and legacy hybrids desktop')
    write_json(OUT/'result.json',report)
    print(json.dumps({k:report[k] for k in ['candidate_cv','control_cv','unclipped_cv','months','paired']},indent=2),flush=True)

if __name__=='__main__':main()
