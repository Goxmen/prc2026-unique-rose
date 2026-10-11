"""Prepare and evaluate scored combinations including the new public models.

Preparation freezes the actual submitted files and matching Jan/Jul vectors
without requiring scores for v75/v76. Evaluation requires their official
successful results and a fresh public score check. It is leaderboard selection,
not independent validation. This tool never uploads or reserves a version.
"""
import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import scored_submission_geometry as geometry
import scored_blend_jan_jul as legacy
import physical_candidate as physical
import scored_blend_submit as transport
from records import sha256, write_json

ROOT = geometry.ROOT
OUT = ROOT/'VALIDATION/runs/v3_20261011_public_models_geometry'
ADDITIONS = [70,72,74,75,76]
DESIGN = dict(parent='retained scored basis plus physical pool, daily and monthly corrections',
    additions=ADDITIONS, new_directions='public Rome and combined public main/neural models',
    bounds=[-3.,4.], ridge_mse=1., sum_weights=1., no_model_fit=True,
    score_selection='own official aggregate RMSE; reused Jan/Jul reported separately',
    no_score_imputation=True, no_rounding_directions=True,
    output='read-only diagnosis; no submission or version reservation',
    required_rows=344841, required_cv_rows=344419)


def read(path):
    return legacy.ts.read(path)


def art(path):
    path=Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)),sha256=sha256(path))


def verify(record):
    path=ROOT/record['path']
    assert sha256(path)==record['sha256'],str(path)
    return path


def manifest_path(version):
    paths=list((ROOT/'VALIDATION/runs').glob(f'**/submission_v{version}/manifest.json'))
    assert len(paths)==1,(version,paths)
    return paths[0]


def cv_input(version,manifest,path):
    if version==70:
        record=physical.OUT/physical.tag(manifest['members'])/'cv/result.json'
        data=read(record)
        return verify(data['prediction']),record,'submitted'
    if version==76:
        record=verify(manifest['cv_record'])
        prediction=path.parent/'cv.parquet'
        assert str(prediction.relative_to(ROOT)) in manifest['sources']
        return prediction,record,'pred'
    record=verify(manifest['cv_record'])
    return verify(read(record)['prediction']),record,'submitted'


def check_geometry():
    # A synthetic hidden target provides an independent oracle for the formula
    # and its four-decimal score plus output-serialization error bound.
    rng=np.random.default_rng(37291)
    y=rng.normal(900,280,3000)
    noise=rng.normal(0,100,(3000,4))
    P=y[:,None]+noise+np.array([0.,30.,-20.,50.])
    scores=np.sqrt(np.mean((P-y[:,None])**2,axis=0))
    versions=[64,70,75,76]
    result,pred=geometry.solve(P,np.round(scores,4),versions,-3.,4.,1.)
    w=np.array([result['weights'][str(v)] for v in versions])
    direct=float(np.sqrt(np.mean((pred-y)**2)))
    lo,hi=result['score_rounding_rmse_interval']
    assert lo-1e-9<=direct<=hi+1e-9
    saved=np.rint(np.maximum(pred,0))
    distance=float(np.sqrt(np.mean((saved-pred)**2)))
    submitted=float(np.sqrt(np.mean((saved-y)**2)))
    assert max(0,lo-distance)-1e-9<=submitted<=hi+distance+1e-9
    # This expanded identity uses absolute predictions instead of the solver's
    # differences from v64, so it also checks reference and weight alignment.
    expanded=float(w@(np.round(scores,4)**2)-w@np.mean(P*P,axis=0)+np.mean(pred*pred))
    np.testing.assert_allclose(np.sqrt(expanded),result['projected_rmse'],atol=1e-8,rtol=0)
    return dict(synthetic_hidden_target=True,score_rounding_bound=True,
        exact_serialization_triangle_bound=True,independent_expanded_mse=True)


def prepare():
    assert not (OUT/'prepared.json').exists(),'Preserve prepared matrices'
    OUT.mkdir(parents=True,exist_ok=True)
    original=geometry.OUT/'validation_ready_basis.json'
    frozen=read(original);result=read(legacy.OUT/'result.json')
    assert result['geometry_sha256']==sha256(original)
    assert result['scored_sha256']==sha256(legacy.OUT/'scored.parquet')
    sources=dict(result['inputs'])
    for path,digest in sources.items():assert sha256(ROOT/path)==digest,path
    for path in [Path(__file__),Path(geometry.__file__),original,legacy.OUT/'result.json',legacy.OUT/'scored.parquet']:
        sources[str(path.relative_to(ROOT))]=sha256(path)
    ids,all_predictions,all_scores,all_versions,records=geometry.load(1)
    versions=list(frozen['versions'])
    positions=[all_versions.index(v) for v in versions]
    ranking=all_predictions[:,positions]
    known_scores={str(v):float(all_scores[all_versions.index(v)]) for v in versions}
    held=pd.read_parquet(legacy.OUT/'scored.parquet')
    assert len(ids)==344841 and len(held)==344419 and held.mvt_id.is_unique
    validation=held[[f'v{v}' for v in versions]].to_numpy(float)
    additions={}
    for version in versions:
        record=records[str(version)]
        sources[record['file']]=record['sha256']
    for version in ADDITIONS:
        path=manifest_path(version);manifest=read(path)
        assert manifest['version']==version and manifest['rows']==344841
        assert manifest['state'] in ['prepared','uploaded','scored']
        for filename,digest in manifest['sources'].items():
            assert sha256(ROOT/filename)==digest,filename
        file=ROOT/manifest['file'];assert sha256(file)==manifest['sha256']
        prediction=pd.read_parquet(file)
        np.testing.assert_array_equal(prediction.MVT_ID_mvt,ids)
        cvpath,cvrecord,column=cv_input(version,manifest,path)
        cvframe=pd.read_parquet(cvpath).set_index('mvt_id').loc[held.mvt_id]
        assert cvframe.index.is_unique and len(cvframe)==len(held)
        for name in ['month','day','y']:np.testing.assert_array_equal(cvframe[name],held[name])
        cvvalues=np.rint(np.maximum(cvframe[column].to_numpy(float),0))
        if column=='submitted':np.testing.assert_array_equal(cvvalues,cvframe[column])
        ranking=np.column_stack([ranking,prediction.TAXITIME_SEC_mvt.to_numpy(float)])
        validation=np.column_stack([validation,cvvalues])
        versions.append(version)
        snapshot=OUT/f'manifest_v{version}_snapshot.json'
        if snapshot.exists():
            assert read(snapshot)==manifest,'Partial preparation snapshot changed'
        else:
            write_json(snapshot,manifest)
        additions[str(version)]=dict(snapshot=art(snapshot),live_manifest=str(path.relative_to(ROOT)),
            ranking=art(file),cv=art(cvpath),cv_record=art(cvrecord),cv_column=column,
            cv_serialization='maximum0 and nearest integer')
        # Mutable upload manifests are represented by immutable snapshots.
        for item in [file,cvpath,cvrecord,snapshot]:sources[str(item.relative_to(ROOT))]=sha256(item)
    assert len(set(versions))==len(versions) and np.isfinite(ranking).all() and np.isfinite(validation).all()
    rankframe=pd.DataFrame(ranking,columns=[f'v{v}' for v in versions]);rankframe.insert(0,'mvt_id',ids)
    cvframe=held[['mvt_id','month','day','airport','y']].copy()
    for i,version in enumerate(versions):cvframe[f'v{version}']=validation[:,i]
    rankframe.to_parquet(OUT/'ranking_basis.parquet',index=False)
    cvframe.to_parquet(OUT/'cv_basis.parquet',index=False)
    record=dict(design=DESIGN,versions=versions,known_scores=known_scores,additions=additions,
        sources=sources,ranking=art(OUT/'ranking_basis.parquet'),cv=art(OUT/'cv_basis.parquet'),
        checks=check_geometry(),historical_cv_caveat=result['reconstruction_caveat'])
    write_json(OUT/'prepared.json',record)
    print(dict(prepared=True,versions=len(versions),rows=len(ids),cv_rows=len(held)),flush=True)


def audit():
    assert not (OUT/'audit.json').exists(),'Preserve completed audit'
    prepared=read(OUT/'prepared.json')
    for record in [prepared['ranking'],prepared['cv']]:verify(record)
    with duckdb.connect(config={'threads':1}) as con:
        con.read_parquet(str(verify(prepared['ranking']))).create_view('ranking')
        con.read_parquet(str(verify(prepared['cv']))).create_view('validation')
        con.read_parquet(str(legacy.OUT/'scored.parquet')).create_view('old')
        assert con.execute('select count(*),count(distinct mvt_id) from ranking').fetchone()==(344841,344841)
        assert con.execute('select count(*),count(distinct mvt_id) from validation').fetchone()==(344419,344419)
        assert con.execute('''select count(*) from validation v full join old o using(mvt_id)
            where v.mvt_id is null or o.mvt_id is null or v.y is distinct from o.y
            or v.month is distinct from o.month or v.day is distinct from o.day''').fetchone()[0]==0
        for version in prepared['versions']:
            if str(version) in prepared['additions']:
                rec=prepared['additions'][str(version)]
                con.read_parquet(str(verify(rec['ranking']))).create_view('original_ranking',replace=True)
                con.read_parquet(str(verify(rec['cv']))).create_view('original_cv',replace=True)
                field=rec['cv_column']
                assert con.execute(f'''select count(*) from ranking r full join original_ranking o
                    on r.mvt_id=o.MVT_ID_mvt where r.mvt_id is null or o.MVT_ID_mvt is null
                    or r.v{version} is distinct from o.TAXITIME_SEC_mvt''').fetchone()[0]==0
                assert con.execute(f'''select count(*) from validation v full join original_cv o using(mvt_id)
                    where v.mvt_id is null or o.mvt_id is null or v.y is distinct from o.y
                    or v.month is distinct from o.month or v.day is distinct from o.day
                    or v.v{version} is distinct from round_even(greatest(0,o.{field}),0)''').fetchone()[0]==0
            else:
                assert con.execute(f'''select count(*) from validation v join old o using(mvt_id)
                    where v.v{version} is distinct from o.v{version}''').fetchone()[0]==0
    write_json(OUT/'audit.json',dict(status='pass',prepared=art(OUT/'prepared.json'),
        all_ranking_rows=344841,all_cv_rows=344419,all_new_vectors_independent_sql_exact=True,
        original_cv_vectors_exact=True,geometry_checks=check_geometry(),source=art(__file__)))
    print('Prepared matrix and score-identity audits pass',flush=True)


def evaluate():
    assert not (OUT/'result.json').exists(),'Preserve completed diagnosis'
    prepared=read(OUT/'prepared.json');audited=read(OUT/'audit.json')
    assert audited['status']=='pass' and audited['prepared']==art(OUT/'prepared.json')
    assert prepared['design']==DESIGN and prepared['sources'][str(Path(__file__).resolve().relative_to(ROOT))]==sha256(__file__)
    for path,digest in prepared['sources'].items():assert sha256(ROOT/path)==digest,path
    scores=dict(prepared['known_scores']);official={}
    for version in ADDITIONS:
        rec=prepared['additions'][str(version)];snapshot=read(verify(rec['snapshot']))
        live=read(ROOT/rec['live_manifest'])
        assert live['state']=='scored',f'v{version} has no official successful score yet'
        assert live['file']==snapshot['file'] and live['sha256']==snapshot['sha256']
        assert live['remote_sha256']==snapshot['sha256']
        outcome=live['official_result'];assert outcome['status']=='Succeeded' and outcome['used_pairs']==344841
        assert abs(live['score']-outcome['score'])<1e-10
        scores[str(version)]=float(outcome['score']);official[str(version)]=outcome
    live=transport.live.fetch_leaderboard()
    ours={r['filename']:r for r in live if r['teamName']==transport.live.TEAM}
    versions=prepared['versions'];s=np.array([scores[str(v)] for v in versions])
    for version,score in zip(versions,s):
        row=ours[f'unique-rose_v{version}.parquet']
        assert row['usedPairs']==344841 and abs(row['score']-score)<1e-8
    ranking=pd.read_parquet(verify(prepared['ranking']));held=pd.read_parquet(verify(prepared['cv']))
    columns=[f'v{v}' for v in versions];P=ranking[columns].to_numpy(float);C=held[columns].to_numpy(float)
    solution,raw=geometry.solve(P,s,versions,*DESIGN['bounds'],DESIGN['ridge_mse'])
    w=np.array([solution['weights'][str(v)] for v in versions])
    independent=float(w@(s*s)-w@np.mean(P*P,axis=0)+np.mean(raw*raw))
    np.testing.assert_allclose(np.sqrt(independent),solution['projected_rmse'],atol=1e-7,rtol=0)
    submitted=np.rint(np.maximum(raw,0));distance=float(np.sqrt(np.mean((submitted-raw)**2)))
    interval=solution['score_rounding_rmse_interval']
    conservative=[max(0.,interval[0]-distance),interval[1]+distance]
    best=int(versions[int(np.argmin(s))]);bestscore=float(s.min())
    pred=np.rint(np.maximum(C@w,0))
    frame=held[['mvt_id','month','day','airport','y']].copy()
    frame['control']=held[f'v{best}'];frame['pred']=pred
    months={m:{name:float(np.sqrt(np.mean((g[name]-g.y)**2))) for name in ['control','pred']}
        for m,g in frame.groupby('month')}
    pair=legacy.ts.paired_bootstrap(frame.day,frame.month,(frame.control-frame.y)**2,(frame.pred-frame.y)**2)
    frame.to_parquet(OUT/'diagnostic_cv.parquet',index=False)
    pd.DataFrame(dict(mvt_id=ranking.mvt_id,raw=raw,submitted=submitted)).to_parquet(OUT/'diagnostic_ranking.parquet',index=False)
    result=dict(design=DESIGN,prepared=art(OUT/'prepared.json'),solution=solution,official=official,
        checked_at=transport.now(),current_best_version=best,current_best_score=bestscore,
        cv=float(np.mean([r['pred'] for r in months.values()])),months=months,paired=pair,
        serialization_rms=distance,conservative_rmse_interval=conservative,
        clears_current_best_with_serialization_bound=conservative[1]<bestscore,
        independent_mse_identity=True,diagnostic_cv=art(OUT/'diagnostic_cv.parquet'),
        diagnostic_ranking=art(OUT/'diagnostic_ranking.parquet'),
        historical_cv_caveat=prepared['historical_cv_caveat'],eligible_for_upload=False,
        next='Review projected gain and CV; any selected file needs independent final arithmetic/schema/source audit')
    write_json(OUT/'result.json',result)
    print(dict(best=bestscore,projected=solution['projected_rmse'],conservative=conservative,cv=result['cv']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','audit','evaluate','check'])
    args=parser.parse_args()
    with threadpool_limits(1):
        if args.stage=='check':print(check_geometry())
        else:globals()[args.stage]()
