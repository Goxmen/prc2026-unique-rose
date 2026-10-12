"""Independent read-only SQL audit of the new basis that excludes v75.

Does not solve weights, package a candidate, mutate a manifest or upload.
Requires a live leaderboard snapshot taken after the completed diagnosis.
Reconstructs geometry, integer serialization and Jan/Jul scores independently.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'VALIDATION/runs/v3_20261012_target205_geometry'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def artifact(path):
    path=Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)),sha256=sha(path))


def verify(record):
    path=ROOT/record['path']
    if sha(path)!=record['sha256']:raise ValueError('Artifact changed: '+str(path))
    return path


def expression(versions,weights,alias='r'):
    assert len(versions)==len(weights) and 64 in versions
    assert all(isinstance(v,int) and v>0 for v in versions)
    assert np.isfinite(weights).all()
    return f'{alias}.v64'+''.join(f'+({float(w):.17g})*({alias}.v{v}-{alias}.v64)'
        for v,w in zip(versions,weights) if v!=64)


def sql_geometry(con,versions,weights,scores):
    """Use only SQL population moments; never reuse the fitted Gram matrix."""
    at=versions.index(64)
    effective=np.array(weights,float,copy=True)
    effective[at]=1-float(np.sum(np.delete(effective,at)))
    raw=expression(versions,effective)
    moments=','.join(f'avg(pow(r.v{v}-r.v64,2)),avg(pow(r.v{v},2))' for v in versions)
    m=np.asarray(con.execute(f'select {moments} from r').fetchone()).reshape(-1,2)
    diagonal,norm=m[:,0],m[:,1]
    values=con.execute(f'''with joined as (
        select r.mvt_id,r.v64,o.raw,o.submitted,({raw}) independent
        from r full join o using(mvt_id))
        select count(*),count(distinct mvt_id),
        sum(case when mvt_id is null or raw is null or submitted is null or independent is null
            or not isfinite(raw) or not isfinite(submitted) or not isfinite(independent)
            or submitted<0 or submitted!=round_even(submitted,0)
            or submitted is distinct from round_even(greatest(raw,0),0) then 1 else 0 end),
        max(abs(independent-raw)),avg(pow(independent-v64,2)),avg(pow(independent,2)),
        sqrt(avg(pow(submitted-independent,2))),max(abs(submitted-greatest(independent,0)))
        from joined''').fetchone()
    count,unique,bad,maxdiff,second,rawnorm,distance,round_distance=values
    assert count==unique and bad==0 and maxdiff<1e-7 and round_distance<=.5000001
    mse0=scores[at]**2
    mse=float(mse0+effective@(scores**2-mse0-diagonal)+second)
    expanded=float(effective@(scores**2-norm)+rawnorm)
    assert mse>=0 and np.isfinite([mse,expanded,distance]).all()
    np.testing.assert_allclose(mse,expanded,atol=1e-5,rtol=0)
    scale=float(abs(mse0)+np.abs(effective)@(np.abs(scores**2)+abs(mse0)+np.abs(diagonal))+abs(second))
    numerical=float(64*np.finfo(float).eps*count*max(scale,1.))
    bounds={}
    for name,epsilon in [('nearest_four_decimals',.00005),('conservative_last_decimal',.0001)]:
        radius=float(np.abs(effective)@(2*np.abs(scores)*epsilon+epsilon**2))
        raw_interval=[float(np.sqrt(max(0,mse-radius-numerical))),float(np.sqrt(mse+radius+numerical))]
        bounds[name]=dict(score_halfwidth=epsilon,score_mse_radius=radius,
            raw_rmse_interval=raw_interval,serialized_rmse_interval=[max(0.,raw_interval[0]-distance),raw_interval[1]+distance])
    legacy_radius=float(np.abs(weights)@(np.abs(scores)*1e-4+2.5e-9))
    return dict(rows=count,unique_ids=unique,sql_projected_rmse=float(np.sqrt(mse)),
        reference_mse=mse,expanded_mse=expanded,maximum_raw_reconstruction_difference=float(maxdiff),
        serialization_rms=float(distance),maximum_rounding_distance=float(round_distance),
        numerical_mse_allowance=numerical,coefficient_l1=float(np.abs(effective).sum()),
        effective_reference_weight=float(effective[at]),bounds=bounds,
        legacy_interval_without_numerical_allowance=[max(0.,float(np.sqrt(max(0,mse-legacy_radius)))-distance),float(np.sqrt(mse+legacy_radius))+distance])


def audit(snapshot):
    result_path=OUT/'result.json';result=read(result_path)
    prepared_path=verify(result['prepared']);prepared=read(prepared_path)
    old=read(verify(prepared['reused_basis']))
    base_audit=read(OUT/'audit.json')
    assert base_audit['status']=='pass' and base_audit['prepared']==artifact(prepared_path)
    assert prepared['design']['target']==205. and prepared['removed_unsubmitted_versions']==[75]
    versions=prepared['versions']
    assert versions==[v for v in old['versions'] if v!=75] and len(versions)==len(set(versions))==51
    assert set(map(int,prepared['additions']))=={70,72,74,76,77,78}
    assert '75' not in prepared['known_scores'] and '75' not in result['official']
    assert all(prepared['sources'][p]==digest for p,digest in old['sources'].items())
    source=ROOT/'VALIDATION/target205_geometry.py'
    assert prepared['sources'][str(source.relative_to(ROOT))]==sha(source)
    solution=result['solution']
    assert set(solution['weights'])=={str(v) for v in versions}
    weights=np.array([solution['weights'][str(v)] for v in versions],float)
    assert np.isfinite(weights).all() and abs(weights.sum()-1)<1e-8
    assert solution['constraints']==dict(lower=-3.,upper=4.,ridge_mse=1.)
    assert ((weights>=-3.-1e-8)&(weights<=4.+1e-8)).all()
    scores=dict(prepared['known_scores'])
    for v,addition in prepared['additions'].items():
        frozen=read(verify(addition['snapshot']));live=read(ROOT/addition['live_manifest'])
        assert live['state']=='scored' and live['file']==frozen['file']
        assert live['sha256']==live['remote_sha256']==frozen['sha256']==sha(verify(addition['ranking']))
        official=live['official_result']
        assert official==result['official'][v] and official['status']=='Succeeded' and official['used_pairs']==344841
        assert live['score']==official['score']
        scores[v]=float(official['score'])
    live=read(snapshot)
    assert live['snapshotSource']=='live'
    assert datetime.fromisoformat(live['updatedAt'])>=datetime.fromisoformat(result['checked_at'])
    successful={r['version']:r for r in live['history'] if r['status']=='Succeeded' and r['usedPairs']==344841}
    for v in versions:
        assert v in successful and successful[v]['file']==f'unique-rose_v{v}.parquet'
        assert abs(successful[v]['score']-scores[str(v)])<1e-8
    official_best=min(r['score'] for r in successful.values())
    score_array=np.array([scores[str(v)] for v in versions],float)
    best_version=int(versions[int(np.argmin(score_array))])
    assert result['current_best_version']==best_version and result['current_best_score']==float(score_array.min())
    with duckdb.connect(config={'threads':1,'memory_limit':'2GB'}) as con:
        for name,path in [('r',verify(prepared['ranking'])),('o',verify(result['diagnostic_ranking'])),
            ('h',verify(prepared['cv'])),('c',verify(result['diagnostic_cv'])),
            ('t',ROOT/'VALIDATION/runs/v3_20261011_daily_delay/rows2025.parquet')]:
            con.read_parquet(str(path)).create_view(name)
        assert [r[0] for r in con.execute('describe r').fetchall()]==['mvt_id']+[f'v{v}' for v in versions]
        for name,expected in [('r',344841),('o',344841),('h',344419),('c',344419)]:
            assert con.execute(f'select count(*),count(distinct mvt_id) from {name}').fetchone()==(expected,expected)
        geometry=sql_geometry(con,versions,weights,score_array)
        np.testing.assert_allclose(geometry['sql_projected_rmse'],solution['projected_rmse'],atol=1e-7,rtol=0)
        np.testing.assert_allclose(geometry['serialization_rms'],result['serialization_rms'],atol=1e-8,rtol=0)
        np.testing.assert_allclose(geometry['legacy_interval_without_numerical_allowance'],result['conservative_rmse_interval'],atol=1e-6,rtol=0)
        raw=expression(versions,weights,'h')
        count,unique,bad,error=con.execute(f'''select count(*),count(distinct c.mvt_id),
            sum(case when c.mvt_id is null or h.mvt_id is null or t.mvt_id is null
                or c.y is distinct from h.y or c.y is distinct from t.y
                or c.month is distinct from h.month or c.month is distinct from t.month
                or c.day is distinct from h.day or c.day is distinct from t.day
                or c.control is distinct from h.v{best_version}
                or c.pred is null or not isfinite(c.pred) or c.pred<0 or c.pred!=round_even(c.pred,0)
                then 1 else 0 end),max(abs(c.pred-greatest(({raw}),0)))
            from c full join h using(mvt_id) left join t using(mvt_id)''').fetchone()
        assert count==unique==344419 and bad==0 and error<=.5000001
        months={m:dict(control=a,pred=b) for m,a,b in con.execute(
            'select month,sqrt(avg(pow(control-y,2))),sqrt(avg(pow(pred-y,2))) from c group by month').fetchall()}
        assert set(months)=={'2025-01','2025-07'}
        for month,value in months.items():
            for arm,score in value.items():np.testing.assert_allclose(score,result['months'][month][arm],atol=1e-8,rtol=0)
        cv=float(np.mean([value['pred'] for value in months.values()]))
        np.testing.assert_allclose(cv,result['cv'],atol=1e-8,rtol=0)
    upper=geometry['bounds']['conservative_last_decimal']['serialized_rmse_interval'][1]
    report=dict(status='pass',source=artifact(Path(__file__)),result=artifact(result_path),
        prepared=artifact(prepared_path),live_snapshot=artifact(snapshot),versions=versions,
        no_v75_score_or_vector=True,v75_ancestry_preserved=True,all_active_scores_live_verified=True,
        geometry=geometry,cv=cv,months=months,official_best=official_best,
        cautious_conservative_gain=official_best-upper,target205_bound_cleared=upper<=205.,
        scope='Independent read-only arithmetic/source/population audit, not independent validation or upload approval')
    path=OUT/'independent_solve_audit.json'
    assert not path.exists(),'Preserve completed independent audit'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='pass',projected=geometry['sql_projected_rmse'],
        cautious_upper=upper,official_best=official_best,conservative_gain=official_best-upper,cv=cv),indent=2),flush=True)


def check():
    rng=np.random.default_rng(123771)
    y=rng.uniform(0,100,257)
    P=np.rint(np.maximum(y[:,None]+rng.normal(0,90,(len(y),4)),0))
    versions=[77,64,76,78];weights=np.array([1.4,-.8,.7,-.3])
    scores=np.round(np.sqrt(np.mean((P-y[:,None])**2,axis=0)),4)
    raw=P@weights
    r=pd.DataFrame(P,columns=[f'v{v}' for v in versions]);r.insert(0,'mvt_id',np.arange(len(y)))
    o=pd.DataFrame(dict(mvt_id=r.mvt_id,raw=raw,submitted=np.rint(np.maximum(raw,0))))
    assert (raw<0).any()
    with duckdb.connect(config={'threads':1}) as con:
        con.register('r',r);con.register('o',o)
        result=sql_geometry(con,versions,weights,scores)
        actual=float(np.sqrt(np.mean((o.submitted-y)**2)))
        for bound in result['bounds'].values():
            lo,hi=bound['serialized_rmse_interval'];assert lo<=actual<=hi
        broken=o.copy();broken.loc[0,'submitted']+=3
        con.unregister('o');con.register('o',broken)
        try:sql_geometry(con,versions,weights,scores)
        except AssertionError:pass
        else:raise AssertionError('Corrupt serialization accepted')
    print('Synthetic signed/reordered basis, active clipping, independent MSE and score bounds pass')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['check','audit']);parser.add_argument('--snapshot',type=Path)
    args=parser.parse_args()
    with threadpool_limits(1):
        if args.stage=='check':check()
        else:audit(args.snapshot.resolve())
