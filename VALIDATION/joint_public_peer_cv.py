"""Fixed explicit-peer Jan/Jul comparison against the selected v78 update."""
import argparse
from pathlib import Path
import shutil
import socket
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import joint_public_moments_cv as parent
import joint_public_discrete_peer as model

ROOT=parent.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261011_joint_public_peer'
art,read,write,verify,sha=parent.art,parent.read,parent.write,parent.verify,parent.sha
DESIGN=dict(settings=model.SETTINGS,parent='exact v78',
    change='one largest-variance long peer retains its full discrete support in each target message',
    primary='replay v77 plus0.5 times refined joint-minus-departure increment versus exact v78',
    retained='same source calibration, priors, source memberships, scope, weight and final constraints',
    limitation='one explicit peer; all remaining peer moments remain approximate; reused development data',
    no_taxi_fit=True,no_weight_scan=True,no_production=True,no_upload=True)


def prepare():
    parent.guard();assert not (OUT/'identity.json').exists()
    assert read(parent.OUT/'audit.json')['status']=='pass'
    assert read(parent.OUT/'input_audit.json')['status']=='pass'
    checks=model.checks();OUT.mkdir(parents=True,exist_ok=True)
    inputs=[]
    for name in ['inputs.parquet','taxi_source.parquet','delay_source.parquet','parameters.json']:
        shutil.copyfile(parent.OUT/name,OUT/name);inputs.append(art(OUT/name))
        assert sha(OUT/name)==sha(parent.OUT/name)
    old=pd.read_parquet(parent.OUT/'predictions.parquet',columns=['mvt_id','month','dep_only','joint','increment','touched'])
    old.to_parquet(OUT/'moment_control.parquet',index=False);inputs.append(art(OUT/'moment_control.parquet'))
    sources=[Path(__file__),Path(model.__file__),Path(parent.__file__),Path(model.base.__file__),
             Path(model.base.prior.__file__),Path(model.base.prior.base.__file__),Path(model.base.taxi.__file__),
             ROOT/'VALIDATION/records.py']
    write(OUT/'identity.json',dict(design=DESIGN,sources=[art(p) for p in sources],inputs=inputs,checks=checks,
        parent_identity=art(parent.OUT/'identity.json'),parent_result=art(parent.OUT/'result.json'),
        parent_audit=art(parent.OUT/'audit.json'),parent_scored=art(parent.OUT/'scored.parquet'),
        input_source_audit=art(parent.OUT/'input_audit.json'),no_labels_or_true_offblock_dates_in_packet=True))
    print(dict(prepared=True,checks=checks),flush=True)


def guard():
    identity=read(OUT/'identity.json');assert identity['design']==DESIGN
    for a in [*identity['sources'],*identity['inputs']]:verify(a)
    return identity


def infer():
    guard();assert not (OUT/'inference.json').exists() and not (OUT/'predictions.parquet').exists()
    inputs=pd.read_parquet(OUT/'inputs.parquet');st=pd.read_parquet(OUT/'taxi_source.parquet');sd=pd.read_parquet(OUT/'delay_source.parquet')
    prm=read(OUT/'parameters.json');old=pd.read_parquet(OUT/'moment_control.parquet').set_index('mvt_id')
    frames=[];cells=[];checks=[];started=time.monotonic()
    for month,p in prm.items():
        rows=inputs[inputs.month.eq(month)].reset_index(drop=True)
        result,cell=model.apply(rows,st,sd,p['taxi'],p['delay'],p['support'])
        baseline=old.loc[result.mvt_id]
        np.testing.assert_array_equal(result.touched,baseline.touched)
        np.testing.assert_allclose(result.moment_dep,baseline.dep_only,atol=1e-7,rtol=0)
        np.testing.assert_allclose(result.moment_joint,baseline.joint,atol=1e-7,rtol=0)
        shuffled=rows.sample(frac=1,random_state=82).assign(y=1e12,true_day='1900-01-01')
        replay,_=model.apply(shuffled,st,sd,p['taxi'],p['delay'],p['support'])
        left=result.set_index('mvt_id').sort_index();right=replay.set_index('mvt_id').sort_index()
        np.testing.assert_array_equal(left[['touched','peer_id']],right[['touched','peer_id']])
        np.testing.assert_allclose(left[['dep_only','joint','increment']],right[['dep_only','joint','increment']],atol=1e-7,rtol=0)
        checks.append(dict(month=month,rows=len(rows),touched=int(result.touched.sum()),explicit_peer_rows=int(result.peer_id.ge(0).sum()),
            original_moment_controls_exact=True,label_poison_invariant=True,
            permutation_max_difference=float(np.max(abs(left.increment-right.increment)))))
        frames.append(result.assign(month=month));cells.append(cell.assign(month=month));print(checks[-1],flush=True)
    pd.concat(frames,ignore_index=True).to_parquet(OUT/'predictions.parquet',index=False)
    pd.concat(cells,ignore_index=True).to_parquet(OUT/'cells.parquet',index=False)
    write(OUT/'inference.json',dict(identity=art(OUT/'identity.json'),host=socket.gethostname(),seconds=time.monotonic()-started,
        checks=checks,artifacts=[art(OUT/p) for p in ['predictions.parquet','cells.parquet']]))


def score():
    import daily_main_refinements_cv as selected
    import tuning_seasonal as ts
    identity=guard();receipt=read(OUT/'inference.json');assert receipt['identity']==art(OUT/'identity.json')
    for a in receipt['artifacts']:verify(a)
    assert not (OUT/'result.json').exists()
    f=pd.read_parquet(verify(identity['parent_scored'])).rename(columns={'control':'predecessor','candidate':'control','increment':'old_increment'})
    p=pd.read_parquet(OUT/'predictions.parquet').set_index('mvt_id').reindex(f.mvt_id)
    f['increment']=p.increment.fillna(0).to_numpy()
    np.testing.assert_array_equal(f.touched,p.touched.fillna(False).to_numpy(bool))
    raw=pd.read_parquet(selected.daily.base.OUT/'rows2025.parquet').set_index('mvt_id').loc[f.mvt_id].reset_index()
    matches=pd.read_parquet(selected.daily.constraints.exclusion.OUT/'matches2025.parquet')
    ids=f.mvt_id.to_numpy()
    np.testing.assert_array_equal(selected.daily.constrain(raw,ids,f.month,f.predecessor.to_numpy()+.5*f.old_increment.to_numpy(),matches),f.control)
    f['candidate']=selected.daily.constrain(raw,ids,f.month,f.predecessor.to_numpy()+.5*f.increment.to_numpy(),matches)
    np.testing.assert_array_equal(f.loc[~f.touched,'candidate'],f.loc[~f.touched,'control'])
    f.to_parquet(OUT/'scored.parquet',index=False)
    scores={m:{c:float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in ['control','candidate']} for m,g in f.groupby('month')}
    paired=ts.paired_bootstrap(f.day,f.month,(f.control-f.y)**2,(f.candidate-f.y)**2)
    result=dict(design=DESIGN,identity=art(OUT/'identity.json'),inference=art(OUT/'inference.json'),scored=art(OUT/'scored.parquet'),
        scores=scores,cv={c:float(np.mean([v[c] for v in scores.values()])) for c in ['control','candidate']},paired=paired,
        touched=int(f.touched.sum()),changed=int(f.candidate.ne(f.control).sum()),parent_and_all_untouched_rows_exact=True)
    write(OUT/'result.json',result);print(dict(cv=result['cv'],paired=paired),flush=True)


def audit():
    import duckdb
    import daily_main_refinements_cv as selected
    guard();result=read(OUT/'result.json');verify(result['scored'])
    with duckdb.connect(config={'threads':1}) as con:
        for name,path in [('p',OUT/'scored.parquet'),('r',selected.daily.base.OUT/'rows2025.parquet'),
            ('old',parent.OUT/'scored.parquet'),('v',OUT/'predictions.parquet'),('i',OUT/'inputs.parquet')]:
            con.read_parquet(str(path)).create_view(name)
        n,u,bad=con.execute('''select count(*),count(distinct p.mvt_id),sum(case when
            p.y is distinct from r.y or p.month is distinct from r.month or p.day is distinct from r.day
            or p.control is distinct from old.candidate or p.predecessor is distinct from old.control
            or p.old_increment is distinct from old.increment or p.increment!=coalesce(v.increment,0)
            or p.touched is distinct from old.touched or (not p.touched and p.candidate!=p.control)
            then 1 else 0 end) from p join r using(mvt_id) join old using(mvt_id) left join v using(mvt_id)''').fetchone()
        assert n==u==344419 and bad==0
        bad=con.execute('''select count(*) from v join i using(mvt_id) where v.increment!=v.joint-v.dep_only
            or v.prior!=i.prior or (v.peer_id>=0 and (not v.touched or v.peer_id=v.mvt_id))''').fetchone()[0]
        assert bad==0
        for name in result['cv']:
            scores=con.execute(f'select month,sqrt(avg(pow({name}-y,2))) from p group by month').fetchall()
            for month,value in scores:np.testing.assert_allclose(value,result['scores'][month][name],atol=1e-8,rtol=0)
            np.testing.assert_allclose(np.mean([v for _,v in scores]),result['cv'][name],atol=1e-8,rtol=0)
    f=pd.read_parquet(OUT/'scored.parquet');raw=pd.read_parquet(selected.daily.base.OUT/'rows2025.parquet').set_index('mvt_id').loc[f.mvt_id].reset_index()
    matches=pd.read_parquet(selected.daily.constraints.exclusion.OUT/'matches2025.parquet')
    np.testing.assert_array_equal(selected.daily.constrain(raw,f.mvt_id.to_numpy(),f.month,
        f.predecessor.to_numpy()+.5*f.increment.to_numpy(),matches),f.candidate)
    write(OUT/'audit.json',dict(status='pass',identity=art(OUT/'identity.json'),result=art(OUT/'result.json'),rows=n,
        independent_sql_raw_truth_control_increment_scope_and_scores=True,final_constraints_replayed=True,
        source=art(__file__),message_audit_pending=True))
    print(dict(score_audit='pass',rows=n),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','infer','score','audit']);a=p.parse_args()
    with threadpool_limits(1):globals()[a.stage]()
