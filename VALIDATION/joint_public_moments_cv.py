"""Portable fixed joint-public-source screen; score only after inference ends."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import joint_public_moments as model

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'VALIDATION/runs/v3_20261011_joint_public_moments'
DESIGN=dict(settings=model.SETTINGS,parent='selected v77',
    prior='serialized v70 means, with retained one-pass off-block date assignment',
    primary='v77 plus0.5 times joint-minus-departure-only posterior means, within input-defined scope',
    retained='all other rows, original monthly quality gate, final matched/recovered bounds and rounding',
    calibration='all completed fold-specific source/noise/quantile parameters reused, no fits',
    limitations=['moment approximation to multimodal peers','same ordinary perturbation shared across overlapping cohorts',
                'v77 already uses public inputs; incremental transfer is an exploratory comparison',
                'extensively reused Jan/Jul development data'],
    no_new_model_fit=True,no_weight_scan=True,no_production=True,no_upload=True)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def art(path):return dict(path=str(Path(path).resolve().relative_to(ROOT)),sha256=sha(path))
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def verify(a):
    p=ROOT/a['path'];assert sha(p)==a['sha256'],p;return p


def prepare():
    import daily_main_refinements_cv as selected
    import daily_heavy_tail as heavy
    import taxi_count_sum_jan_jul as taxi
    assert not (OUT/'identity.json').exists(),'Preserve prepared screen'
    for p in [selected.OUT/'audit.json',heavy.OUT/'audit.json',taxi.OUT/'audit.json']:
        assert read(p)['status']=='pass'
    paths=[Path(__file__),Path(model.__file__),Path(model.taxi.__file__),Path(model.prior.__file__),
        Path(model.prior.base.__file__),ROOT/'VALIDATION/records.py',
        selected.OUT/'scored.parquet',selected.OUT/'result.json',selected.OUT/'audit.json',
        heavy.aligned.OUT/'rows2025.parquet',heavy.aligned.OUT/'scored.parquet',heavy.aligned.OUT/'result.json',
        heavy.OUT/'audit.json',taxi.OUT/'parameters.json',taxi.OUT/'audit.json',taxi.TAU,
        selected.quality.OUT/'scored.parquet',taxi.daily.CACHE,taxi.daily.SOURCE/'daily_runway.parquet',
        model.prior.base.SOURCE/'all_pre_departure_delays_2025.csv']
    raw=pd.read_parquet(heavy.aligned.OUT/'rows2025.parquet',filters=[('airport','==','LIRF')]).set_index('mvt_id')
    f=pd.read_parquet(heavy.aligned.OUT/'scored.parquet',columns=['mvt_id','month','parent','assigned_day','new_scope'])
    f=f[f.mvt_id.isin(raw.index)].reset_index(drop=True);r=raw.loc[f.mvt_id]
    np.testing.assert_array_equal(f.month,r.month)
    assigned=heavy.aligned.dates(r.takeoff_epoch,f.parent)
    np.testing.assert_array_equal(assigned,f.assigned_day)
    eligible=taxi.daily.rows().drop(columns='y').set_index('mvt_id').loc[f.mvt_id]
    quality=pd.read_parquet(selected.quality.OUT/'scored.parquet',columns=['mvt_id','quality_eligible']).set_index('mvt_id').loc[f.mvt_id]
    inputs=pd.DataFrame(dict(mvt_id=f.mvt_id,month=f.month,gap=r.gap.to_numpy(),prior=f.parent,
        eligible=eligible.eligible.to_numpy(),quality=quality.quality_eligible.to_numpy(),
        runway=eligible.runway.to_numpy(),local_day=eligible.local_day.to_numpy(),
        assigned_day=assigned,fltrule=r.fltrule.to_numpy(),delay_scope=f.new_scope))
    assert inputs.mvt_id.is_unique and len(inputs)==26528
    assert 'y' not in inputs and 'true_day' not in inputs
    st=taxi.daily.public();st=st[st.airport.eq('LIRF')].drop(columns='airport')
    sd=model.prior.base.public(2025);sd=sd[sd.airport.eq('LIRF')].drop(columns='airport')
    old=read(heavy.aligned.OUT/'result.json')['parameters'];tp=read(taxi.OUT/'parameters.json');sp=read(taxi.TAU)['parameters']
    parameters={m:dict(taxi=tp[m],delay=old[m]['offblock']['LIRF'],support=sp[m]['LIRF']) for m in old}
    for m,par in parameters.items():assert par['taxi']['allowed']==taxi.daily.base.FOLDS[m]
    OUT.mkdir(parents=True,exist_ok=True)
    inputs.to_parquet(OUT/'inputs.parquet',index=False);st.to_parquet(OUT/'taxi_source.parquet',index=False)
    sd.to_parquet(OUT/'delay_source.parquet',index=False);write(OUT/'parameters.json',parameters)
    identity=dict(design=DESIGN,sources=[art(p) for p in paths],source=art(__file__),model=art(model.__file__),
        helpers=[art(p) for p in [Path(model.taxi.__file__),Path(model.prior.__file__),Path(model.prior.base.__file__),ROOT/'VALIDATION/records.py']],
        inputs=[art(OUT/p) for p in ['inputs.parquet','taxi_source.parquet','delay_source.parquet','parameters.json']],
        checks=model.checks(),rows=len(inputs),dates_replayed_exact=True,labels_absent_from_inference_packet=True)
    write(OUT/'identity.json',identity);print(dict(rows=len(inputs),checks=identity['checks']),flush=True)


def guard():
    identity=read(OUT/'identity.json');assert identity['design']==DESIGN
    for a in [identity['source'],identity['model'],*identity['helpers'],*identity['inputs']]:verify(a)
    return identity


def infer():
    identity=guard();assert not (OUT/'inference.json').exists(),'Preserve inference'
    assert not (OUT/'predictions.parquet').exists(),'Unrecorded output needs inspection'
    inputs=pd.read_parquet(OUT/'inputs.parquet');st=pd.read_parquet(OUT/'taxi_source.parquet');sd=pd.read_parquet(OUT/'delay_source.parquet')
    prm=read(OUT/'parameters.json');frames=[];cells=[];checks=[];started=time.monotonic()
    for month,p in prm.items():
        rows=inputs[inputs.month.eq(month)].reset_index(drop=True)
        result,cell=model.apply(rows,st,sd,p['taxi'],p['delay'],p['support'])
        # Labels cannot enter through extra columns, and row order cannot alter
        # memberships or messages. This is cheap relative to a model fit.
        shuffled=rows.sample(frac=1,random_state=82).assign(y=1e12,true_day='1900-01-01')
        replay,other=model.apply(shuffled,st,sd,p['taxi'],p['delay'],p['support'])
        left=result.set_index('mvt_id').sort_index();right=replay.set_index('mvt_id').sort_index()
        np.testing.assert_array_equal(left.touched,right.touched)
        np.testing.assert_allclose(left[['dep_only','joint','increment']],right[['dep_only','joint','increment']],atol=1e-7,rtol=0)
        checks.append(dict(month=month,rows=len(rows),touched=int(result.touched.sum()),
            permutation_max_difference=float(np.max(abs(left.increment-right.increment))),label_poison_invariant=True))
        frames.append(result.assign(month=month));cells.append(cell.assign(month=month))
        print(checks[-1],flush=True)
    pd.concat(frames,ignore_index=True).to_parquet(OUT/'predictions.parquet',index=False)
    pd.concat(cells,ignore_index=True).to_parquet(OUT/'cells.parquet',index=False)
    write(OUT/'inference.json',dict(identity=art(OUT/'identity.json'),host=socket.gethostname(),
        seconds=time.monotonic()-started,checks=checks,artifacts=[art(OUT/p) for p in ['predictions.parquet','cells.parquet']]))


def score():
    import daily_main_refinements_cv as selected
    import tuning_seasonal as ts
    guard();receipt=read(OUT/'inference.json');assert receipt['identity']==art(OUT/'identity.json')
    for a in receipt['artifacts']:verify(a)
    assert not (OUT/'result.json').exists(),'Preserve completed score'
    f=pd.read_parquet(selected.OUT/'scored.parquet',columns=['mvt_id','month','day','y','combined']).rename(columns={'combined':'control'})
    p=pd.read_parquet(OUT/'predictions.parquet').set_index('mvt_id').reindex(f.mvt_id)
    f['increment']=p.increment.fillna(0).to_numpy();f['touched']=p.touched.fillna(False).to_numpy(bool)
    raw=pd.read_parquet(selected.daily.base.OUT/'rows2025.parquet').set_index('mvt_id').loc[f.mvt_id].reset_index()
    matches=pd.read_parquet(selected.daily.constraints.exclusion.OUT/'matches2025.parquet')
    np.testing.assert_array_equal(selected.daily.constrain(raw,f.mvt_id.to_numpy(),f.month,f.control.to_numpy(),matches),f.control)
    candidate=selected.daily.constrain(raw,f.mvt_id.to_numpy(),f.month,f.control.to_numpy()+.5*f.increment.to_numpy(),matches)
    np.testing.assert_array_equal(candidate[~f.touched],f.control.to_numpy()[~f.touched])
    f['candidate']=candidate;f.to_parquet(OUT/'scored.parquet',index=False)
    scores={m:{c:float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in ['control','candidate']} for m,g in f.groupby('month')}
    paired=ts.paired_bootstrap(f.day,f.month,(f.control-f.y)**2,(f.candidate-f.y)**2)
    result=dict(design=DESIGN,identity=art(OUT/'identity.json'),inference=art(OUT/'inference.json'),
        scored=art(OUT/'scored.parquet'),scores=scores,cv={c:float(np.mean([v[c] for v in scores.values()])) for c in ['control','candidate']},
        paired=paired,touched=int(f.touched.sum()),changed=int(f.candidate.ne(f.control).sum()),
        original_control_and_unchanged_rows_exact=True)
    write(OUT/'result.json',result);print(dict(cv=result['cv'],paired=paired),flush=True)


def audit():
    import duckdb
    import daily_main_refinements_cv as selected
    guard();result=read(OUT/'result.json');verify(result['scored'])
    with duckdb.connect(config={'threads':1}) as con:
        for name,path in [('p',OUT/'scored.parquet'),('r',selected.daily.base.OUT/'rows2025.parquet'),
            ('c',selected.OUT/'scored.parquet'),('v',OUT/'predictions.parquet'),('i',OUT/'inputs.parquet')]:
            con.read_parquet(str(path)).create_view(name)
        n,u,bad=con.execute('''select count(*),count(distinct p.mvt_id),sum(case when
            p.y is distinct from r.y or p.month is distinct from r.month or p.day is distinct from r.day
            or p.control is distinct from c.combined or p.increment!=coalesce(v.increment,0)
            or (not p.touched and p.candidate!=p.control) or (p.touched and r.airport!='LIRF')
            then 1 else 0 end) from p join r using(mvt_id) join c using(mvt_id) left join v using(mvt_id)''').fetchone()
        assert n==u==344419 and bad==0
        wrong=con.execute('''select count(*) from v join i using(mvt_id) where v.increment!=v.joint-v.dep_only
            or v.prior!=i.prior or (v.touched and (not i.eligible or not i.quality or not i.delay_scope or i.fltrule='V'
            or i.gap<=3600 or i.gap>=50400))''').fetchone()[0];assert wrong==0
        for name in result['cv']:
            scores=con.execute(f'select month,sqrt(avg(pow({name}-y,2))) from p group by month').fetchall()
            for month,value in scores:np.testing.assert_allclose(value,result['scores'][month][name],atol=1e-8,rtol=0)
            np.testing.assert_allclose(np.mean([v for _,v in scores]),result['cv'][name],atol=1e-8,rtol=0)
    f=pd.read_parquet(OUT/'scored.parquet');raw=pd.read_parquet(selected.daily.base.OUT/'rows2025.parquet').set_index('mvt_id').loc[f.mvt_id].reset_index()
    matches=pd.read_parquet(selected.daily.constraints.exclusion.OUT/'matches2025.parquet')
    np.testing.assert_array_equal(selected.daily.constrain(raw,f.mvt_id.to_numpy(),f.month,
        f.control.to_numpy()+.5*f.increment.to_numpy(),matches),f.candidate)
    write(OUT/'audit.json',dict(status='pass',identity=art(OUT/'identity.json'),result=art(OUT/'result.json'),
        rows=n,independent_sql_truth_control_increment_scope_scores=True,final_constraints_replayed=True,source=art(__file__)))
    print('All-row independent SQL and final-constraint audit passed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','infer','score','audit']);a=p.parse_args()
    with threadpool_limits(1):globals()[a.stage]()
