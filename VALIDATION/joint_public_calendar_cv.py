"""Fixed Jan/Jul screen of uncertain UTC-day membership against exact v78."""
import argparse
from pathlib import Path
import socket
import time

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import joint_public_moments_cv as parent
import joint_public_calendar as model

ROOT=parent.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261011_joint_public_calendar'
art,read,write,verify=parent.art,parent.read,parent.write,parent.verify
DESIGN=dict(settings=model.SETTINGS,parent='exact v78',
    primary='replace fixed-day joint-minus-departure increment with uncertain-day increment, same half strength',
    scope='original touched Rome rows, all candidate days supported and interior; all other increments unchanged',
    prior='same serialized v70; same quantile/copy support; no public-input model enters prior',
    calibration='all original allowed-training parameters reused exactly',
    limitations=['Gaussian peer approximation','same day-specific calibration with expected counts',
        'one-day edge exclusion may miss extreme multi-day peers','extensively reused development months'],
    no_model_fit=True,no_weight_scan=True,no_production=True,no_upload=True)


def prepare():
    import daily_heavy_tail as heavy
    parent.guard();assert read(parent.OUT/'audit.json')['status']=='pass'
    assert not (OUT/'identity.json').exists()
    inputs=pd.read_parquet(parent.OUT/'inputs.parquet')
    raw_path=heavy.aligned.OUT/'rows2025.parquet'
    raw=pd.read_parquet(raw_path,columns=['mvt_id','takeoff_epoch']).set_index('mvt_id').loc[inputs.mvt_id]
    inputs['takeoff_epoch']=raw.takeoff_epoch.to_numpy()
    assert np.isfinite(inputs.takeoff_epoch).all() and len(inputs)==26528 and inputs.mvt_id.is_unique
    OUT.mkdir(parents=True,exist_ok=True);inputs.to_parquet(OUT/'inputs.parquet',index=False)
    for name in ['taxi_source.parquet','delay_source.parquet','parameters.json']:
        (OUT/name).write_bytes((parent.OUT/name).read_bytes())
    old=pd.read_parquet(parent.OUT/'predictions.parquet')
    old.to_parquet(OUT/'control.parquet',index=False)
    with duckdb.connect(config={'threads':1}) as con:
        con.read_parquet(str(OUT/'inputs.parquet')).create_view('i')
        con.read_parquet(str(raw_path)).create_view('r')
        n,u,bad=con.execute('''select count(*),count(distinct i.mvt_id),sum(cast(
            i.takeoff_epoch is distinct from r.takeoff_epoch or r.airport!='LIRF' as int))
            from i join r using(mvt_id)''').fetchone()
        assert n==u==26528 and bad==0
    original=pd.read_parquet(parent.OUT/'inputs.parquet')
    pd.testing.assert_frame_equal(inputs.drop(columns='takeoff_epoch'),original,check_exact=True)
    assert not {'y','true_day','BLOCK_TIME_UTC_mvt'}.intersection(inputs.columns)
    sources=[parent.OUT/'identity.json',parent.OUT/'input_audit.json',parent.OUT/'audit.json',parent.OUT/'scored.parquet',
        parent.OUT/'predictions.parquet',parent.OUT/'result.json',raw_path]
    helpers=[Path(__file__),Path(model.__file__),Path(model.base.__file__),Path(parent.__file__),
        Path(model.base.prior.__file__),Path(model.base.prior.base.__file__),Path(model.base.taxi.__file__),ROOT/'VALIDATION/records.py']
    write(OUT/'identity.json',dict(design=DESIGN,sources=[art(p) for p in sources],helpers=[art(p) for p in helpers],
        inputs=[art(OUT/n) for n in ['inputs.parquet','taxi_source.parquet','delay_source.parquet','parameters.json','control.parquet']],
        checks=model.checks(),input_audit=dict(rows=n,all_previous_inputs_exact=True,
            new_epoch_column_independent_sql_exact=True,all_parameters_and_sources_byte_identical=True,labels_absent=True)))
    print(dict(rows=n,checks=read(OUT/'identity.json')['checks']),flush=True)


def guard():
    identity=read(OUT/'identity.json');assert identity['design']==DESIGN
    for a in [*identity['helpers'],*identity['inputs']]:verify(a)
    return identity


def infer():
    guard();assert not (OUT/'inference.json').exists() and not (OUT/'predictions.parquet').exists()
    inputs=pd.read_parquet(OUT/'inputs.parquet');ts=pd.read_parquet(OUT/'taxi_source.parquet');ds=pd.read_parquet(OUT/'delay_source.parquet')
    prm=read(OUT/'parameters.json');control=pd.read_parquet(OUT/'control.parquet').set_index('mvt_id')
    frames=[];cells=[];checks=[];started=time.monotonic()
    for month,rows in inputs.groupby('month'):
        rows=rows.reset_index(drop=True);p=prm[month]
        result,cell=model.apply(rows,ts,ds,p['taxi'],p['delay'],p['support'])
        expected=control.loc[result.mvt_id]
        np.testing.assert_array_equal(result.touched,expected.touched)
        for col in ['dep_only','joint','increment']:
            np.testing.assert_allclose(result['old_'+col],expected[col],atol=1e-7,rtol=0)
        shuffled=rows.sample(frac=1,random_state=83).assign(y=1e12,true_day='1900-01-01')
        other,_=model.apply(shuffled,ts,ds,p['taxi'],p['delay'],p['support'])
        left=result.set_index('mvt_id').sort_index();right=other.set_index('mvt_id').sort_index()
        np.testing.assert_array_equal(left.calendar_touched,right.calendar_touched)
        np.testing.assert_allclose(left[['dep_only','joint','increment']],right[['dep_only','joint','increment']],atol=1e-7,rtol=0)
        item=dict(month=month,rows=len(rows),old_touched=int(result.touched.sum()),
            calendar_touched=int(result.calendar_touched.sum()),multi_day_targets=int(result.calendar_days.gt(1).sum()),
            permutation_max_error=float(np.max(abs(left.increment-right.increment))),old_control_reproduced=True,label_poison_invariant=True)
        checks.append(item);frames.append(result.assign(month=month));cells.append(cell.assign(month=month));print(item,flush=True)
    pd.concat(frames,ignore_index=True).to_parquet(OUT/'predictions.parquet',index=False)
    pd.concat(cells,ignore_index=True).to_parquet(OUT/'cells.parquet',index=False)
    write(OUT/'inference.json',dict(identity=art(OUT/'identity.json'),host=socket.gethostname(),seconds=time.monotonic()-started,
        checks=checks,artifacts=[art(OUT/n) for n in ['predictions.parquet','cells.parquet']]))


def score():
    import daily_main_refinements_cv as selected
    import tuning_seasonal as ts
    identity=guard();receipt=read(OUT/'inference.json');assert receipt['identity']==art(OUT/'identity.json')
    for a in [*identity['sources'],*receipt['artifacts']]:verify(a)
    assert not (OUT/'result.json').exists()
    f=pd.read_parquet(parent.OUT/'scored.parquet',columns=['mvt_id','month','day','y','control','candidate']).rename(columns={'control':'v77','candidate':'control'})
    p=pd.read_parquet(OUT/'predictions.parquet').set_index('mvt_id').reindex(f.mvt_id)
    f['increment']=p.increment.fillna(0).to_numpy();f['old_increment']=p.old_increment.fillna(0).to_numpy()
    f['calendar_touched']=p.calendar_touched.fillna(False).to_numpy(bool)
    raw=pd.read_parquet(selected.daily.base.OUT/'rows2025.parquet').set_index('mvt_id').loc[f.mvt_id].reset_index()
    matches=pd.read_parquet(selected.daily.constraints.exclusion.OUT/'matches2025.parquet')
    def constrain(increment):
        return selected.daily.constrain(raw,f.mvt_id.to_numpy(),f.month,f.v77.to_numpy()+.5*increment,matches)
    np.testing.assert_array_equal(constrain(f.old_increment.to_numpy()),f.control)
    f['candidate']=constrain(f.increment.to_numpy())
    np.testing.assert_array_equal(f.loc[~f.calendar_touched,'candidate'],f.loc[~f.calendar_touched,'control'])
    f.to_parquet(OUT/'scored.parquet',index=False)
    scores={m:{c:float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in ['control','candidate']} for m,g in f.groupby('month')}
    paired=ts.paired_bootstrap(f.day,f.month,(f.control-f.y)**2,(f.candidate-f.y)**2)
    write(OUT/'result.json',dict(design=DESIGN,identity=art(OUT/'identity.json'),inference=art(OUT/'inference.json'),scored=art(OUT/'scored.parquet'),
        scores=scores,cv={c:float(np.mean([s[c] for s in scores.values()])) for c in ['control','candidate']},paired=paired,
        touched=int(f.calendar_touched.sum()),changed=int(f.candidate.ne(f.control).sum()),exact_v78_and_fallback_replayed=True,
        independent_final_audit_pending=True))
    print(dict(cv=read(OUT/'result.json')['cv'],paired=paired),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','infer','score']);a=p.parse_args()
    with threadpool_limits(1):globals()[a.stage]()
