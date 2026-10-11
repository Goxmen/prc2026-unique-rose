"""Fixed non-Rome extension of the selected joint-public-source correction."""
import argparse
from pathlib import Path
import socket
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import joint_public_moments_cv as parent
import joint_public_airports as model

ROOT=parent.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261011_joint_public_airports'
art,read,write,verify,sha=parent.art,parent.read,parent.write,parent.verify,parent.sha
DESIGN=dict(settings=model.SETTINGS,parent='exact selected v78',
    primary='v78 plus fixed0.5 joint-minus-departure-only increment at all nine non-Rome airports',
    scope='same eligible/quality/supported-source long-gap gate; no held-label airport selection',
    calibration='same source count/sum fit at each native airport; other fold parameters reused',
    retained='all Rome and ordinary predictions, fixed v70 prior, own source calendars and final constraints',
    limitations=['moment approximation','already public-informed parent','extensively reused development months'],
    no_new_taxi_fit=True,no_weight_scan=True,no_production=True,no_upload=True)


def prepare():
    import daily_main_refinements_cv as selected
    import daily_heavy_tail as heavy
    import taxi_count_sum_jan_jul as taxi
    parent.guard();assert read(parent.OUT/'audit.json')['status']=='pass'
    assert not (OUT/'identity.json').exists()
    raw=pd.read_parquet(heavy.aligned.OUT/'rows2025.parquet').set_index('mvt_id')
    f=pd.read_parquet(heavy.aligned.OUT/'scored.parquet',columns=['mvt_id','month','parent','assigned_day','new_scope'])
    f=f[raw.loc[f.mvt_id,'airport'].to_numpy()!='LIRF'].reset_index(drop=True);r=raw.loc[f.mvt_id]
    np.testing.assert_array_equal(f.month,r.month)
    assigned=heavy.aligned.dates(r.takeoff_epoch,f.parent);np.testing.assert_array_equal(assigned,f.assigned_day)
    allrows=taxi.daily.rows();eligible=allrows.drop(columns='y').set_index('mvt_id').loc[f.mvt_id]
    quality=pd.read_parquet(selected.quality.OUT/'scored.parquet',columns=['mvt_id','quality_eligible']).set_index('mvt_id').loc[f.mvt_id]
    inputs=pd.DataFrame(dict(mvt_id=f.mvt_id,airport=r.airport.to_numpy(),month=f.month,gap=r.gap.to_numpy(),prior=f.parent,
        eligible=eligible.eligible.to_numpy(),quality=quality.quality_eligible.to_numpy(),runway=eligible.runway.to_numpy(),
        local_day=eligible.local_day.to_numpy(),assigned_day=assigned,fltrule=r.fltrule.to_numpy(),delay_scope=f.new_scope))
    assert len(inputs)==317891 and inputs.mvt_id.is_unique
    airports=sorted(inputs.airport.unique());assert len(airports)==9 and 'LIRF' not in airports
    st=taxi.daily.public();sd=model.base.prior.base.public(2025)
    old=read(heavy.aligned.OUT/'result.json')['parameters'];sp=read(taxi.TAU)['parameters']
    parameters={};calibration=[]
    for month,allowed in taxi.daily.base.FOLDS.items():
        parameters[month]={};train=allrows[allrows.month.isin(allowed)]
        for airport in airports:
            tp,cells=model.calibrate(airport,train,st,allowed)
            parameters[month][airport]=dict(taxi=tp,delay=old[month]['offblock'][airport],support=sp[month][airport])
            calibration.append(cells.assign(fold=month))
    OUT.mkdir(parents=True,exist_ok=True)
    inputs.to_parquet(OUT/'inputs.parquet',index=False);st.to_parquet(OUT/'taxi_source.parquet',index=False)
    sd.to_parquet(OUT/'delay_source.parquet',index=False);write(OUT/'parameters.json',parameters)
    pd.concat(calibration,ignore_index=True).to_parquet(OUT/'calibration.parquet',index=False)
    helpers=[Path(__file__),Path(model.__file__),Path(model.base.__file__),Path(parent.__file__),Path(taxi.__file__),
        Path(model.base.prior.__file__),Path(model.base.prior.base.__file__),Path(model.base.taxi.__file__),ROOT/'VALIDATION/records.py']
    sources=[heavy.aligned.OUT/'rows2025.parquet',heavy.aligned.OUT/'scored.parquet',heavy.aligned.OUT/'result.json',
        taxi.TAU,taxi.daily.CACHE,taxi.daily.SOURCE/'daily_runway.parquet',model.base.prior.base.SOURCE/'all_pre_departure_delays_2025.csv',
        selected.quality.OUT/'scored.parquet',parent.OUT/'scored.parquet',parent.OUT/'audit.json',parent.OUT/'result.json']
    identity=dict(design=DESIGN,helpers=[art(p) for p in helpers],sources=[art(p) for p in sources],
        inputs=[art(OUT/p) for p in ['inputs.parquet','taxi_source.parquet','delay_source.parquet','parameters.json']],
        calibration=art(OUT/'calibration.parquet'),checks=model.checks(),airports=airports,rows=len(inputs),
        input_dates_exact=True,no_labels_or_true_offblock_dates_in_inference_packet=True)
    write(OUT/'identity.json',identity);print(dict(rows=len(inputs),airports=airports,calibration_cells=sum(len(c) for c in calibration)),flush=True)


def guard():
    identity=read(OUT/'identity.json');assert identity['design']==DESIGN
    for a in [*identity['helpers'],*identity['inputs']]:verify(a)
    return identity


def infer():
    guard();assert not (OUT/'inference.json').exists() and not (OUT/'predictions.parquet').exists()
    inputs=pd.read_parquet(OUT/'inputs.parquet');st=pd.read_parquet(OUT/'taxi_source.parquet');sd=pd.read_parquet(OUT/'delay_source.parquet')
    prm=read(OUT/'parameters.json');frames=[];cells=[];checks=[];started=time.monotonic()
    for (month,airport),rows in inputs.groupby(['month','airport']):
        rows=rows.reset_index(drop=True);p=prm[month][airport]
        a=st[st.airport.eq(airport)].drop(columns='airport');b=sd[sd.airport.eq(airport)].drop(columns='airport')
        result,cell=model.apply(airport,rows,a,b,p['taxi'],p['delay'],p['support'])
        shuffled=rows.sample(frac=1,random_state=82).assign(y=1e12,true_day='1900-01-01')
        replay,_=model.apply(airport,shuffled,a,b,p['taxi'],p['delay'],p['support'])
        left=result.set_index('mvt_id').sort_index();right=replay.set_index('mvt_id').sort_index()
        np.testing.assert_array_equal(left.touched,right.touched)
        np.testing.assert_allclose(left[['dep_only','joint','increment']],right[['dep_only','joint','increment']],atol=1e-7,rtol=0)
        record=dict(month=month,airport=airport,rows=len(rows),touched=int(result.touched.sum()),
            permutation_max_difference=float(np.max(abs(left.increment-right.increment))),label_poison_invariant=True)
        checks.append(record);frames.append(result.assign(month=month,airport=airport));cells.append(cell.assign(month=month,airport=airport))
        print(record,flush=True)
    pd.concat(frames,ignore_index=True).to_parquet(OUT/'predictions.parquet',index=False)
    pd.concat(cells,ignore_index=True).to_parquet(OUT/'cells.parquet',index=False)
    write(OUT/'inference.json',dict(identity=art(OUT/'identity.json'),host=socket.gethostname(),seconds=time.monotonic()-started,
        checks=checks,artifacts=[art(OUT/p) for p in ['predictions.parquet','cells.parquet']]))


def score():
    import daily_main_refinements_cv as selected
    import tuning_seasonal as ts
    identity=guard();receipt=read(OUT/'inference.json');assert receipt['identity']==art(OUT/'identity.json')
    for a in receipt['artifacts']:verify(a)
    for a in identity['sources']:verify(a)
    assert not (OUT/'result.json').exists()
    f=pd.read_parquet(parent.OUT/'scored.parquet',columns=['mvt_id','month','day','y','candidate']).rename(columns={'candidate':'control'})
    p=pd.read_parquet(OUT/'predictions.parquet').set_index('mvt_id').reindex(f.mvt_id)
    f['increment']=p.increment.fillna(0).to_numpy();f['touched']=p.touched.fillna(False).to_numpy(bool)
    raw=pd.read_parquet(selected.daily.base.OUT/'rows2025.parquet').set_index('mvt_id').loc[f.mvt_id].reset_index()
    f['airport']=raw.airport
    matches=pd.read_parquet(selected.daily.constraints.exclusion.OUT/'matches2025.parquet')
    np.testing.assert_array_equal(selected.daily.constrain(raw,f.mvt_id.to_numpy(),f.month,f.control.to_numpy(),matches),f.control)
    f['candidate']=selected.daily.constrain(raw,f.mvt_id.to_numpy(),f.month,f.control.to_numpy()+.5*f.increment.to_numpy(),matches)
    np.testing.assert_array_equal(f.loc[~f.touched,'candidate'],f.loc[~f.touched,'control'])
    assert not f.loc[f.airport.eq('LIRF'),'touched'].any()
    f.to_parquet(OUT/'scored.parquet',index=False)
    scores={m:{c:float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in ['control','candidate']} for m,g in f.groupby('month')}
    paired=ts.paired_bootstrap(f.day,f.month,(f.control-f.y)**2,(f.candidate-f.y)**2)
    result=dict(design=DESIGN,identity=art(OUT/'identity.json'),inference=art(OUT/'inference.json'),scored=art(OUT/'scored.parquet'),
        scores=scores,cv={c:float(np.mean([v[c] for v in scores.values()])) for c in ['control','candidate']},paired=paired,
        touched=int(f.touched.sum()),changed=int(f.candidate.ne(f.control).sum()),all_untouched_and_Rome_rows_exact=True)
    write(OUT/'result.json',result);print(dict(cv=result['cv'],paired=paired),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','infer','score']);a=p.parse_args()
    with threadpool_limits(1):globals()[a.stage]()
