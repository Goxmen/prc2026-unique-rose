"""Prepare the selected joint-source increment on the accepted Jan/Jul test.

Reuse v77 and serialized v70, fit only the two small source calibrations on
2025, then run the fixed validated posterior. No taxi-model refit or upload.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

import joint_public_moments_cv as cv
import daily_main_refinements_submit as parent
import daily_offblock_alignment as alignment
import taxi_count_sum_jan_jul as taxi
from submit import fill_template

ROOT=cv.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261011_joint_public_moments_submit'
MANIFEST=OUT/'identity.json'
V77=ROOT/'VALIDATION/runs/v3_20261011_daily_refinements_candidate/submission_v77'
V72=ROOT/'VALIDATION/runs/v3_20261011_daily_delay_submit/submission_v72'
art,read,write,verify,sha=cv.art,cv.read,cv.write,cv.verify,cv.sha
DESIGN=dict(parent='v77',change=cv.DESIGN,rows=344841,
    source_calibration='same functions and settings, all2025 labels; preserve full source calibration rows',
    support='reuse v72 full2025 all-departure quantiles and mean, matching the CV support lineage',
    prior='exact serialized v70 in original submitting order',
    dates='once estimated from takeoff minus v70; retain original interior-day and IFR scope',
    no_taxi_fit=True,no_version=True,no_upload=True)


def prepare():
    cv.guard();result=read(cv.OUT/'result.json');audit=read(cv.OUT/'audit.json')
    assert result['paired']['decision']=='improvement' and audit['status']=='pass'
    assert audit['result']==art(cv.OUT/'result.json')
    assert not MANIFEST.exists(),'Preserve prepared component'
    old=read(V77/'manifest.json')
    assert old['version']==77 and old['audit']['independent_review']['status']=='pass'
    assert sha(ROOT/old['file'])==old['sha256']
    for p,h in old['sources'].items():assert sha(ROOT/p)==h,p
    template=pq.read_table(ROOT/'data/submitting.parquet');ids=template['MVT_ID_mvt'].to_numpy()
    p70=pq.read_table(ROOT/'submissions/unique-rose_v70.parquet')
    p77=pq.read_table(ROOT/old['file'])
    for t in [p70,p77]:assert t.schema==template.schema and t['MVT_ID_mvt'].equals(template['MVT_ID_mvt'])
    prior=p70['TAXITIME_SEC_mvt'].to_numpy().astype(float)
    previous=pd.read_parquet(parent.OUT/'predictions.parquet').set_index('mvt_id').loc[ids]
    np.testing.assert_array_equal(previous.candidate_pred,p77['TAXITIME_SEC_mvt'].to_numpy())
    raw=pd.read_parquet(taxi.daily.base.OUT/'ranking_inputs.parquet').set_index('mvt_id').loc[ids]
    clocks=pd.read_parquet(ROOT/'data/ranking.parquet',columns=['MVT_ID_mvt','MVT_TIME_UTC_mvt','FLIGHT_RULE_mvt'],
        filters=[('PHASE_mvt','==','DEP')]).set_index('MVT_ID_mvt').loc[ids]
    epoch=clocks.MVT_TIME_UTC_mvt.astype('int64').to_numpy()/1e6
    # Source timestamps are microseconds in the retained Arrow schema. Assert
    # against Python timestamps instead of assuming pandas nanosecond storage.
    check=np.array([t.timestamp() for t in clocks.MVT_TIME_UTC_mvt.iloc[::4096]])
    np.testing.assert_array_equal(epoch[::4096],check)
    assigned=alignment.dates(epoch,prior)
    months=clocks.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m').to_numpy()
    scope=np.zeros(len(ids),bool)
    for month in ['2026-01','2026-07']:
        ix=np.flatnonzero(months==month);scope[ix]=alignment.interior(assigned[ix],month)
    scope&=~clocks.FLIGHT_RULE_mvt.eq('V').to_numpy()
    inputs=pd.DataFrame(dict(mvt_id=ids,airport=raw.airport.to_numpy(),month=months,gap=raw.gap.to_numpy(),
        prior=prior,eligible=previous.eligible.to_numpy(),quality=previous.quality_eligible.to_numpy(),
        runway=previous.runway.to_numpy(),local_day=previous.local_day.to_numpy(),assigned_day=assigned,
        fltrule=clocks.FLIGHT_RULE_mvt.to_numpy(),delay_scope=scope))
    inputs=inputs[inputs.airport.eq('LIRF')].drop(columns='airport').reset_index(drop=True)
    assert not {'y','true_day','BLOCK_TIME_UTC_mvt'}.intersection(inputs.columns)
    allowed=[f'2025-{m:02d}' for m in range(1,13)]
    taxipar,calibration=taxi.calibrate(taxi.daily.rows(),taxi.daily.public(),allowed)
    full=pd.read_parquet(alignment.OUT/'rows2025.parquet',filters=[('airport','==','LIRF')])
    train=full[~full.fltrule.eq('V')].assign(day=lambda x:x.true_day)
    safe=np.zeros(len(train),bool)
    for month in allowed:safe|=alignment.interior(train.day,month)
    train=train[safe].copy();assert set(train.month).issubset(allowed)
    delay_source=cv.model.prior.base.public(2025)
    dep=cv.model.prior.fit(train,delay_source)['LIRF']
    support=read(V72/'parameters.json')['parameters']['LIRF']
    parameters=dict(taxi=taxipar,delay=dep,support=support,allowed=allowed)
    st=taxi.daily.public();st=st[st.airport.eq('LIRF')].drop(columns='airport')
    sd=cv.model.prior.base.public(2026);sd=sd[sd.airport.eq('LIRF')].drop(columns='airport')
    OUT.mkdir(parents=True,exist_ok=True);write(OUT/'parent_v77_snapshot.json',old)
    inputs.to_parquet(OUT/'inputs.parquet',index=False);st.to_parquet(OUT/'taxi_source.parquet',index=False)
    sd.to_parquet(OUT/'delay_source.parquet',index=False);calibration.to_parquet(OUT/'taxi_calibration.parquet',index=False)
    train[['mvt_id','month','day','gap','y','fltrule']].to_parquet(OUT/'delay_calibration_rows.parquet',index=False)
    write(OUT/'parameters.json',parameters)
    source_paths=[Path(__file__),Path(cv.__file__),Path(cv.model.__file__),Path(alignment.__file__),
        Path(taxi.__file__),Path(cv.model.taxi.__file__),Path(cv.model.prior.__file__),Path(cv.model.prior.base.__file__),
        Path(parent.__file__),Path(taxi.daily.constraints.__file__),Path(taxi.daily.constraints.exclusion.__file__),
        ROOT/'VALIDATION/submit.py',ROOT/'VALIDATION/records.py',
        cv.OUT/'identity.json',cv.OUT/'result.json',cv.OUT/'audit.json',cv.OUT/'scored.parquet',
        cv.OUT/'input_audit.json',cv.OUT/'inference.json',parent.OUT/'predictions.parquet',parent.OUT/'audit.json',
        ROOT/'data/ranking.parquet',ROOT/'data/submitting.parquet',ROOT/'submissions/unique-rose_v70.parquet',ROOT/old['file'],
        taxi.daily.base.OUT/'ranking_inputs.parquet',alignment.OUT/'rows2025.parquet',taxi.daily.CACHE,
        taxi.daily.SOURCE/'daily_runway.parquet',taxi.daily.constraints.exclusion.OUT/'ranking_matches.parquet',
        V72/'parameters.json',cv.model.prior.base.SOURCE/'all_pre_departure_delays_2025.csv',
        cv.model.prior.base.SOURCE/'all_pre_departure_delays_2026.csv']
    source_paths += [OUT/n for n in ['parent_v77_snapshot.json','inputs.parquet','taxi_source.parquet','delay_source.parquet',
        'taxi_calibration.parquet','delay_calibration_rows.parquet','parameters.json']]
    write(MANIFEST,dict(design=DESIGN,sources=[art(p) for p in source_paths],
        parent_file=old['file'],parent_sha256=old['sha256'],rome_rows=len(inputs),full_rows=len(ids),
        full2025_source_calibration_only=True))
    print(dict(rome_rows=len(inputs),taxi_calibration_cells=len(calibration),delay_training_rows=len(train)),flush=True)


def guard():
    identity=read(MANIFEST);assert identity['design']==DESIGN
    for a in identity['sources']:verify(a)
    return identity


def infer():
    guard();assert not (OUT/'inference.json').exists(),'Preserve inference'
    f=pd.read_parquet(OUT/'inputs.parquet');p=read(OUT/'parameters.json')
    st=pd.read_parquet(OUT/'taxi_source.parquet');sd=pd.read_parquet(OUT/'delay_source.parquet')
    frames=[];cells=[]
    for month,q in f.groupby('month',sort=True):
        out,c=cv.model.apply(q,st,sd,p['taxi'],p['delay'],p['support'])
        frames.append(out.assign(month=month));cells.append(c.assign(month=month))
    out=pd.concat(frames,ignore_index=True);out.to_parquet(OUT/'inference.parquet',index=False)
    pd.concat(cells,ignore_index=True).to_parquet(OUT/'cells.parquet',index=False)
    write(OUT/'inference.json',dict(identity=art(MANIFEST),artifacts=[art(OUT/n) for n in ['inference.parquet','cells.parquet']],
        touched=int(out.touched.sum()),rows=len(out)))
    print(dict(rome_rows=len(out),touched=int(out.touched.sum())),flush=True)


def build():
    identity=guard();receipt=read(OUT/'inference.json');assert receipt['identity']==art(MANIFEST)
    for a in receipt['artifacts']:verify(a)
    assert not (OUT/'result.json').exists(),'Preserve built component'
    parent_table=pq.read_table(ROOT/identity['parent_file']);ids=parent_table['MVT_ID_mvt'].to_numpy()
    original=parent_table['TAXITIME_SEC_mvt'].to_numpy().astype(float)
    p=pd.read_parquet(OUT/'inference.parquet').set_index('mvt_id').reindex(ids)
    increment=p.increment.fillna(0).to_numpy();touched=p.touched.fillna(False).to_numpy(bool)
    raw=pd.read_parquet(taxi.daily.base.OUT/'ranking_inputs.parquet').set_index('mvt_id').loc[ids].reset_index()
    matches=pd.read_parquet(taxi.daily.constraints.exclusion.OUT/'ranking_matches.parquet')
    def constrain(values):
        out=taxi.daily.constraints.matched_bound(raw,values)
        out,_=taxi.daily.constraints.exclusion.apply(ids,np.maximum(out,0),matches)
        return np.rint(np.maximum(out,0))
    np.testing.assert_array_equal(constrain(original),original)
    final=constrain(original+.5*increment);np.testing.assert_array_equal(final[~touched],original[~touched])
    template=pq.read_table(ROOT/'data/submitting.parquet');table=fill_template(template,ids,final)
    assert table.schema==template.schema
    for name in template.column_names:
        if name!='TAXITIME_SEC_mvt':assert table[name].equals(template[name])
    np.testing.assert_array_equal(table['TAXITIME_SEC_mvt'].to_numpy(),final)
    pq.write_table(table,OUT/'candidate.parquet');assert pq.read_table(OUT/'candidate.parquet').equals(table)
    pd.DataFrame(dict(mvt_id=ids,parent=original,increment=increment,touched=touched,pred=final)).to_parquet(OUT/'predictions.parquet',index=False)
    f=pd.read_parquet(cv.OUT/'scored.parquet').rename(columns={'candidate':'pred'})
    f.to_parquet(OUT/'cv.parquet',index=False);result=read(cv.OUT/'result.json')
    write(OUT/'result.json',dict(design=DESIGN,identity=art(MANIFEST),rows=len(ids),
        jan_jul_cv=result['cv']['candidate'],cv_months={m:v['candidate'] for m,v in result['scores'].items()},paired=result['paired'],
        changed=int(np.sum(final!=original)),touched=int(touched.sum()),rms_change=float(np.sqrt(np.mean((final-original)**2))),
        parent_and_unchanged_rows_exact=True,
        artifacts=[art(OUT/n) for n in ['candidate.parquet','predictions.parquet','cv.parquet']],no_version=True,no_upload=True))
    print(dict(rows=len(ids),changed=int(np.sum(final!=original)),cv=result['cv']['candidate']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','infer','build']);a=p.parse_args()
    with threadpool_limits(1):globals()[a.stage]()
