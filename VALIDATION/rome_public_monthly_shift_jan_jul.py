"""Fixed public-aggregate airline/month adaptation on the selected joint stack.

Only one arm and one prior scale. No departure target is used to infer shifts;
public held-month totals are intentionally used as aggregate observations.
Jan/Jul labels enter scoring only. Existing training-only source-noise estimates
are reused without fitting or selecting new parameters.
"""
import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import rome_public_monthly_shift as shift
import public_stack_cv as joint
import daily_offblock_alignment as aligned
import rome_signals
from records import write_json

ROOT = joint.ROOT
OUT = ROOT / 'VALIDATION/runs/v3_20261011_rome_public_monthly_shift'
art, read, verify = joint.art, joint.read, joint.verify
DESIGN = dict(parent='selected joint public main/neural stack, fixed public Cat3/R20',
    change='infer one regularized Rome airline/month logit shift from public daily totals',
    settings=shift.SETTINGS, grouping='movement flight-number airline prefix, existing parser',
    prior_scale=.5, public_input='retained EUROCONTROL daily pre-departure total and count',
    mode_outputs='original full correction stack at classifier probabilities0 and1',
    noise='unchanged permitted-training off-block-date source parameters',
    months='use only interior held dates assigned from frozen incoming primary predictions',
    production=False, no_new_taxi_model_fits=True, no_target_label_adaptation=True,
    limitation='MAP logit shifts and independent-mode variance approximation; conditional-mode mean may differ from clipped blended prediction',
    evaluation='reused Jan/Jul development; public aggregates of held flights intentionally observed')


def guard():
    assert read(joint.OUT/'audit.json')['status']=='pass'
    assert read(aligned.OUT/'audit.json')['status']=='pass'
    assert read(aligned.OUT/'source_audit.json')['status']=='pass'
    sources=[Path(__file__),Path(shift.__file__),Path(joint.__file__),Path(aligned.__file__),Path(rome_signals.__file__),
        joint.OUT/'identity.json',joint.OUT/'audit.json',joint.OUT/'scored.parquet',
        aligned.OUT/'identity.json',aligned.OUT/'result.json',aligned.OUT/'audit.json',
        aligned.OUT/'rows2025.parquet',aligned.base.SOURCE/'all_pre_departure_delays_2025.csv',
        joint.neural.cat.OUT/'scored.parquet']
    value=dict(design=DESIGN, sources=[art(p) for p in sources],checks=shift.checks())
    OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/'identity.json'
    if p.exists():
        assert read(p)==value
    else:
        assert not list(OUT.glob('*.parquet'))
        write_json(p,value)
    return value


def screen():
    guard()
    assert not (OUT/'result.json').exists(),'Preserve completed screen'
    neural=joint.neural
    cat=neural.cat
    replay=cat.r20.physical.replay
    identity=neural.guard()
    sc,r20,_,trees=replay.baseline.baseline()
    truth=pd.read_parquet(aligned.OUT/'rows2025.parquet').set_index('mvt_id')
    prior=pd.read_parquet(cat.OUT/'scored.parquet').set_index('mvt_id')
    joint_cv=pd.read_parquet(joint.OUT/'scored.parquet').set_index('mvt_id')
    matches=pd.read_parquet(cat.r20.constraints.exclusion.OUT/'matches2025.parquet')
    source=aligned.base.public(2025).query("airport=='LIRF'")
    noise=read(aligned.OUT/'result.json')['parameters']
    frames,packets,days=[],[],[]
    logs={}
    for month in neural.single.STOP:
        d=sc.d[month];ids=d['ids']
        raw=truth.loc[ids].reset_index()
        np.testing.assert_array_equal(raw.y,d['y'])
        calls=replay.values(replay.combo.sequence.CALLS_PRED,ids,'candidate_nn')
        bank=[replay.values(verify(neural.completed(month,k,identity)['prediction']),ids,'base') for k in range(3)]
        main=replay.values(joint.main.OUT/'predictions'/f'{month}_s42.parquet',ids,'pred')
        ordinary=.71*.5*(calls+np.mean(bank,axis=0))+.29*np.mean([main,trees['airport'][42][month],trees['catboost'][42][month]],axis=0)
        p=.5*prior.loc[ids,'newcat'].to_numpy()+.5*r20[month]
        control=sc.final(month,ordinary,p)
        np.testing.assert_array_equal(control,joint_cv.loc[ids,'combined'])
        at_0=sc.final(month,ordinary,np.zeros(len(ids)))
        at_1=sc.final(month,ordinary,np.ones(len(ids)))
        rome=raw.airport.eq('LIRF').to_numpy()
        # Inference packet is built from a strict input whitelist, no y or true_day.
        inputs=raw.loc[rome,['mvt_id','gap','fltrule','takeoff_epoch']].reset_index(drop=True)
        movement=d['held']['raw']
        np.testing.assert_array_equal(movement.mvt_id,ids)
        airline=rome_signals.airline_code(movement.FLIGHT_mvt.to_numpy()[rome]).fillna('__unknown__').to_numpy()
        assigned=aligned.dates(inputs.takeoff_epoch,control[rome])
        eligible=~inputs.fltrule.eq('V').to_numpy() & aligned.interior(assigned,month)
        frame=pd.DataFrame(dict(mvt_id=inputs.mvt_id.to_numpy(),airline=airline,day=assigned,p=p[rome],
            c0=np.maximum(0,inputs.gap.to_numpy()-at_0[rome]),
            c1=np.maximum(0,inputs.gap.to_numpy()-at_1[rome]),eligible=eligible))
        prm=noise[month]
        assert prm['allowed']==aligned.base.FOLDS[month] and month not in prm['allowed']
        updated,log,day=shift.infer(frame,source[['day','total','n_public']],prm['offblock']['LIRF'])
        poisoned,_,_=shift.infer(frame.assign(y=1e12,true_day='1900-01-01'),
            source[['day','total','n_public']],prm['offblock']['LIRF'])
        np.testing.assert_array_equal(updated,poisoned)
        new_p=p.copy();new_p[rome]=updated
        candidate=sc.final(month,ordinary,new_p)
        permitted=matches[matches.donor_month.isin(aligned.base.FOLDS[month]+[month])]
        outputs={}
        for name,final in [('control',control),('candidate',candidate)]:
            transferred=cat.r20.constraints.matched_bound(raw,prior.loc[ids,'v74'].to_numpy()+final-prior.loc[ids,'control'].to_numpy())
            transferred,_=cat.r20.constraints.exclusion.apply(ids,np.maximum(transferred,0),permitted)
            outputs[name]=final;outputs[name+'_transfer']=transferred
        np.testing.assert_array_equal(outputs['control_transfer'],joint_cv.loc[ids,'combined_transfer'])
        np.testing.assert_array_equal(outputs['candidate_transfer'][~rome],outputs['control_transfer'][~rome])
        log['mode_mean_vs_control_max_seconds']=float(np.max(abs((1-p[rome])*at_0[rome]+p[rome]*at_1[rome]-control[rome])))
        log['fixed_noise']=prm['offblock']['LIRF']
        logs[month]=log
        frame['updated_p']=updated;frame['month']=month;packets.append(frame)
        day['month']=month;days.append(day)
        frames.append(pd.DataFrame(dict(mvt_id=ids,month=month,day=d['day'],y=d['y'],**outputs)))
    f=pd.concat(frames,ignore_index=True)
    assert len(f)==344419 and f.mvt_id.is_unique
    columns=list(outputs)
    assert np.isfinite(f[columns]).all().all() and f[columns].ge(0).all().all()
    scores={m:{c:float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in columns} for m,g in f.groupby('month')}
    pairs=[('control','candidate'),('control_transfer','candidate_transfer')]
    paired={f'{b}_vs_{a}':neural.single.shared.ts.paired_bootstrap(f.day,f.month,(f[a]-f.y)**2,(f[b]-f.y)**2) for a,b in pairs}
    f.to_parquet(OUT/'scored.parquet',index=False)
    pd.concat(packets,ignore_index=True).to_parquet(OUT/'inference_packets.parquet',index=False)
    pd.concat(days,ignore_index=True).to_parquet(OUT/'days.parquet',index=False)
    result=dict(design=DESIGN,scores=scores,cv={c:float(np.mean([s[c] for s in scores.values()])) for c in columns},
        paired=paired,months=logs,identity=art(OUT/'identity.json'),predictions=art(OUT/'scored.parquet'),
        inference_packets=art(OUT/'inference_packets.parquet'),days=art(OUT/'days.parquet'),
        selected_controls_exact=True,non_rome_unchanged=True,all_rome_label_poisoning_identity=True,
        interpretation='One fixed public-only monthly adaptation screen; no production selected')
    write_json(OUT/'result.json',result)
    print(dict(cv=result['cv'],paired=paired,months=logs),flush=True)


def audit():
    guard()
    assert not (OUT/'audit.json').exists(),'Preserve completed audit'
    result=read(OUT/'result.json')
    assert result['identity']==art(OUT/'identity.json')
    for key in ['predictions','inference_packets','days']:
        verify(result[key])
    packets=pd.read_parquet(OUT/'inference_packets.parquet')
    day=pd.read_parquet(OUT/'days.parquet')
    source=aligned.base.public(2025).query("airport=='LIRF'").set_index('day')
    maximum=0.
    for month,g in packets.groupby('month'):
        prm=result['months'][month]['fixed_noise']
        eligible=g[g.eligible]
        # Recompute daily means in a direct row loop, separately from bincount.
        for stamp,rows in eligible.groupby('day'):
            record=day[(day.month==month)&(day.day==stamp)].iloc[0]
            expected=sum((1-r.p)*r.c0+r.p*r.c1 for r in rows.itertuples())
            actual=sum((1-r.updated_p)*r.c0+r.updated_p*r.c1 for r in rows.itertuples())
            error=max(abs(expected-record.baseline),abs(actual-record.adapted))
            maximum=max(maximum,error)
            assert record.n==len(rows) and record.dn==source.loc[stamp,'n_public']-len(rows)
            np.testing.assert_allclose(record.target,source.loc[stamp,'total']-prm['bias'],atol=1e-8,rtol=0)
            np.testing.assert_allclose([expected,actual],[record.baseline,record.adapted],atol=1e-7,rtol=0)
            latent=sum(r.p*(1-r.p)*(r.c1-r.c0)**2 for r in rows.itertuples())
            sd=np.sqrt(prm['sigma']**2+(prm['kappa']*record.dn)**2+latent+
                len(rows)*shift.SETTINGS['ordinary_sigma']**2+len(rows)**2*shift.SETTINGS['common_day_sigma']**2)
            np.testing.assert_allclose([record.latent_variance,record.sd],[latent,sd],atol=1e-7,rtol=1e-14)
        assert g.loc[~g.eligible,'updated_p'].equals(g.loc[~g.eligible,'p'])
        shifts=result['months'][month]['airlines']
        active=g.eligible & g.p.gt(0) & g.p.lt(1) & g.c0.ne(g.c1)
        expected=shift.expit(shift.logit(g.loc[active,'p'])+g.loc[active,'airline'].map(shifts))
        np.testing.assert_array_equal(expected,g.loc[active,'updated_p'])
    with duckdb.connect(config={'threads':1}) as con:
        for name,path in [('candidate',OUT/'scored.parquet'),('truth',aligned.OUT/'rows2025.parquet'),('control',joint.OUT/'scored.parquet')]:
            con.read_parquet(str(path)).create_view(name)
        n,u,wrong=con.execute('''select count(*),count(distinct c.mvt_id),sum(case when
            c.y is distinct from t.y or c.month is distinct from t.month or c.day is distinct from t.day
            or c.control is distinct from p.combined or c.control_transfer is distinct from p.combined_transfer
            or (t.airport!='LIRF' and c.candidate_transfer is distinct from c.control_transfer)
            then 1 else 0 end) from candidate c join truth t using(mvt_id) join control p using(mvt_id)''').fetchone()
        assert n==u==344419 and wrong==0
        for name in result['cv']:
            values=con.execute(f'select month,sqrt(avg(pow({name}-y,2))) AS value from candidate group by month').fetchall()
            for month,value in values:
                np.testing.assert_allclose(value,result['scores'][month][name],atol=1e-8,rtol=0)
            np.testing.assert_allclose(np.mean([v for m,v in values]),result['cv'][name],atol=1e-8,rtol=0)
    value=dict(status='pass',rows=n,independent_daily_aggregation_max_error=maximum,
        public_source_targets_counts_exact=True,shared_airline_shifts_exact=True,all_sql_truth_controls_scores=True,
        non_rome_unchanged=True,result=art(OUT/'result.json'),identity=art(OUT/'identity.json'),source=art(__file__),
        scope='Direct row-loop public aggregate/probability audit and independent SQL score/population audit; all-row poisoned labels leave inference unchanged')
    write_json(OUT/'audit.json',value)
    print(value,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['guard','screen','audit'])
    args=parser.parse_args()
    with threadpool_limits(1):
        globals()[args.stage]()
