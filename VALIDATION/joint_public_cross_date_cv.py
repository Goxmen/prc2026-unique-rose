"""One fixed cross-date inference comparison, retaining all previous controls."""
import argparse
from pathlib import Path
import time

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import joint_public_cross_date as model
import joint_public_cross_date_pilot as pilot
import daily_main_refinements_cv as selected
import tuning_seasonal as validation

ROOT=pilot.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261012_joint_public_cross_date'
BASE=pilot.PARENT
SAME=ROOT/'VALIDATION/runs/v3_20261012_joint_public_other_runway'
P79=ROOT/'VALIDATION/runs/v3_20261012_target205_candidate/submission_v79'
POOL=ROOT/'VALIDATION/runs/v3_20261012_nn_public_tabm_pool3_stack'
COMBO=ROOT/'VALIDATION/runs/v3_20261012_tabm_pool3_other_runway_combo'
read,art,write,verify=pilot.read,pilot.art,pilot.write,pilot.verify
DESIGN=dict(settings=model.SETTINGS,source_only_pilot=pilot.DESIGN,
    parent='selected same-date robust other-runway correction, on exactv78 and transported onto v79',
    prediction='original constrain(v78 +0.5*(cross_date_joint-saved_original_joint)); existing same-date correction replays exactly',
    transport='rint(max(0,v79+candidate_v78-exactv78)); coefficient one',
    gates='paired-day upper endpoint below zero AND both months improve versus same-date parent on both v78 and v79',
    combo='Only if both component gates pass: replace same-date runway delta by cross-date delta in the already selected three-TabM/runway combination; coefficient one and one clip',
    tolerances='old and selected floating posterior replay atol1e-8; integer controls exact; no-extra rows use saved selected posterior exactly',
    no_fit=True,no_weight_search=True,no_subgroup_selection=True,no_upload=True,
    caveat='Reused development data and approximate moment conditioning; not independent validation')


def freeze():
    assert not (OUT/'identity.json').exists()
    result=read(pilot.OUT/'result.json')
    assert result['status']=='pass' and result['targets_with_information']>0
    assert read(SAME/'audit.json')['status']=='pass'
    assert read(COMBO/'result.json')['paired']['one_tabm_runway']['decision']=='improvement'
    assert read(ROOT/'VALIDATION/reports/public_tabm_20261012/pool3_score_audit.json')['status']=='pass'
    paths=[Path(__file__),Path(model.__file__),Path(model.same.__file__),Path(model.base.__file__),Path(pilot.__file__),
        pilot.OUT/'identity.json',pilot.OUT/'result.json',P79/'cv.parquet',
        POOL/'scored.parquet',POOL/'result.json',COMBO/'scored.parquet',COMBO/'result.json',
        ROOT/'VALIDATION/reports/public_tabm_20261012/pool3_score_audit.json',
        *[BASE/p for p in ['identity.json','inputs.parquet','predictions.parquet','taxi_source.parquet','delay_source.parquet','parameters.json','scored.parquet','audit.json']],
        *[SAME/p for p in ['identity.json','predictions.parquet','scored.parquet','audit.json']],
        Path(selected.daily.__file__),Path(selected.daily.constraints.__file__),Path(selected.daily.constraints.exclusion.__file__),
        selected.daily.base.OUT/'rows2025.parquet',selected.daily.constraints.exclusion.OUT/'matches2025.parquet']
    write(OUT/'identity.json',dict(design=DESIGN,checks=model.checks(),sources=[art(p) for p in paths]))


def guard():
    identity=read(OUT/'identity.json');assert identity['design']==DESIGN
    for record in identity['sources']:verify(record)
    return identity


def infer():
    guard();assert not (OUT/'inference.json').exists() and not (OUT/'predictions.parquet').exists()
    started=time.monotonic()
    inputs=pd.read_parquet(BASE/'inputs.parquet',columns=model.INPUTS)
    original=pd.read_parquet(BASE/'predictions.parquet',columns=['mvt_id','joint','touched']).set_index('mvt_id')
    same=pd.read_parquet(SAME/'predictions.parquet',columns=['mvt_id','new_joint','delta']).set_index('mvt_id')
    taxi=pd.read_parquet(BASE/'taxi_source.parquet');delay=pd.read_parquet(BASE/'delay_source.parquet')
    params=read(BASE/'parameters.json');frames=[];diagnostics=[]
    for month,prm in params.items():
        rows=inputs[inputs.month.eq(month)].reset_index(drop=True)
        bank=model.Messages(rows,taxi,delay,prm['taxi'],prm['delay'],prm['support'])
        ids=rows.mvt_id;old=original.loc[ids];parent=same.loc[ids]
        f=pd.DataFrame(dict(mvt_id=ids,month=month,base_joint=old.joint.to_numpy(),
            same_joint=parent.new_joint.to_numpy(),cross_joint=parent.new_joint.to_numpy(),
            touched=old.touched.to_numpy(),new_cells=0,new_weight=1.,new_q=0.))
        errors=[]
        for i in np.flatnonzero(f.touched.to_numpy(bool)):
            r=bank.condition(i);g=r['group'];y,w,v=bank.supports[i]
            old_post=model.base.posterior_mean(y,w,v,g['observed'][:3],r['rawmean'],r['rawcov'])
            same_post=model.base.posterior_mean(y,w,v,g['observed'][:3],r['oldmean'],r['oldcov'])
            np.testing.assert_allclose(old_post,f.base_joint.iloc[i],atol=1e-8,rtol=0)
            np.testing.assert_allclose(same_post,f.same_joint.iloc[i],atol=1e-8,rtol=0)
            errors.append([abs(old_post-f.base_joint.iloc[i]),abs(same_post-f.same_joint.iloc[i])])
            if not g['new_cells']:continue
            new=model.base.posterior_mean(y,w,v,g['observed'][:3],r['newmean'],r['newcov'])
            f.loc[i,['cross_joint','new_cells','new_weight','new_q']]=[
                new,len(g['new_cells']),r['new_diagnostic']['weight'],r['new_diagnostic']['q']]
        f['delta']=f.cross_joint-f.base_joint;f['same_delta']=f.same_joint-f.base_joint
        f['additional_delta']=f.cross_joint-f.same_joint
        np.testing.assert_array_equal(f.same_delta,parent.delta)
        assert not f.loc[f.new_cells.eq(0),'additional_delta'].any()
        assert np.isfinite(f[['delta','same_delta','additional_delta']]).all().all()
        frames.append(f)
        diagnostics.append(dict(month=month,rows=len(f),targets=int(f.touched.sum()),
            targets_with_new_cells=int(f.new_cells.gt(0).sum()),
            maximum_control_replay_errors=np.max(errors,axis=0).tolist()))
    pd.concat(frames,ignore_index=True).to_parquet(OUT/'predictions.parquet',index=False)
    write(OUT/'inference.json',dict(identity=art(OUT/'identity.json'),prediction=art(OUT/'predictions.parquet'),
        seconds=time.monotonic()-started,diagnostics=diagnostics,no_label_or_score_values_loaded=True,
        unchanged_old_and_same_date_posteriors_replayed=True,no_extra_uses_saved_same_date_exactly=True))
    print(dict(seconds=time.monotonic()-started,diagnostics=diagnostics),flush=True)


def score():
    guard();receipt=read(OUT/'inference.json');verify(receipt['prediction'])
    assert receipt['identity']==art(OUT/'identity.json') and not (OUT/'result.json').exists()
    old=pd.read_parquet(BASE/'scored.parquet');ids=old.mvt_id
    previous=pd.read_parquet(SAME/'scored.parquet').set_index('mvt_id').loc[ids]
    parent=pd.read_parquet(P79/'cv.parquet').set_index('mvt_id').loc[ids]
    predictions=pd.read_parquet(OUT/'predictions.parquet').set_index('mvt_id').reindex(ids)
    for f in [previous,parent]:
        for name in ['y','month','day']:np.testing.assert_array_equal(f[name],old[name])
    f=old[['mvt_id','month','day','y']].copy()
    f['v78']=old.candidate;f['same78']=previous.candidate.to_numpy()
    f['v79']=parent.pred.to_numpy();f['delta']=predictions.delta.fillna(0).to_numpy()
    raw=pd.read_parquet(selected.daily.base.OUT/'rows2025.parquet').set_index('mvt_id').loc[ids].reset_index()
    matches=pd.read_parquet(selected.daily.constraints.exclusion.OUT/'matches2025.parquet')
    constrain=selected.daily.constrain
    replay=constrain(raw,ids.to_numpy(),f.month,f.v78.to_numpy()+.5*predictions.same_delta.fillna(0).to_numpy(),matches)
    np.testing.assert_array_equal(replay,f.same78)
    f['cross78']=constrain(raw,ids.to_numpy(),f.month,f.v78.to_numpy()+.5*f.delta.to_numpy(),matches)
    unchanged=predictions.additional_delta.fillna(0).eq(0).to_numpy()
    np.testing.assert_array_equal(f.cross78[unchanged],f.same78[unchanged])
    f['same79']=np.rint(np.maximum(0,f.v79+f.same78-f.v78))
    f['cross79']=np.rint(np.maximum(0,f.v79+f.cross78-f.v78))
    cols=['v78','same78','cross78','v79','same79','cross79']
    scores={m:{c:float(np.sqrt(np.mean((q[c]-q.y)**2))) for c in cols} for m,q in f.groupby('month')}
    paired={v:validation.paired_bootstrap(f.day,f.month,(f['same'+v]-f.y)**2,(f['cross'+v]-f.y)**2) for v in ['78','79']}
    gate=all(r['decision']=='improvement' for r in paired.values()) and all(
        q['cross'+v]<q['same'+v] for q in scores.values() for v in ['78','79'])
    combination=None
    if gate:
        pool=pd.read_parquet(POOL/'scored.parquet').set_index('mvt_id').loc[ids]
        before=pd.read_parquet(COMBO/'scored.parquet').set_index('mvt_id').loc[ids]
        np.testing.assert_array_equal(pool.selected,f.v78)
        f['combo_control']=np.rint(np.maximum(0,f.v79+pool.three_tabm.to_numpy()-f.v78+f.same78-f.v78))
        np.testing.assert_array_equal(f.combo_control,before.three_tabm_runway)
        f['combo_candidate']=np.rint(np.maximum(0,f.v79+pool.three_tabm.to_numpy()-f.v78+f.cross78-f.v78))
        combination=dict(paired=validation.paired_bootstrap(f.day,f.month,(f.combo_control-f.y)**2,(f.combo_candidate-f.y)**2),
            months={m:{c:float(np.sqrt(np.mean((q[c]-q.y)**2))) for c in ['combo_control','combo_candidate']} for m,q in f.groupby('month')})
    f.to_parquet(OUT/'scored.parquet',index=False)
    value=dict(design=DESIGN,identity=art(OUT/'identity.json'),inference=art(OUT/'inference.json'),
        scored=art(OUT/'scored.parquet'),scores=scores,cv={c:float(np.mean([q[c] for q in scores.values()])) for c in cols},
        paired=paired,standalone_gate_pass=gate,combination=combination,rows=len(f),
        exact_selected_same_date_control=True,no_extra_rows_unchanged=True,no_upload=True)
    write(OUT/'result.json',value)
    with duckdb.connect(config={'threads':1}) as con:
        con.read_parquet(str(OUT/'scored.parquet')).create_view('f')
        con.read_parquet(str(selected.daily.base.OUT/'rows2025.parquet')).create_view('r')
        n,u,bad=con.execute('''SELECT count(*),count(DISTINCT f.mvt_id),sum(CASE WHEN
            f.y IS DISTINCT FROM r.y OR f.month IS DISTINCT FROM r.month OR f.day IS DISTINCT FROM r.day
            OR f.same79 IS DISTINCT FROM round_even(greatest(0,f.v79+f.same78-f.v78),0)
            OR f.cross79 IS DISTINCT FROM round_even(greatest(0,f.v79+f.cross78-f.v78),0)
            THEN 1 ELSE 0 END) FROM f JOIN r USING(mvt_id)''').fetchone()
        assert n==u==344419 and bad==0
        for c in cols:
            for m,s in con.execute(f'SELECT month,sqrt(avg(pow({c}-y,2))) FROM f GROUP BY month').fetchall():
                assert abs(s-scores[m][c])<1e-8
    write(OUT/'audit.json',dict(status='pass',result=art(OUT/'result.json'),rows=n,
        exact_all_row_parent_truth_calendar_transport_and_SQL_scores=True,
        scope='Within-runner score/SQL audit; separate posterior/file audit still required before selection'))
    print(dict(cv=value['cv'],paired=paired,gate=gate,combination=combination),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','infer','score'])
    with threadpool_limits(1):globals()[parser.parse_args().stage]()
