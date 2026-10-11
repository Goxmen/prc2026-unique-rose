"""Score fixed coefficient screens through the full selected v77 recipe."""
import argparse
import gc

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import rome_continuous_mixture as experiment
import daily_main_refinements_cv as current

ROOT,OUT=experiment.ROOT,experiment.OUT
art,read,verify,write=experiment.art,experiment.read,experiment.verify,experiment.write


def audit_models():
    from catboost import CatBoostClassifier,CatBoostRegressor,Pool
    identity=experiment.guard(); checks=[]
    assert not (OUT/'model_audit.json').exists(),'Preserve completed audit'
    for month,fold in identity['folds'].items():
        packet=pd.read_pickle(verify(fold['packet']))
        frame=packet['B'].copy()
        for name in packet['categoricals']:
            frame[name]=frame[name].astype(object).fillna('__missing__').map(str)
        assert list(frame)==fold['columns']
        for arm in experiment.ARMS:
            rec=read(OUT/'predictions'/f'{month}_{arm}.json')
            model_rec=read(verify(rec['model_record']))
            assert model_rec['identity']==art(OUT/'identity.json')
            cls=CatBoostClassifier if arm=='logloss' else CatBoostRegressor
            model=cls();model.load_model(str(verify(model_rec['model'])))
            assert model.tree_count_==packet['rounds'] and model.feature_names_==fold['columns']
            expected_loss='Logloss' if arm=='logloss' else 'RMSE'
            assert model.get_all_params()['loss_function']==expected_loss
            x=Pool(frame,cat_features=packet['categoricals'])
            raw=model.predict_proba(x)[:,1] if arm=='logloss' else model.predict(x)
            actual=np.zeros(len(packet['ids']));actual[packet['use']]=np.clip(raw,0,1)
            saved=pd.read_parquet(verify(rec['prediction']))
            np.testing.assert_array_equal(saved.mvt_id,packet['ids'])
            np.testing.assert_array_equal(actual,saved.p)
            checks.append(dict(month=month,arm=arm,rows=len(actual),all_desktop_predictions_exact=True,
                model_record=rec['model_record'],prediction=rec['prediction']))
            del model,x;gc.collect()
        del packet,frame;gc.collect()
    write(OUT/'model_audit.json',dict(status='pass',identity=art(OUT/'identity.json'),source=art(__file__),checks=checks))
    print('All six laptop models reload exactly on desktop',flush=True)


def score():
    identity=experiment.guard()
    assert read(OUT/'fit_complete.json')['identity']==art(OUT/'identity.json')
    assert read(OUT/'model_audit.json')['status']=='pass'
    assert not (OUT/'result.json').exists(),'Preserve completed screen'
    assert read(current.OUT/'audit.json')['status']=='pass'
    current.guard()
    neural=current.joint.neural; replay=neural.cat.r20.physical.replay
    sc,rome,_,trees=replay.baseline.baseline(); pool_identity=read(neural.single.MANIFEST)
    prior=pd.read_parquet(neural.cat.OUT/'scored.parquet').set_index('mvt_id')
    serialized=pd.read_parquet(current.rounded.OUT/'scored.parquet').set_index('mvt_id')
    selected=pd.read_parquet(current.OUT/'scored.parquet').set_index('mvt_id')
    quality=pd.read_parquet(current.quality.OUT/'scored.parquet').set_index('mvt_id')
    raw=pd.read_parquet(current.daily.base.OUT/'rows2025.parquet').set_index('mvt_id')
    inputs=current.daily.rows().drop(columns='y').set_index('mvt_id')
    source,parameters=current.daily.public(),read(current.daily.OUT/'parameters.json')
    matches=pd.read_parquet(current.daily.constraints.exclusion.OUT/'matches2025.parquet')
    frames=[]; provenance=[]
    for month in experiment.MONTHS:
        d=sc.d[month]; ids=d['ids']; packet=pd.read_pickle(verify(identity['folds'][month]['packet']))
        np.testing.assert_array_equal(packet['ids'],ids)
        gate=packet['held_gate']; del packet;gc.collect()
        calls=replay.values(replay.combo.sequence.CALLS_PRED,ids,'candidate_nn')
        physical=np.mean([replay.values(verify(neural.completed(month,k,pool_identity)['prediction']),ids,'base')
            for k in range(3)],axis=0)
        main=replay.values(verify(current.main.completed(month)['prediction']),ids,'pred')
        airport=replay.values(current.rounded.original.airport.OUT/'predictions'/f'{month}_s42.parquet',ids,'pred')
        cat=replay.values(verify(current.cv.completed(month)['prediction']),ids,'pred')
        ordinary=.71*.5*(calls+physical)+.29*np.mean([main,airport,cat],axis=0)
        p_old=.5*prior.loc[ids,'newcat'].to_numpy()+.5*rome[month]
        old_path=neural.cat.OUT/'predictions'/f'{month}_s42.parquet'
        old_seed=replay.values(old_path,ids,'p'); provenance.append(art(old_path))
        permitted=matches[matches.donor_month.isin(current.daily.base.FOLDS[month]+[month])]
        probabilities={'selected':p_old}
        for arm in experiment.ARMS:
            rec_path=OUT/'predictions'/f'{month}_{arm}.json'; rec=read(rec_path)
            assert rec['all_rome_reload_exact']
            prediction=pd.read_parquet(verify(rec['prediction']))
            np.testing.assert_array_equal(prediction.mvt_id,ids)
            model_rec=read(verify(rec['model_record'])); verify(model_rec['model'])
            assert model_rec['identity']==art(OUT/'identity.json')
            probabilities[arm]=p_old+np.where(gate,(prediction.p.to_numpy()-old_seed)/6,0)
            assert np.all((probabilities[arm]>=0)&(probabilities[arm]<=1))
            provenance.append(art(rec_path))
        values={}
        for name,p in probabilities.items():
            direct=sc.final(month,ordinary,p)
            start=serialized.loc[ids,'candidate_boundary'].to_numpy()+direct-prior.loc[ids,'control'].to_numpy()
            before=current.daily.constraints.matched_bound(raw.loc[ids].reset_index(),start)
            before,_=current.daily.constraints.exclusion.apply(ids,np.maximum(before,0),permitted)
            before=np.rint(np.maximum(before,0))
            updated,_,shift=current.daily.apply(inputs.loc[ids].reset_index(),before,source,parameters[month])
            gated=np.where(quality.loc[ids,'quality_eligible'],updated,before)
            final=current.daily.constrain(raw.loc[ids].reset_index(),ids,pd.Series(month,index=range(len(ids))),gated,matches)
            if name=='selected':
                np.testing.assert_array_equal(before,selected.loc[ids,'combined_before'])
                np.testing.assert_array_equal(shift,selected.loc[ids,'combined_shift'])
                np.testing.assert_array_equal(final,selected.loc[ids,'combined'])
            else: np.testing.assert_array_equal(before[~gate],values['selected_before'][~gate])
            values[name+'_before'],values[name+'_shift'],values[name]=before,shift,final
        frames.append(pd.DataFrame(dict(mvt_id=ids,month=month,day=d['day'],y=d['y'],gate=gate,**values)))
    f=pd.concat(frames,ignore_index=True); assert len(f)==344419 and f.mvt_id.is_unique
    names=['selected']+experiment.ARMS
    scores={m:{c:float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in names} for m,g in f.groupby('month')}
    pairs=[('selected','logloss'),('logloss','binary_rmse'),('binary_rmse','continuous_rmse'),
           ('logloss','continuous_rmse'),('selected','continuous_rmse')]
    import tuning_seasonal as ts
    paired={f'{b}_vs_{a}':ts.paired_bootstrap(f.day,f.month,(f[a]-f.y)**2,(f[b]-f.y)**2) for a,b in pairs}
    f.to_parquet(OUT/'scored.parquet',index=False)
    result=dict(design=experiment.DESIGN, identity=art(OUT/'identity.json'),
        score_source=art(__file__), sources=provenance+[art(current.OUT/'result.json'),art(current.OUT/'audit.json')],
        scores=scores, cv={c:float(np.mean([v[c] for v in scores.values()])) for c in names},paired=paired,
        prediction=art(OUT/'scored.parquet'),all_v77_stages_replayed_exact=True,
        all_before_quality_changes_within_fixed_gate=True,
        interpretation='Fixed same-laptop controls; final day correction can redistribute within Rome. No production or upload.')
    write(OUT/'result.json',result); print(dict(cv=result['cv'],paired=paired),flush=True)


def audit():
    identity=experiment.guard(); result=read(OUT/'result.json'); verify(result['prediction'])
    assert result['score_source']==art(__file__)
    assert not (OUT/'score_audit.json').exists(),'Preserve completed audit'
    with duckdb.connect(config={'threads':1}) as con:
        for name,path in [('p',OUT/'scored.parquet'),('r',verify(identity['raw'])),('v',current.OUT/'scored.parquet')]:
            con.read_parquet(str(path)).create_view(name)
        n,u,bad=con.execute('''select count(*),count(distinct p.mvt_id),sum(case when
            p.y is distinct from r.y or p.month is distinct from r.month or p.day is distinct from r.day
            or p.selected is distinct from v.combined or p.selected_before is distinct from v.combined_before
            or p.selected_shift is distinct from v.combined_shift
            or p.gate != (r.airport='LIRF' and not r.has_id and r.gap>3600 and r.gap<14400)
            then 1 else 0 end) from p join r using(mvt_id) join v using(mvt_id)''').fetchone()
        assert n==u==344419 and bad==0
        for name in result['cv']:
            values=con.execute(f'select month,sqrt(avg(pow({name}-y,2))) from p group by month').fetchall()
            for month,value in values: np.testing.assert_allclose(value,result['scores'][month][name],atol=1e-8,rtol=0)
            np.testing.assert_allclose(np.mean([v for _,v in values]),result['cv'][name],atol=1e-8,rtol=0)
        for arm in experiment.ARMS:
            bad=con.execute(f'''select count(*) from p join r using(mvt_id) where
                (not p.gate and p.{arm}_before != p.selected_before)
                or (r.airport!='LIRF' and p.{arm} != p.selected)''').fetchone()[0]
            assert bad==0
    write(OUT/'score_audit.json',dict(status='pass',rows=n,source=art(__file__),result=art(OUT/'result.json'),
        identity=art(OUT/'identity.json'),independent_sql_truth_gate_controls_and_scores=True))
    print('Full-row score audit passed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['audit_models','score','audit']);a=p.parse_args()
    with threadpool_limits(1):globals()[a.stage]()
