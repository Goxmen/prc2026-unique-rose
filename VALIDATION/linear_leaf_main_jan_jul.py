"""Fixed original184-column constant-versus-linear LightGBM comparison.

Four same-host fits; original ordinary rows and uncapped targets. No search,
early stopping, production fit, or upload. The separate v79 transport is frozen
before either arm is scored. Models are committed before held inference.
"""
import argparse
import gc
import hashlib
from pathlib import Path
import platform
import time

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import linear_leaf_preflight as scaling
import public_daily_taxi_lgb as base
import daily_main_refinements_cv as stack
import joint_public_moments_cv as v78
from records import write_json

ROOT = scaling.ROOT
OUT = ROOT/'VALIDATION/runs/v3_20261012_linear_leaf_main'
MANIFEST = OUT/'identity.json'
FROZEN = OUT/'design.json'
ARMS = ['constant', 'candidate']
art, read, verify = base.art, base.read, base.verify
DESIGN = dict(
    features='unchanged original184; no source-phase features', columns=184,
    rows_targets='exact permitted ordinary-training population and original uncapped targets',
    normalization='training-only finite mean/population SD; constant scale1; exact NaNs/categories/index retained',
    control='new constant-leaf LightGBM on the same standardized inputs',
    candidate_change={'linear_tree': True, 'linear_lambda': 1.0},
    folds=['2025-01', '2025-07'], arms=ARMS,
    seed=42, trial='LB1', leaves=127, rounds=10000, threads=4,
    other_settings='all retained original184-control parameters unchanged',
    order='January constant then candidate; July constant then candidate; same desktop',
    primary='candidate minus matched standardized constant through full exactv78 correction stack',
    actual_selected='both arms separately versus retained exactv78',
    secondary_v79='separate predeclared adapter: round_even(max(0,v79+arm_on_v78-exactv78)); coefficient1',
    predictions='nonnegative float32 as parent; save model first; allheld in-memory/reloaded equality',
    no_hpo=True, no_stopping_search=True, no_weight_search=True,
    no_production=True, no_upload=True,
    caveats=['extensively reused Jan/Jul development; not independent confirmation',
             'competitor gains came from a different pipeline and are not transferable evidence',
             'missing numeric path features can force constant-leaf fallback'],
)


def array_sha(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes()).hexdigest()


def params_for(control, arm):
    assert arm in ARMS
    p = dict(control, linear_tree=arm == 'candidate')
    assert p['seed'] == 42 and p['num_leaves'] == 127 and p['num_threads'] == 4
    assert 'linear_lambda' not in control and not control.get('linear_tree', False)
    if arm == 'candidate':
        p['linear_lambda'] = 1.
    return p


def freeze():
    """Small source/design freeze; full input verification occurs in guard."""
    assert not list(OUT.glob('models/*')) and not (OUT/'result.json').exists()
    paths = [Path(__file__), Path(scaling.__file__),
        ROOT/'VALIDATION/linear_leaf_v79_transport.py',
        Path(base.__file__), Path(stack.__file__), Path(v78.__file__),
        base.OUT/'identity.json', base.OUT/'audit.json',
        stack.OUT/'identity.json', stack.OUT/'audit.json',
        v78.OUT/'identity.json', v78.OUT/'audit.json']
    value = dict(design=DESIGN, sources=[art(p) for p in paths])
    OUT.mkdir(parents=True, exist_ok=True)
    if FROZEN.exists():
        assert read(FROZEN) == value
    else:
        write_json(FROZEN, value)
    print('Linear comparison frozen before model fitting or held scoring.', flush=True)


def guard():
    frozen = read(FROZEN); assert frozen['design'] == DESIGN
    for record in frozen['sources']:
        verify(record)
    for folder in [base.OUT, stack.OUT, v78.OUT]:
        assert read(folder/'audit.json')['status'] == 'pass'
        for record in read(folder/'identity.json').get('sources', []):
            verify(record)
    # The separate transport registry must already be frozen before training.
    transport = read(OUT/'v79'/'design.json')
    assert transport['primary_design'] == art(FROZEN)
    for record in transport['sources']:
        verify(record)
    paths = [FROZEN, OUT/'v79'/'design.json', Path(__file__), Path(scaling.__file__),
        Path(base.__file__), Path(stack.__file__), Path(v78.__file__),
        Path(base.parent.__file__), Path(base.main.__file__), Path(stack.daily.__file__),
        Path(stack.daily.constraints.__file__), Path(stack.daily.constraints.exclusion.__file__),
        Path(stack.joint.neural.__file__), ROOT/'VALIDATION/modeling.py', ROOT/'VALIDATION/records.py',
        base.OUT/'identity.json', base.OUT/'audit.json', base.OUT/'result.json',
        stack.OUT/'identity.json', stack.OUT/'audit.json', stack.OUT/'scored.parquet',
        v78.OUT/'identity.json', v78.OUT/'audit.json', v78.OUT/'scored.parquet', v78.OUT/'result.json',
        base.main.features.OUT/'features2025.parquet', base.features.OUT/'features2025.parquet',
        base.main.features.base.OUT/'rows2025.parquet', stack.daily.CACHE,
        stack.daily.SOURCE/'daily_runway.parquet', stack.daily.OUT/'parameters.json',
        stack.daily.constraints.exclusion.OUT/'matches2025.parquet',
        stack.quality.OUT/'scored.parquet', stack.rounded.OUT/'scored.parquet',
        stack.joint.neural.cat.OUT/'scored.parquet']
    for month in base.parent.ts.SPLITS:
        paths.extend([base.parent.input_cache.input_path(month),
            base.OUT/'predictions'/f'{month}_s42.json', base.OUT/'predictions'/f'{month}_s42.parquet',
            base.OUT/'models'/f'{month}_s42.txt', verify(stack.cv.completed(month)['prediction']),
            stack.rounded.original.airport.OUT/'predictions'/f'{month}_s42.parquet'])
    identity = read(stack.joint.neural.single.MANIFEST)
    for month in base.parent.ts.SPLITS:
        for k in range(3):
            paths.append(verify(stack.joint.neural.completed(month, k, identity)['prediction']))
    paths.append(stack.joint.neural.cat.r20.physical.replay.combo.sequence.CALLS_PRED)
    value = dict(design=DESIGN, sources=[art(p) for p in dict.fromkeys(paths)])
    if MANIFEST.exists():
        assert read(MANIFEST) == value
    else:
        assert not list(OUT.glob('models/*'))
        write_json(MANIFEST, value)
    return value


def inputs(month, tables, create_scaler=False):
    d = base.inputs(month, *tables); control = base.completed(month)
    assert control and len(control['columns']) == 184 and control['rounds'] == 10000
    assert control['machine'].lower() == platform.node().lower(), 'Same-host comparison'
    assert list(d['X']) == list(d['V']) == control['columns']
    for arm in ARMS:
        params_for(control['params'], arm)
    stats = scaling.scaler_fit(d['X'], control['categoricals'])
    record = dict(identity=art(MANIFEST), month=month, scaler=stats,
        columns=list(d['X']), categoricals=control['categoricals'],
        categories={c:list(d['X'][c].cat.categories) for c in control['categoricals']},
        train_rows=len(d['X']), held_rows=len(d['V']),
        train_ids_sha256=array_sha(d['train_ids']), held_ids_sha256=array_sha(d['ids']),
        targets_sha256=array_sha(d['y']), input=control['input'])
    path = OUT/'scalers'/f'{month}.json'
    if create_scaler:
        path.parent.mkdir(exist_ok=True)
        if path.exists():
            assert read(path) == record, 'Preserve exact partial preflight preprocessing'
        else:
            write_json(path, record)
    else:
        assert read(path) == record, 'Exact permitted rows/targets/categories/training-only scaler'
    for name in ['X', 'V']:
        d[name] = scaling.scaler_apply(d[name], stats)
    d['control'] = control; d['scaler'] = art(path)
    return d


def preflight():
    guard(); assert not (OUT/'preflight.json').exists()
    tables = base.tables(); checks = []
    for month in base.parent.ts.SPLITS:
        d = inputs(month, tables, create_scaler=True)
        checks.append(dict(month=month, train_rows=len(d['X']), held_rows=len(d['V']),
            scaler=d['scaler'], columns=len(d['X'].columns),
            exact_original_memberships_and_uncapped_targets=True,
            train_only_scaling=True, exact_nan_masks_categories_and_order=True))
        del d; gc.collect()
    write_json(OUT/'preflight.json', dict(status='pass', identity=art(MANIFEST), checks=checks))
    print('Both original184 input/scaler views audited; no fit or held score.', flush=True)


def model_record(month, arm):
    model = OUT/'models'/f'{month}_{arm}.txt'; metadata = model.with_suffix('.json')
    if not metadata.exists():
        assert not model.exists(), 'Unrecorded model; inspect partial save before retry'
        return None
    record = read(metadata)
    assert record['identity'] == art(MANIFEST) and record['month'] == month and record['arm'] == arm
    original = base.completed(month)
    assert record['rounds'] == 10000 and record['columns'] == original['columns']
    assert record['params'] == params_for(original['params'], arm)
    assert record['model_saved_before_held_inference']
    verify(record['model']); verify(record['scaler'])
    return record


def completed(month, arm):
    path = OUT/'predictions'/f'{month}_{arm}.parquet'; meta = path.with_suffix('.json')
    if not meta.exists():
        assert not path.exists(), 'Unrecorded prediction; inspect partial output before retry'
        return None
    record = read(meta); staged = model_record(month, arm)
    assert record['identity'] == art(MANIFEST) and staged
    assert record['month'] == month and record['arm'] == arm and record['all_held_reload_exact']
    assert record['scaler'] == staged['scaler'] and record['held_rows'] == staged['held_rows']
    assert record['staged_model'] == art(OUT/'models'/f'{month}_{arm}.json')
    verify(record['prediction']); return record


def fit(max_new_records=None):
    assert max_new_records is None or max_new_records >= 1
    newly_completed = 0
    guard(); assert read(OUT/'preflight.json')['identity'] == art(MANIFEST)
    for name in ['models', 'predictions']:
        (OUT/name).mkdir(exist_ok=True)
    tables = base.tables()
    for month in base.parent.ts.SPLITS:
        if all(completed(month, arm) for arm in ARMS):
            continue
        d = inputs(month, tables); control = d['control']
        for arm in ARMS:
            if completed(month, arm):
                continue
            params = params_for(control['params'], arm); staged = model_record(month, arm)
            model_path = OUT/'models'/f'{month}_{arm}.txt'
            if staged:
                assert staged['params'] == params and staged['scaler'] == d['scaler']
                model = lgb.Booster(model_file=str(verify(staged['model'])))
                print(month, arm, 'resuming saved model; no refit', flush=True)
            else:
                started = time.monotonic()
                print(month, arm, len(d['X']), 'rows,184 inputs,LB1/10000', flush=True)
                dataset = lgb.Dataset(d['X'], d['y'], params=params,
                    categorical_feature=control['categoricals'])
                def progress(env):
                    if (env.iteration+1)%1000 == 0:
                        print(month, arm, env.iteration+1, 'seconds',
                            round(time.monotonic()-started, 1), flush=True)
                model = lgb.train(params, dataset, num_boost_round=10000, callbacks=[progress])
                assert model.current_iteration() == 10000
                model.save_model(str(model_path))
                staged = dict(identity=art(MANIFEST), month=month, arm=arm, model=art(model_path),
                    scaler=d['scaler'], params=params, rounds=10000, columns=list(d['X']),
                    categoricals=control['categoricals'], train_rows=len(d['X']), held_rows=len(d['V']),
                    model_saved_before_held_inference=True, machine=platform.node(),
                    fit_minutes=(time.monotonic()-started)/60)
                write_json(model_path.with_suffix('.json'), staged)
                del dataset
            assert model.current_iteration() == 10000
            inference_started = time.monotonic()
            prediction = np.maximum(model.predict(d['V'], num_threads=4), 0).astype('float32')
            restored = lgb.Booster(model_file=str(model_path))
            replay = np.maximum(restored.predict(d['V'], num_threads=4), 0).astype('float32')
            np.testing.assert_array_equal(prediction, replay)
            assert np.isfinite(prediction).all() and (prediction >= 0).all()
            path = OUT/'predictions'/f'{month}_{arm}.parquet'
            pd.DataFrame(dict(mvt_id=d['ids'], pred=prediction)).to_parquet(path, index=False)
            write_json(path.with_suffix('.json'), dict(identity=art(MANIFEST),
                staged_model=art(model_path.with_suffix('.json')), prediction=art(path),
                month=month, arm=arm, all_held_reload_exact=True, scaler=d['scaler'],
                held_rows=len(prediction), inference_reload_minutes=(time.monotonic()-inference_started)/60))
            del model, restored, prediction, replay; gc.collect()
            newly_completed += 1
            if max_new_records is not None and newly_completed >= max_new_records:
                print('Operational pause after', newly_completed, 'new model+prediction records;',
                    'resume fit to continue the unchanged fold/arm order.', flush=True)
                del d; gc.collect()
                return
        del d; gc.collect()


def score():
    guard();assert not (OUT/'result.json').exists(),'Preserve completed score'
    daily,joint,rounded=stack.daily,stack.joint,stack.rounded
    neural=joint.neural;replay=neural.cat.r20.physical.replay
    sc,rome,_,trees=replay.baseline.baseline();identity=read(neural.single.MANIFEST)
    prior=pd.read_parquet(neural.cat.OUT/'scored.parquet').set_index('mvt_id')
    serialized=pd.read_parquet(rounded.OUT/'scored.parquet').set_index('mvt_id')
    selected=pd.read_parquet(stack.OUT/'scored.parquet').set_index('mvt_id')
    moment=pd.read_parquet(v78.OUT/'scored.parquet').set_index('mvt_id')
    quality=pd.read_parquet(stack.quality.OUT/'scored.parquet').set_index('mvt_id')
    raw=pd.read_parquet(daily.base.OUT/'rows2025.parquet').set_index('mvt_id')
    source,parameters=daily.public(),read(daily.OUT/'parameters.json')
    context=daily.rows().drop(columns='y').set_index('mvt_id')
    matches=pd.read_parquet(daily.constraints.exclusion.OUT/'matches2025.parquet');frames=[];cells=[]
    for month in base.parent.ts.SPLITS:
        d=sc.d[month];ids=d['ids'];records={arm:completed(month,arm) for arm in ARMS};assert all(records.values())
        calls=replay.values(replay.combo.sequence.CALLS_PRED,ids,'candidate_nn')
        physical=np.mean([replay.values(verify(neural.completed(month,k,identity)['prediction']),ids,'base') for k in range(3)],axis=0)
        old_main=replay.values(verify(base.completed(month)['prediction']),ids,'pred')
        matched_main=replay.values(verify(records['constant']['prediction']),ids,'pred')
        new_main=replay.values(verify(records['candidate']['prediction']),ids,'pred')
        airport=replay.values(rounded.original.airport.OUT/'predictions'/f'{month}_s42.parquet',ids,'pred')
        cat=replay.values(verify(stack.cv.completed(month)['prediction']),ids,'pred')
        probability=.5*prior.loc[ids,'newcat'].to_numpy()+.5*rome[month]
        permitted=matches[matches.donor_month.isin(daily.base.FOLDS[month]+[month])]
        outputs={}
        for arm,main_prediction in [('selected',old_main),('constant',matched_main),('candidate',new_main)]:
            ordinary=.71*.5*(calls+physical)+.29*np.mean([main_prediction,airport,cat],axis=0)
            direct=sc.final(month,ordinary,probability)
            start=serialized.loc[ids,'candidate_boundary'].to_numpy()+direct-prior.loc[ids,'control'].to_numpy()
            refined=daily.constraints.matched_bound(raw.loc[ids].reset_index(),start)
            refined,_=daily.constraints.exclusion.apply(ids,np.maximum(refined,0),permitted)
            refined=np.rint(np.maximum(refined,0))
            updated,cell,shift=daily.apply(context.loc[ids].reset_index(),refined,source,parameters[month])
            gated=np.where(quality.loc[ids,'quality_eligible'],updated,refined)
            pred77=daily.constrain(raw.loc[ids].reset_index(),ids,pd.Series(month,index=range(len(ids))),gated,matches)
            pred78=daily.constrain(raw.loc[ids].reset_index(),ids,pd.Series(month,index=range(len(ids))),
                pred77+.5*moment.loc[ids,'increment'].to_numpy(),matches)
            if arm=='selected':
                for got,col in [(refined,'combined_before'),(shift,'combined_shift'),(pred77,'combined')]:
                    np.testing.assert_array_equal(got,selected.loc[ids,col])
                np.testing.assert_array_equal(pred77,moment.loc[ids,'control'])
                np.testing.assert_array_equal(pred78,moment.loc[ids,'candidate'])
            outputs.update({arm:pred78,arm+'_v77':pred77,arm+'_before':refined,arm+'_shift':shift})
            cells.append(cell.assign(fold=month,arm=arm))
        frames.append(pd.DataFrame(dict(mvt_id=ids,month=month,day=d['day'],y=d['y'],
            airport=raw.loc[ids,'airport'].to_numpy(),increment=moment.loc[ids,'increment'].to_numpy(),**outputs)))
    frame=pd.concat(frames,ignore_index=True);assert len(frame)==344419 and frame.mvt_id.is_unique
    columns=['selected','constant','candidate']
    scores={m:{c:float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in columns} for m,g in frame.groupby('month')}
    paired={name:base.parent.ts.paired_bootstrap(frame.day,frame.month,(frame[a]-frame.y)**2,(frame[b]-frame.y)**2)
            for name,a,b in [('matched','constant','candidate'),('candidate_vs_selected','selected','candidate'),('constant_vs_selected','selected','constant')]}
    frame.to_parquet(OUT/'scored.parquet',index=False)
    pd.concat(cells,ignore_index=True).to_parquet(OUT/'cells.parquet',index=False)
    result=dict(design=DESIGN,identity=art(MANIFEST),scores=scores,
        cv={c:float(np.mean([s[c] for s in scores.values()])) for c in columns},paired=paired,
        exact_v77_and_v78_controls=True,rows=len(frame),
        model_records=[art(OUT/'predictions'/f'{month}_{arm}.json') for month in base.parent.ts.SPLITS for arm in ARMS],
        artifacts=[art(OUT/'scored.parquet'),art(OUT/'cells.parquet')])
    write_json(OUT/'result.json',result);print(dict(cv=result['cv'],paired=paired),flush=True)


def audit():
    guard();result=read(OUT/'result.json')
    for a in result['artifacts']+result['model_records']:verify(a)
    for month in base.parent.ts.SPLITS:
        for arm in ARMS:assert completed(month,arm) is not None
    with duckdb.connect(config={'threads':1}) as con:
        con.execute("SET TimeZone='UTC'")
        for name,path in [('p',OUT/'scored.parquet'),('r',stack.daily.base.OUT/'rows2025.parquet'),
            ('c',v78.OUT/'scored.parquet'),('v',stack.OUT/'scored.parquet')]:con.read_parquet(str(path)).create_view(name)
        n,u,bad=con.execute('''SELECT count(*),count(DISTINCT p.mvt_id),sum(CASE WHEN
          p.y IS DISTINCT FROM r.y OR p.month IS DISTINCT FROM r.month OR p.day IS DISTINCT FROM r.day
          OR p.airport IS DISTINCT FROM r.airport OR p.selected IS DISTINCT FROM c.candidate
          OR p.increment IS DISTINCT FROM c.increment OR p.selected_v77 IS DISTINCT FROM v.combined
          OR p.selected_before IS DISTINCT FROM v.combined_before OR p.selected_shift IS DISTINCT FROM v.combined_shift
          THEN 1 ELSE 0 END) FROM p JOIN r USING(mvt_id) JOIN c USING(mvt_id) JOIN v USING(mvt_id)''').fetchone()
        assert n==u==344419 and bad==0
        for name in result['cv']:
            scored=con.execute(f'SELECT month,sqrt(avg(pow({name}-y,2))) FROM p GROUP BY month').fetchall()
            for month,value in scored:np.testing.assert_allclose(value,result['scores'][month][name],rtol=0,atol=1e-8)
            np.testing.assert_allclose(np.mean([v for _,v in scored]),result['cv'][name],rtol=0,atol=1e-8)
    f=pd.read_parquet(OUT/'scored.parquet');ids=f.mvt_id.to_numpy();daily=stack.daily
    raw=pd.read_parquet(daily.base.OUT/'rows2025.parquet').set_index('mvt_id').loc[ids].reset_index()
    quality=pd.read_parquet(stack.quality.OUT/'scored.parquet').set_index('mvt_id').loc[ids]
    context=daily.rows().drop(columns='y').set_index('mvt_id');source=daily.public()
    parameters=read(daily.OUT/'parameters.json');matches=pd.read_parquet(daily.constraints.exclusion.OUT/'matches2025.parquet')
    for arm in ['selected','constant','candidate']:
        for month in base.parent.ts.SPLITS:
            ix=np.flatnonzero(f.month.eq(month));_,_,shift=daily.apply(context.loc[ids[ix]].reset_index(),
                f[arm+'_before'].to_numpy()[ix],source,parameters[month])
            np.testing.assert_array_equal(shift,f[arm+'_shift'].to_numpy()[ix])
        pred=f[arm+'_before'].to_numpy()+np.where(quality.quality_eligible,f[arm+'_shift'],0.)
        np.testing.assert_array_equal(daily.constrain(raw,ids,f.month,pred,matches),f[arm+'_v77'])
        np.testing.assert_array_equal(daily.constrain(raw,ids,f.month,f[arm+'_v77'].to_numpy()+.5*f.increment.to_numpy(),matches),f[arm])
    write_json(OUT/'audit.json',dict(status='pass',rows=n,identity=art(MANIFEST),result=art(OUT/'result.json'),
        original_labels_and_all_rows_retained=True,all_parent_controls_exact=True,
        independent_sql_scores=True,all_daily_shifts_and_v78_constraints_replayed=True,source=art(__file__)))
    print('All344419 rows, original labels, exactv77/v78 controls and final stages audited',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'guard', 'preflight', 'fit', 'score', 'audit'])
    parser.add_argument('--max-new-records', type=int,
        help='Fit-stage operational pause after N newly completed model+prediction records; resume skips completed records.')
    args = parser.parse_args()
    if args.max_new_records is not None and (args.stage != 'fit' or args.max_new_records < 1):
        parser.error('--max-new-records requires fit and a positive integer')
    with threadpool_limits(1):
        if args.stage == 'fit':
            fit(max_new_records=args.max_new_records)
        else:
            globals()[args.stage]()
