"""Fixed single-seed comparison of a wider physical track encoder.

Reuse the audited public single as the matched control. Keep its original
156-column encoder, twelve public inputs, epochs and complete training bytecode.
Only the model factory changes. No production or upload is selected here.
"""
import argparse
import importlib.util
from pathlib import Path
import types

import duckdb
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import nn_public_scratch_jan_jul as single
import nn_track_wide as model_type
import rome_public_delay_cat as cat
from records import write_json, sha256

ROOT = single.ROOT
OUT = ROOT / 'VALIDATION/runs/v3_20261011_nn_track_wide'
MANIFEST = OUT / 'identity.json'
STOP, art, verify, read = single.STOP, single.artifact, single.verify, single.shared.ts.read
DESIGN = dict(single.DESIGN, variant='public', change='double the physical track encoder width32 to64',
    model=model_type.SETTINGS, primary='same-machine public single versus new single, actual public Cat3/R20 stack',
    selected_reference='public three-member pool; different pool size, diagnostic only',
    no_width_search=True, no_epoch_search=True,
    scope='One fixed capacity change; original physical features, tabular/public branches and all training settings retained')


def clone(function, **namespace):
    return types.FunctionType(function.__code__, {**function.__globals__, **namespace},
        function.__name__, function.__defaults__, function.__closure__)


def training_function():
    function = clone(single.base.train, OUT=OUT / 'runtime', prototype=model_type)
    assert function.__code__ is single.base.train.__code__
    assert single.base.train.__globals__['prototype'] is single.base.prototype
    return function


def paths(month):
    return OUT / 'networks' / f'{month}_s42.parquet', OUT / 'models' / f'{month}_s42.pt', OUT / 'networks' / f'{month}_s42.json'


def completed(month, identity):
    return clone(single.completed, OUT=OUT, MANIFEST=MANIFEST, DESIGN=DESIGN, paths=paths)(month, identity)


def control(month, identity):
    expected = dict(single.DESIGN, variant='public')
    assert identity['design'] == expected
    return clone(single.completed, DESIGN=expected)(month, identity)


def guard():
    parent = read(single.MANIFEST)
    assert parent['source'] == art(single.__file__)
    for source in parent['helpers']:
        verify(source)
    for field in ['public_features', 'feature_audit', 'metadata', 'metadata_values']:
        verify(parent[field])
    assert read(single.OUT / 'audit.json')['status'] == 'pass'
    controls = {}
    for month in STOP:
        record = control(month, parent)
        assert record is not None
        controls[month] = art(single.paths(month)[2])
    training_function()
    value = dict(design=DESIGN, source=art(__file__), cache=parent['cache'], folds=parent['folds'],
        public_features=parent['public_features'], feature_audit=parent['feature_audit'],
        metadata=parent['metadata'], metadata_values=parent['metadata_values'],
        control_identity=art(single.MANIFEST), controls=controls, control_audit=art(single.OUT / 'audit.json'),
        control_score=art(single.OUT / 'scored.parquet'), control_result=art(single.OUT / 'score.json'),
        selected_result=parent['selected_result'], selected_errors=parent['selected_errors'],
        helpers=[art(m.__file__) for m in [single, model_type, model_type.parent, model_type.physical,
            single.base, single.adapter, cat, cat.r20]],
        retained_auditor=art(ROOT / 'VALIDATION/tools/audit_public_neural.py'),
        rome_predictions=art(cat.OUT / 'scored.parquet'),
        pool3_predictions=art(ROOT / 'VALIDATION/runs/v3_20261011_nn_public_pool3/scored.parquet'),
        checks=model_type.check('cpu'), training_bytecode_exact=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists():
        assert read(MANIFEST) == value
    else:
        assert not list(OUT.glob('**/*.pt')) and not list(OUT.glob('**/*.parquet'))
        write_json(MANIFEST, value)
    return value


def fit():
    identity = guard()
    single.base.guard(single.base.cache_identity())
    function = clone(single.fit, OUT=OUT, MANIFEST=MANIFEST, DESIGN=DESIGN,
        airport_tune=model_type, training_function=training_function, paths=paths, completed=completed)
    function(list(STOP), identity)


def audit_models():
    guard()
    assert not (OUT / 'audit.json').exists(), 'Retain completed model audit'
    # Reuse the unchanged, previously reviewed auditor with an isolated facade.
    path = ROOT / 'VALIDATION/tools/audit_public_neural.py'
    spec = importlib.util.spec_from_file_location('retained_public_auditor', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    facade = types.SimpleNamespace(**{**vars(single), 'OUT': OUT, 'MANIFEST': MANIFEST,
        'DESIGN': DESIGN, 'airport_tune': model_type, 'completed': completed})
    clone(module.audit, single=facade)(False)
    assert single.OUT != OUT and single.airport_tune is model_type.parent


def score():
    identity = guard()
    assert read(OUT / 'audit.json')['status'] == 'pass'
    assert not (OUT / 'result.json').exists(), 'Preserve completed score'
    parent = read(verify(identity['control_identity']))
    replay = cat.r20.physical.replay
    sc, rome, _, trees = replay.baseline.baseline()
    prior = pd.read_parquet(verify(identity['rome_predictions'])).set_index('mvt_id')
    original_errors = pd.read_parquet(verify(identity['control_score'])).set_index('mvt_id')
    pool = pd.read_parquet(verify(identity['pool3_predictions'])).set_index('mvt_id')
    raw = pd.read_parquet(cat.r20.features.base.OUT / 'rows2025.parquet').set_index('mvt_id')
    matches = pd.read_parquet(cat.r20.constraints.exclusion.OUT / 'matches2025.parquet')
    frames = []
    for month in STOP:
        d = sc.d[month]
        ids = d['ids']
        old = pd.read_parquet(verify(control(month, parent)['prediction']))
        new = pd.read_parquet(verify(completed(month, identity)['prediction']))
        for frame in [old, new]:
            np.testing.assert_array_equal(frame.mvt_id, ids)
        calls = replay.values(replay.combo.sequence.CALLS_PRED, ids, 'candidate_nn')
        main = replay.values(replay.combo.parent.control.candidate.OUT / 'predictions' / f'{month}_s42.parquet', ids, 'pred')
        tree = np.mean([main, trees['airport'][42][month], trees['catboost'][42][month]], axis=0)
        original_probability = .5*cat.parent.probabilities(month, ids)+.5*rome[month]
        members = {single.shared.hpo.MEMBER_OF[part]: trees[part][42][month] for part in trees}
        members[single.shared.hpo.MEMBER_OF['lgb']] = main
        existing_error = sc.se(month, 42, nn=.5*(calls+old.base.to_numpy(float)), p=original_probability, **members)
        np.testing.assert_array_equal(existing_error, original_errors.loc[ids, 'public_single_mix'])
        source = prior.loc[ids]
        probability = .5*source.newcat.to_numpy()+.5*rome[month]
        permitted = matches[matches.donor_month.isin(cat.r20.features.base.FOLDS[month]+[month])]
        values = {}
        for name, neural in [('control', old.base.to_numpy(float)), ('candidate', new.base.to_numpy(float))]:
            pred = sc.final(month, .71*.5*(calls+neural)+.29*tree, probability)
            transferred = cat.r20.constraints.matched_bound(raw.loc[ids].reset_index(),
                source.v74.to_numpy()+pred-source.control.to_numpy())
            transferred, _ = cat.r20.constraints.exclusion.apply(ids, np.maximum(transferred, 0), permitted)
            values[name] = pred
            values[name+'_transfer'] = transferred
        frames.append(pd.DataFrame(dict(mvt_id=ids, month=month, day=d['day'], y=d['y'],
            pool3_transfer=pool.loc[ids, 'transferred'].to_numpy(), **values)))
    f = pd.concat(frames, ignore_index=True)
    columns = ['control', 'candidate', 'control_transfer', 'candidate_transfer', 'pool3_transfer']
    assert len(f) == 344419 and f.mvt_id.is_unique and np.isfinite(f[columns]).all().all()
    scores = {m: {c: float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in columns} for m, g in f.groupby('month')}
    pairs = [('control', 'candidate'), ('control_transfer', 'candidate_transfer'), ('pool3_transfer', 'candidate_transfer')]
    f.to_parquet(OUT / 'scored.parquet', index=False)
    result = dict(design=DESIGN, scores=scores, cv={c: float(np.mean([s[c] for s in scores.values()])) for c in columns},
        paired={f'{b}_vs_{a}': single.shared.ts.paired_bootstrap(f.day, f.month, (f[a]-f.y)**2, (f[b]-f.y)**2)
            for a, b in pairs}, identity=art(MANIFEST), predictions=art(OUT / 'scored.parquet'),
        original_public_single_squared_errors_exact=True,
        interpretation='Matched public single is primary; selected pool3 is a different-size reference; no production selected')
    write_json(OUT / 'result.json', result)
    print(result, flush=True)


def audit_scores():
    guard()
    assert not (OUT / 'score_audit.json').exists(), 'Preserve completed score audit'
    result = read(OUT / 'result.json')
    assert result['identity'] == art(MANIFEST) and result['predictions'] == art(OUT / 'scored.parquet')
    with duckdb.connect(config={'threads': 1}) as con:
        con.read_parquet(str(OUT / 'scored.parquet')).create_view('candidate')
        con.read_parquet(str(cat.r20.features.base.OUT / 'rows2025.parquet')).create_view('truth')
        count, unique, errors = con.execute('''select count(*),count(distinct c.mvt_id),
            sum(case when c.y is distinct from t.y or c.day is distinct from t.day or
            c.month is distinct from t.month then 1 else 0 end)
            from candidate c join truth t using(mvt_id)''').fetchone()
        assert count == unique == 344419 and errors == 0
        for name in result['cv']:
            metrics = con.execute(f'select month,sqrt(avg(pow({name}-y,2))) as value from candidate group by month').fetchall()
            for month, value in metrics:
                np.testing.assert_allclose(value, result['scores'][month][name], atol=1e-8, rtol=0)
            np.testing.assert_allclose(np.mean([v for m, v in metrics]), result['cv'][name], atol=1e-8, rtol=0)
    value = dict(status='pass', rows=count, truth_and_calendar_exact=True, independent_sql_scores=True,
        source=art(__file__), identity=art(MANIFEST), result=art(OUT / 'result.json'), predictions=art(OUT / 'scored.parquet'))
    write_json(OUT / 'score_audit.json', value)
    print(value, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['guard', 'fit', 'audit_models', 'score', 'audit_scores'])
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(2):
        globals()[args.stage]()
