"""One fixed reconstruction of the absent v30 model pair and its descendants.

No retained fit is repeated. All four original integer vectors must replay
exactly before final inference. Stages freeze, prepare, fit and infer are
separate so that prepared inputs can be reviewed before either missing fit.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline')]
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '8')

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

from common import TRAIN_GLOB
import dayshift_hedge
import overnight_models
import overnight_run
import plan_window
import record_fixes
from records import runtime_identity, sha256, write_json
import rome_hybrid
import rome_signals as signals
import stand_day
import stand_day_run
import stand_day_submit as original
import weather

OUT = ROOT / 'VALIDATION/runs/v3_20261012_final_missing_stand_reconstruct'
INPUTS = ROOT / 'VALIDATION/runs/v3_20261008_final_submit'
RUN_NAMES = {30: 'v3_20260918T151436598224Z', 33: 'v3_20260918T181924599658Z',
             34: 'v3_20260919T104157403241Z', 37: 'v3_20260919T112057144323Z'}
VERSIONS = tuple(RUN_NAMES)
DESIGN = dict(purpose='Recover two absent original v30 boosters and deterministic v33/v34/v37 descendants',
    seed=42, ordinary_rounds=800, classifier_rounds=300, threads=8, train_rows=2084659,
    freeze='2025-12-31', kind='clock', late=True, dayshift=False,
    thread_provenance='Original validation and unchanged submission CLI default are 8; full-year CLI override was not recorded',
    success='All 344841 original integers for each of v30/v33/v34/v37 must match before final inference',
    failure='Save all models and differences; stop on mismatch, no thread/seed/library sweep or silent substitution',
    scope='Missing-model reconstruction, no CV refit, new weights, new version or upload',
    inference='Reuse independently audited original/final four-month input packets; no repeated retained-model inference')


def read(path):
    return json.loads(Path(path).read_text())


def art(path):
    path = Path(path)
    return dict(path=str(path.relative_to(ROOT)), sha256=sha256(path))


def manifests():
    return {v: read(ROOT / 'VALIDATION/runs' / name / f'submission_v{v}.json')
            for v, name in RUN_NAMES.items()}


def freeze():
    assert not (OUT / 'identity.json').exists()
    recs = manifests()
    paths = {Path(__file__), ROOT / 'VALIDATION/records.py'}
    for v, manifest in recs.items():
        folder = ROOT / 'VALIDATION/runs' / RUN_NAMES[v]
        paths.update([folder / f'submission_v{v}.json', folder / 'result.json'])
        result = read(folder / 'result.json')
        for signature in [manifest['validation_signature'], manifest['preparation_source'], result['signature']]:
            for rel, digest in signature.items():
                assert sha256(ROOT / rel) == digest, rel
                paths.add(ROOT / rel)
        for entry in manifest['data_identity']:
            path = weather.METAR if entry['file'] == 'metar.parquet' else ROOT / 'data' / entry['file']
            assert sha256(path) == entry['sha256'], path
            paths.add(path)
    cfg = read(ROOT / 'VALIDATION/runs' / RUN_NAMES[30] / 'result.json')['config']
    assert cfg['threads'] == DESIGN['threads'] == cfg['prediction_threads']
    assert cfg['rounds'] == DESIGN['ordinary_rounds']
    assert cfg['ordinaries']['stand'] == list(original.ORDINARY)
    assert cfg['classifiers']['stand'] == list(original.CLASSIFIER)
    assert recs[30]['ordinary_extra'] == list(original.ORDINARY)
    assert recs[30]['classifier_extra'] == list(original.CLASSIFIER)
    cache = ROOT / 'data/derived/scored_submission_cache_20261011/manifest.json'
    refs = read(cache); paths.add(cache)
    for v in VERSIONS:
        path = ROOT / refs[str(v)]['file']
        assert sha256(path) == refs[str(v)]['sha256']; paths.add(path)
    for view in ['old', 'final']:
        path = INPUTS / 'work' / f'inputs_{view}.pkl'
        receipt = INPUTS / f'inputs_{view}.json'
        assert sha256(path) == read(receipt)['sha256']
        paths.update([path, receipt])
    for name in ['submitting.parquet', 'final_submitting.parquet', 'final_ranking.parquet']:
        paths.add(ROOT / 'data' / name)
    paths.add(ROOT / 'VALIDATION/runs/overnight_20260917/models/refit_2025__control.train_ids.parquet')
    OUT.mkdir(parents=True, exist_ok=True)
    for path in sorted(paths):
        if path.suffix == '.py':
            target = OUT / 'source' / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
    model_config = dict(ordinary=overnight_models.model_params('control', 42, 8),
        classifier=dict(rome_hybrid.CLASSIFIER_PARAMS, seed=42, bagging_seed=43,
                        feature_fraction_seed=44, num_threads=8),
        ordinary_features=rome_hybrid.regressor_features('clock', original.ORDINARY),
        classifier_features=rome_hybrid.regressor_features('clock', original.CLASSIFIER))
    write_json(OUT / 'identity.json', dict(design=DESIGN, sources=[art(p) for p in sorted(paths)],
        config=cfg, model_config=model_config, parameters=recs[30]['model']['parameters'],
        dayshift_history=recs[37]['history'], runtime=runtime_identity(),
        original_model_hashes=recs[30]['model']['artifacts'],
        inference_packets_previously_audited=True, source_bytes_match_historical_records=True))
    print(json.dumps(dict(stage='frozen', sources=len(paths), models_fitted=0,
        ordinary_features=len(model_config['ordinary_features']), classifier_features=len(model_config['classifier_features']))))


def guard():
    identity = read(OUT / 'identity.json')
    assert identity['design'] == DESIGN
    for item in identity['sources']:
        assert sha256(ROOT / item['path']) == item['sha256'], item['path']
    assert runtime_identity()['versions'] == identity['runtime']['versions']
    return identity


def independent_preflight(X, y, raw, arr):
    """Raw SQL target membership, model routing, basic inputs and stand witnesses."""
    ids_path = ROOT / 'VALIDATION/runs/overnight_20260917/models/refit_2025__control.train_ids.parquet'
    np.testing.assert_array_equal(pd.read_parquet(ids_path).mvt_id, raw.mvt_id)
    assert len(raw) == DESIGN['train_rows'] and X.index.equals(raw.index)
    np.testing.assert_array_equal(y, raw.taxi_sec.to_numpy(float))
    with duckdb.connect(config={'threads': 2, 'memory_limit': '1GB'}) as con:
        con.execute("SET TimeZone='UTC'")
        targets = con.execute("SELECT MVT_ID_mvt mvt_id,TAXITIME_SEC_mvt taxi_sec FROM read_parquet(?) WHERE PHASE_mvt='DEP' AND TAXITIME_SEC_mvt BETWEEN 1 AND 200000", [TRAIN_GLOB]).df().set_index('mvt_id')
        assert targets.index.is_unique and set(targets.index) == set(raw.mvt_id)
        np.testing.assert_array_equal(targets.loc[raw.mvt_id, 'taxi_sec'], y)
        del targets
        measured = X.iloc[:, :18].copy()
        measured['mvt_id'] = raw.mvt_id.to_numpy()
        measured['substituted'] = rome_hybrid.substituted(X, y, raw)
        con.register('measured', measured); con.register('raw', raw)
        expected = dict(airport='r.ADEP_mvt', dest='r.ADES_mvt', aircraft_type='r.AIRCRAFT_TYPE_mvt',
            runway="r.ADEP_mvt||':'||r.RUNWAY_mvt", stand="r.ADEP_mvt||':'||r.STAND_mvt",
            flight_rule='r.FLIGHT_RULE_mvt', market_segment='r.MARKET_SEGMENT_flt',
            flight_type='r.FLIGHT_TYPE_flt', wake_cat='r.WK_TBL_CAT_flt', operator='r.AIRCRAFT_OPERATOR_flt',
            hour='hour(r.MVT_TIME_UTC_mvt)', dow='isodow(r.MVT_TIME_UTC_mvt)-1', month='month(r.MVT_TIME_UTC_mvt)',
            substituted="r.ADEP_mvt='LIRF' AND abs(epoch(r.MVT_TIME_UTC_mvt-r.SCHED_TIME_UTC_mvt)-r.taxi_sec)<=60")
        for name, col in [('sched', 'SCHED_TIME_UTC_mvt'), ('lobt', 'LOBT_flt'), ('iobt', 'IOBT_flt'),
                          ('eobt1', 'EOBT_1_flt'), ('aobt3', 'AOBT_3_flt')]:
            expected[f'mvt_minus_{name}_sec'] = f'epoch(r.MVT_TIME_UTC_mvt-r.{col})'
        expr = ' OR '.join(f'm."{key}" IS DISTINCT FROM ({value})' for key, value in expected.items())
        n, bad, copies, rome = con.execute(f"SELECT count(*),sum(CASE WHEN {expr} THEN 1 ELSE 0 END),sum(m.substituted::INT),sum((r.ADEP_mvt='LIRF')::INT) FROM measured m JOIN raw r USING(mvt_id)").fetchone()
        assert n == len(raw) and bad == 0, (n, bad)
        con.unregister('measured'); del measured
        # Independent stable duplicate handling: the last arrival at an equal
        # in-block instant wins, and its preceding arrival may have that same time.
        arrivals = arr.loc[:, stand_day.ARR_COLUMNS].copy()
        arrivals['arrival_order'] = np.arange(len(arrivals))
        con.register('arrivals', arrivals)
        airline = lambda col: f"CASE WHEN length({col})<3 THEN NULL WHEN regexp_full_match(substr({col},1,3),'[A-Za-z]{{3}}') THEN substr({col},1,3) ELSE substr({col},1,2) END"
        con.execute(f"""CREATE TEMP TABLE stand_arrivals AS
            WITH a AS (SELECT ADES_mvt||':'||STAND_mvt k,
                epoch_ns(MVT_TIME_UTC_mvt)+cast(round(taxi_sec*1000000000) AS BIGINT) inb,
                AIRCRAFT_TYPE_mvt typ,{airline('FLIGHT_mvt')} airline,arrival_order
                FROM arrivals WHERE isfinite(taxi_sec) AND MVT_TIME_UTC_mvt IS NOT NULL
                AND ADES_mvt IS NOT NULL AND STAND_mvt IS NOT NULL),
            b AS (SELECT *,lag(inb) OVER(PARTITION BY k ORDER BY inb,arrival_order) prev_inb FROM a)
            SELECT * FROM b QUALIFY row_number() OVER(PARTITION BY k,inb ORDER BY arrival_order DESC)=1""")
        con.unregister('arrivals'); del arrivals
        got = con.execute(f"""SELECT r.mvt_id,
            cast((cast(epoch_ns(r.MVT_TIME_UTC_mvt) AS DOUBLE)-cast(a.inb AS DOUBLE))/1e9 AS FLOAT) st_last_arr_sec,
            cast((cast(epoch_ns(r.MVT_TIME_UTC_mvt) AS DOUBLE)-cast(a.prev_inb AS DOUBLE))/1e9 AS FLOAT) st_prev_arr_sec,
            cast((r.AIRCRAFT_TYPE_mvt=a.typ) AS FLOAT) st_same_type,
            cast(({airline('r.FLIGHT_mvt')}=a.airline) AS FLOAT) st_same_airline,
            cast((cast(a.inb AS DOUBLE)-cast(epoch_ns(r.SCHED_TIME_UTC_mvt) AS DOUBLE))/1e9 AS FLOAT) st_inblock_minus_sched_sec
            FROM raw r ASOF LEFT JOIN stand_arrivals a
            ON r.ADEP_mvt||':'||r.STAND_mvt=a.k AND epoch_ns(r.MVT_TIME_UTC_mvt)>a.inb""").df().set_index('mvt_id')
        assert got.index.is_unique and len(got) == len(X)
        for name in stand_day.STAND_FEATURES:
            np.testing.assert_array_equal(got.loc[raw.mvt_id, name].to_numpy(), X[name].to_numpy(), err_msg=name)
    return dict(rows=len(X), ordinary_rows=int(len(X)-copies), classifier_rows=int(rome),
        substituted_rows=int(copies), raw_SQL_targets_membership_and_original_ID_order_exact=True,
        basic18_inputs_and_classifier_membership_independent_SQL_exact=True,
        all_five_stand_features_independent_SQL_exact=True,
        remaining_history_clock_weather_queue_handler_features_original_source_hash_verified=True,
        no_independent_all_value_claim_for_remaining_features=True)


def prepare():
    identity = guard()
    assert not (OUT / 'preflight.json').exists() and not (OUT / 'training.pkl').exists()
    started = time.monotonic()
    dep, arr = overnight_run.load_inputs(1.0, ranking=False)
    X, y, cats, raw = overnight_run.prepare_training(dep, arr, identity['config'])
    X = stand_day_run.add_inputs(X, raw, weather.load_reports(), dep, arr)
    parameters = signals.fit_parameters(X, y, raw)
    assert parameters == identity['parameters']
    history = dayshift_hedge.history(dep, dep.taxi_sec.to_numpy(float))
    assert history == identity['dayshift_history']
    checks = independent_preflight(X, y, raw, arr)
    for key in ['ordinary_features', 'classifier_features']:
        assert set(identity['model_config'][key]).issubset(X)
    pd.to_pickle(dict(X=X, y=y, raw=raw), OUT / 'training.pkl')
    write_json(OUT / 'categories.json', {k: v.tolist() for k, v in cats.items()})
    write_json(OUT / 'preflight.json', dict(checks, identity=art(OUT / 'identity.json'),
        training=art(OUT / 'training.pkl'), categories=art(OUT / 'categories.json'),
        prepared_columns=list(X), parameters=parameters, dayshift_history=history,
        model_config=identity['model_config'], minutes=(time.monotonic()-started)/60,
        fits_started=0, runtime=runtime_identity()))
    print(json.dumps(dict(stage='preflight_pass', **checks, minutes=(time.monotonic()-started)/60)), flush=True)


def fit():
    identity = guard(); preflight = read(OUT / 'preflight.json')
    assert preflight['identity'] == art(OUT / 'identity.json')
    assert preflight['training'] == art(OUT / 'training.pkl')
    packet = pd.read_pickle(OUT / 'training.pkl')
    X, y, raw = packet['X'], packet['y'], packet['raw']
    for part in ['ordinary', 'classifier']:
        model = OUT / f'{part}.txt'; receipt = OUT / f'{part}_trained.json'
        if receipt.exists():
            saved = read(receipt)
            assert saved['identity'] == art(OUT / 'identity.json') and saved['model'] == art(model)
            continue
        assert not model.exists(), 'Preserve any saved model lacking its receipt; never refit it'
        started = time.monotonic()
        if part == 'ordinary':
            booster = signals.fit_ordinary(X, y, raw, 'clock', 42, rounds=800, threads=8, extra=original.ORDINARY)
        else:
            booster = signals.fit_classifier(X, y, raw, 'clock', 42, threads=8, extra=original.CLASSIFIER)
        booster.save_model(str(model))
        assert booster.feature_name() == identity['model_config'][part+'_features']
        oldhash = manifests()[30]['model']['artifacts'][f'models/refit_2025__v30__42.{part}.txt']
        write_json(receipt, dict(identity=art(OUT / 'identity.json'), preflight=art(OUT / 'preflight.json'),
            model=art(model), model_text_hash_matches_original=sha256(model)==oldhash,
            features=booster.feature_name(), trees=booster.num_trees(), runtime=runtime_identity(),
            seconds=time.monotonic()-started, saved_before_inference=True))
        print(json.dumps(dict(stage='saved_model', part=part, seconds=time.monotonic()-started)), flush=True)


def infer():
    identity = guard(); assert not (OUT / 'result.json').exists()
    models = {}; reloads = {}
    for part in ['ordinary', 'classifier']:
        receipt = read(OUT / f'{part}_trained.json')
        assert receipt['identity'] == art(OUT / 'identity.json')
        assert receipt['model'] == art(OUT / f'{part}.txt')
        models[part] = lgb.Booster(model_file=str(OUT / f'{part}.txt'))
        reloads[part] = lgb.Booster(model_file=str(OUT / f'{part}.txt'))
    refs = read(ROOT / 'data/derived/scored_submission_cache_20261011/manifest.json')
    results = {}
    for view in ['old', 'final']:
        if (OUT / f'{view}.json').exists():
            results[view] = read(OUT / f'{view}.json')
            for rec in results[view].values():
                assert rec['prediction'] == art(ROOT / rec['prediction']['path'])
            if view == 'old':
                assert all(r['exact_original_integer_replay'] for r in results[view].values())
            continue
        packet = pd.read_pickle(INPUTS / 'work' / f'inputs_{view}.pkl')
        template = ROOT / 'data' / ('submitting.parquet' if view=='old' else 'final_submitting.parquet')
        ids = pq.read_table(template, columns=['MVT_ID_mvt'])['MVT_ID_mvt'].to_numpy()
        assert set(packet['rank_ids']) == set(ids)
        frames = {v: [] for v in VERSIONS}
        for month, item in packet['months'].items():
            V, raw = item['V'], item['raw']
            assert X_aligned(V, raw)
            args = dict(threads=8, extra=original.ORDINARY, classifier_extra=original.CLASSIFIER,
                late_mean=identity['parameters']['late_mean'], dayshift_weight=None)
            out = signals.predict(models['ordinary'], models['classifier'], V, raw, 'clock', **args)
            reloaded = signals.predict(reloads['ordinary'], reloads['classifier'], V, raw, 'clock', **args)
            for key in ['ordinary', 'p', 'blended', 'final']:
                np.testing.assert_array_equal(out[key], reloaded[key])
            values = {30: out['final']}
            values[33] = plan_window.apply(values[30], raw)
            values[34] = record_fixes.noid_cap(values[33], raw)
            values[37] = dayshift_hedge.hedge(values[34], raw, identity['dayshift_history'])
            for version, pred in values.items():
                frames[version].append(pd.DataFrame(dict(mvt_id=raw.mvt_id.to_numpy(), month=month,
                    ordinary=out['ordinary'], p=out['p'], blended=out['blended'], pred=pred,
                    rule_triggered=out['rule_triggered'], late_rows=out['late_rows'],
                    submitted=np.rint(np.maximum(pred, 0)))))
        results[view] = {}
        for version, pieces in frames.items():
            frame = pd.concat(pieces, ignore_index=True).set_index('mvt_id').loc[ids].reset_index()
            assert frame.mvt_id.is_unique and np.isfinite(frame[['ordinary', 'p', 'pred', 'submitted']]).all().all()
            path = OUT / f'{view}_v{version}.parquet'; frame.to_parquet(path, index=False)
            report = dict(prediction=art(path), rows=len(frame), months=frame.month.value_counts().sort_index().to_dict(),
                all_model_reload_predictions_exact=True, template_order_exact=True)
            if view == 'old':
                target = pd.read_parquet(ROOT / refs[str(version)]['file']).set_index('MVT_ID_mvt').loc[ids, 'TAXITIME_SEC_mvt'].to_numpy()
                delta = frame.submitted.to_numpy()-target
                report.update(exact_original_integer_replay=bool(np.all(delta == 0)), changed_rows=int(np.count_nonzero(delta)),
                    rms_difference=float(np.sqrt(np.mean(delta**2))), maximum_absolute_difference=float(np.max(abs(delta))))
            results[view][str(version)] = report
        write_json(OUT / f'{view}.json', results[view])
        del packet, frames, frame; gc.collect()
        if view == 'old' and not all(x['exact_original_integer_replay'] for x in results[view].values()):
            write_json(OUT / 'result.json', dict(state='closed_approximation_old_replay_failed', identity=art(OUT / 'identity.json'),
                views=results, no_final_inference=True, no_sweep=True, no_upload=True))
            print(json.dumps(dict(stage='closed_approximation', old=results[view])), flush=True)
            return
        print(json.dumps(dict(stage=view+'_complete', versions=list(results[view]))), flush=True)
    write_json(OUT / 'result.json', dict(state='exact_old_replay_final_inference_complete', identity=art(OUT / 'identity.json'),
        views=results, no_upload=True, independent_final_audit_pending=True))


def X_aligned(V, raw):
    return V.index.equals(raw.index) and len(V) == len(raw)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'prepare', 'fit', 'infer'])
    args = parser.parse_args()
    with threadpool_limits(8):
        globals()[args.stage]()
