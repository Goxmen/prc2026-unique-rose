"""One fixed reconstruction per absent v4-v9 model, exact old integers required.

Stages freeze, prepare, fit and infer; the latter three require --version.
No batch loop, automatic restart, CV fit, tuning or upload is implemented.
The v4 model source is exact; v5-v9 retain the documented absent-source caveat.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline')]
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from common import connect, TRAIN_GLOB
from movement_data import load_phase
from records import runtime_identity

OUT = ROOT / 'VALIDATION/runs/v3_20261012_final_missing_early_reconstruct'
SOURCE = ROOT / 'VALIDATION/archive/runs/v3_20260909T161759260709Z/source/VALIDATION/modeling.py'
ARCHIVE = ROOT / 'VALIDATION/archive/runs/v3_20260909T161759260709Z/result.json'
SOURCE_SHA = '5eda81a506acb91e1baba28e369fac379bf7bfbbed31680deaf7474da8ed920d'
ABSENT_SHA = '5233094e85cdabfb067f82471788f82d141e396d98c0a285b2398510b782fb14'
VERSIONS = [4, 8, 5, 7, 6, 9]
MANIFEST = ROOT / 'data/derived/scored_submission_cache_20261011/manifest.json'
MEMBERSHIP = ROOT / 'VALIDATION/runs/overnight_20260917/models/refit_2025__control.train_ids.parquet'
DESIGN = dict(versions=VERSIONS, rounds=800, seed=42, threads=8, freeze='2025-12-31',
    acceptance='All 344841 old integer predictions exactly match the original version before any final inference',
    source='v4 exact archived source; v5-v9 archived original27 implementation with exact recorded config and helper hashes',
    unknowns=['Original v5-v9 modeling.py source is absent.',
              'Original v4-v9 model weights, runtime versions and individual training-order hashes are absent.'],
    failure='Preserve model and mismatch; no seed/thread/library/source sweep or silent substitution',
    no_cv_fit=True, no_retained_model_refit=True, no_upload=True)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            digest.update(block)
    return digest.hexdigest()


def art(path):
    return dict(path=str(Path(path).relative_to(ROOT)), sha256=sha(path))


def save(path, value):
    temp = Path(path).with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def backend():
    assert sha(SOURCE) == SOURCE_SHA
    spec = importlib.util.spec_from_file_location('missing_original_early', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metadata(version):
    return read(ROOT / f'submissions/unique-rose_v{version}.meta.json')


def freeze():
    assert not (OUT / 'identity.json').exists()
    module = backend()
    paths = {Path(__file__), SOURCE, ARCHIVE, MANIFEST, MEMBERSHIP,
             ROOT / 'VALIDATION/movement_data.py', ROOT / 'VALIDATION/records.py',
             ROOT / 'VALIDATION/protocol.py', ROOT / 'data/submitting.parquet',
             ROOT / 'data/final_ranking.parquet', ROOT / 'data/final_submitting.parquet'}
    cache = read(MANIFEST)
    for version in VERSIONS:
        meta = metadata(version)
        assert meta['params'] == module.PARAMS and meta['seed'] == 42
        assert meta['config']['rounds'] == meta['selected_rounds'] == 800
        assert meta['config']['threads'] == 8 and meta['freeze'] == DESIGN['freeze']
        assert meta['config']['lag_train_days'] == meta['config']['early_stopping_rounds'] == 0
        assert meta['inner_selection'] is None
        assert len(meta['features']) == (28 if version == 4 else 27)
        assert meta['config']['fpl_flag'] == (version == 4)
        assert meta['model_source']['VALIDATION/modeling.py'] == (SOURCE_SHA if version == 4 else ABSENT_SHA)
        paths.add(ROOT / f'submissions/unique-rose_v{version}.meta.json')
        for rel, digest in meta['model_source'].items():
            if rel != 'VALIDATION/modeling.py':
                assert sha(ROOT / rel) == digest, rel
                paths.add(ROOT / rel)
        original = ROOT / cache[str(version)]['file']
        assert sha(original) == cache[str(version)]['sha256']
        paths.add(original)
    for item in read(ARCHIVE)['data_identity']:
        path = ROOT / 'data' / item['file']
        assert sha(path) == item['sha256'], path
        paths.add(path)
    OUT.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.suffix == '.py':
            copy = OUT / 'source' / path.relative_to(ROOT)
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, copy)
    save(OUT / 'identity.json', dict(design=DESIGN, sources=[art(p) for p in sorted(paths)],
        runtime=runtime_identity(), original_models_absent=True,
        expected={str(v): dict(config=metadata(v)['config'], rows=metadata(v)['train_rows'],
            features=metadata(v)['features'], historical_source_exact=v == 4) for v in VERSIONS}))
    print(json.dumps(dict(stage='frozen', sources=len(paths), no_preparation_or_fit=True)), flush=True)


def guard():
    identity = read(OUT / 'identity.json')
    assert identity['design'] == DESIGN
    for item in identity['sources']:
        assert sha(ROOT / item['path']) == item['sha256'], item['path']
    return identity


def load_training():
    con = connect()
    con.execute('SET threads=1')
    dep = load_phase(con, TRAIN_GLOB, 'DEP')
    arr = load_phase(con, TRAIN_GLOB, 'ARR')
    con.close()
    return dep, arr


def prepare(version):
    guard()
    directory = OUT / f'v{version}'
    assert not directory.exists(), 'Preserve any interrupted preparation'
    directory.mkdir()
    module, meta = backend(), metadata(version)
    dep, arr = load_training()
    X, y, cats = module.training_matrix(dep, arr, meta['config'])
    keep = dep.taxi_sec.between(1, meta['config']['max_taxi_sec'])
    raw = dep.loc[keep]
    assert list(X) == meta['features'] and len(X) == meta['train_rows']
    assert X.index.equals(raw.index)
    np.testing.assert_array_equal(y, raw.taxi_sec.to_numpy(float))
    cap_control = dep.loc[dep.taxi_sec.between(1, 200000), 'mvt_id']
    np.testing.assert_array_equal(cap_control, pd.read_parquet(MEMBERSHIP).mvt_id)
    row_features = X.iloc[:, :18].copy()
    row_features['mvt_id'], row_features['label'] = raw.mvt_id.to_numpy(), y
    if version == 4:
        row_features['has_flight_plan'] = X.has_flight_plan.to_numpy()
    with duckdb.connect(config={'threads': 1, 'memory_limit': '512MB'}) as con:
        con.execute("SET TimeZone='UTC'")
        con.register('features', row_features)
        con.read_parquet(TRAIN_GLOB).create_view('raw')
        expected = {'airport': 'r.ADEP_mvt', 'dest': 'r.ADES_mvt', 'aircraft_type': 'r.AIRCRAFT_TYPE_mvt',
            'runway': "r.ADEP_mvt||':'||r.RUNWAY_mvt", 'stand': "r.ADEP_mvt||':'||r.STAND_mvt",
            'flight_rule': 'r.FLIGHT_RULE_mvt', 'market_segment': 'r.MARKET_SEGMENT_flt',
            'flight_type': 'r.FLIGHT_TYPE_flt', 'wake_cat': 'r.WK_TBL_CAT_flt', 'operator': 'r.AIRCRAFT_OPERATOR_flt',
            'hour': 'hour(r.MVT_TIME_UTC_mvt)', 'dow': 'isodow(r.MVT_TIME_UTC_mvt)-1', 'month': 'month(r.MVT_TIME_UTC_mvt)',
            'label': 'r.TAXITIME_SEC_mvt'}
        for name, col in [('sched', 'SCHED_TIME_UTC_mvt'), ('lobt', 'LOBT_flt'), ('iobt', 'IOBT_flt'),
                          ('eobt1', 'EOBT_1_flt'), ('aobt3', 'AOBT_3_flt')]:
            expected[f'mvt_minus_{name}_sec'] = f'epoch(r.MVT_TIME_UTC_mvt-r.{col})'
        if version == 4:
            expected['has_flight_plan'] = '(r.AOBT_3_flt IS NOT NULL)::INTEGER'
        mismatch = ' OR '.join(f'f."{key}" IS DISTINCT FROM {expr}' for key, expr in expected.items())
        count, unique, bad = con.execute(f'''WITH eligible AS (SELECT * FROM raw
            WHERE PHASE_mvt='DEP' AND TAXITIME_SEC_mvt BETWEEN 1 AND ?)
            SELECT count(*),count(DISTINCT f.mvt_id),sum(CASE WHEN f.mvt_id IS NULL
            OR r.MVT_ID_mvt IS NULL OR {mismatch} THEN 1 ELSE 0 END)
            FROM features f FULL JOIN eligible r ON f.mvt_id=r.MVT_ID_mvt''',
            [meta['config']['max_taxi_sec']]).fetchone()
        assert count == unique == len(X) and bad == 0, (count, unique, bad)
    pd.to_pickle(dict(X=X, y=y, ids=raw.mvt_id.to_numpy(), cats=cats), directory / 'training.pkl')
    save(directory / 'preflight.json', dict(identity=art(OUT / 'identity.json'),
        training=art(directory / 'training.pkl'), version=version, rows=len(X), features=list(X),
        original_metadata_feature_order_config_and_count_exact=True,
        independent_SQL_raw_membership_labels_basic18_and_fpl_exact=True,
        raw_order_matches_retained_200000_cap_control=True, original_per_version_order_hash_absent=True,
        source_exact=version == 4, history_helper_hash_exact=True,
        temporal_history_columns_independently_rederived=False, model_fit_started=False))
    print(json.dumps(dict(stage='preflight_pass', version=version, rows=len(X), features=len(X.columns))), flush=True)


def fit(version):
    guard()
    directory = OUT / f'v{version}'
    preflight = read(directory / 'preflight.json')
    assert preflight['identity'] == art(OUT / 'identity.json')
    assert preflight['training'] == art(directory / 'training.pkl')
    assert not (directory / 'fit_started.json').exists(), 'No automatic repeated fit'
    assert not (directory / 'model.txt').exists()
    data, meta, module = pd.read_pickle(directory / 'training.pkl'), metadata(version), backend()
    save(directory / 'fit_started.json', dict(identity=art(OUT / 'identity.json'),
        preflight=art(directory / 'preflight.json'), seed=42, rounds=800, threads=8,
        missing_model_reconstruction=True, automatic_restart_forbidden=True))
    started = time.monotonic()
    model = module.fit_booster(data['X'], data['y'], 42, 800, meta['config'], f'missing v{version}')
    model.save_model(str(directory / 'model.txt'))
    save(directory / 'trained.json', dict(identity=art(OUT / 'identity.json'), version=version,
        model=art(directory / 'model.txt'), preflight=art(directory / 'preflight.json'),
        rounds=model.current_iteration(), seconds=time.monotonic()-started,
        saved_before_inference=True, historical_source_exact=version == 4,
        original_model_hash_unavailable=True, runtime=runtime_identity()))
    print(json.dumps(dict(stage='trained', version=version, seconds=time.monotonic()-started)), flush=True)


def infer(version):
    guard()
    directory = OUT / f'v{version}'
    assert not (directory / 'result.json').exists(), 'Preserve completed or failed replay'
    assert not (directory / 'old.parquet').exists(), 'Preserve interrupted inference'
    trained = read(directory / 'trained.json')
    assert trained['identity'] == art(OUT / 'identity.json')
    assert trained['model'] == art(directory / 'model.txt') and trained['rounds'] == 800
    prepared = read(directory / 'preflight.json')
    assert prepared['training'] == art(directory / 'training.pkl')
    data = pd.read_pickle(directory / 'training.pkl')
    cats = data['cats']
    del data
    gc.collect()
    module, meta = backend(), metadata(version)
    model = lgb.Booster(model_file=str(directory / 'model.txt'))
    replay = lgb.Booster(model_file=str(directory / 'model.txt'))
    dep, arr = load_training()
    reports = {}
    for view in ['old', 'final']:
        source = ROOT / 'data' / ('ranking.parquet' if view == 'old' else 'final_ranking.parquet')
        template = ROOT / 'data' / ('submitting.parquet' if view == 'old' else 'final_submitting.parquet')
        ids = pd.read_parquet(template, columns=['MVT_ID_mvt']).MVT_ID_mvt.to_numpy()
        con = connect()
        con.execute('SET threads=1')
        raw, arrivals = load_phase(con, str(source), 'DEP'), load_phase(con, str(source), 'ARR')
        con.close()
        assert raw.mvt_id.is_unique and set(raw.mvt_id) == set(ids)
        frames = []
        for month in sorted(pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True).dt.strftime('%Y-%m').unique()):
            rows = module.month_rows(raw, month)
            V = module.validation_matrix(dep, arr, rows, module.month_rows(arrivals, month),
                                         DESIGN['freeze'], cats, meta['config'])
            assert list(V) == meta['features']
            pred = module.predict(model, V)
            np.testing.assert_array_equal(pred, module.predict(replay, V))
            frames.append(pd.DataFrame(dict(mvt_id=rows.mvt_id.to_numpy(), month=month,
                                           pred=pred, submitted=np.rint(pred))))
        frame = pd.concat(frames, ignore_index=True).set_index('mvt_id').loc[ids].reset_index()
        assert np.isfinite(frame[['pred', 'submitted']]).all().all()
        assert (frame[['pred', 'submitted']] >= 0).all().all()
        frame.to_parquet(directory / f'{view}.parquet', index=False)
        report = dict(prediction=art(directory / f'{view}.parquet'), rows=len(frame),
            months=frame.month.value_counts().sort_index().to_dict(), template_order_exact=True,
            all_model_reload_predictions_exact=True, original_feature_order_exact=True)
        if view == 'old':
            cache = read(MANIFEST)[str(version)]
            original = pd.read_parquet(ROOT / cache['file']).set_index('MVT_ID_mvt').loc[ids, 'TAXITIME_SEC_mvt']
            diff = frame.submitted.to_numpy() - original.to_numpy()
            report.update(exact_original_integer_replay=bool(np.all(diff == 0)),
                changed_rows=int(np.count_nonzero(diff)), rms_difference=float(np.sqrt(np.mean(diff**2))),
                maximum_absolute_difference=float(np.max(abs(diff))))
        reports[view] = report
        save(directory / f'{view}.json', report)
        del raw, arrivals, frames, frame
        gc.collect()
        if view == 'old' and not report['exact_original_integer_replay']:
            save(directory / 'result.json', dict(state='closed_nonexact_reconstruction',
                identity=art(OUT / 'identity.json'), version=version, views=reports,
                no_final_inference=True, no_sweep=True, no_upload=True))
            print(json.dumps(dict(stage='closed_nonexact_reconstruction', version=version, replay=report)), flush=True)
            return
    save(directory / 'result.json', dict(state='exact_old_replay_final_inference_complete',
        identity=art(OUT / 'identity.json'), version=version, views=reports,
        historical_model_source_exact=version == 4, original_model_hash_unavailable=True,
        independent_final_audit_pending=True, no_upload=True))
    print(json.dumps(dict(stage='exact_old_replay_final_inference_complete', version=version)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'prepare', 'fit', 'infer'])
    parser.add_argument('--version', type=int, choices=VERSIONS)
    args = parser.parse_args()
    with threadpool_limits(limits=1, user_api='blas'):
        if args.stage == 'freeze':
            freeze()
        else:
            assert args.version is not None
            globals()[args.stage](args.version)
