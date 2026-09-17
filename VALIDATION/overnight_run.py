"""Resumable, frozen-recipe validation and full-year refits for the CV study.

This runner only writes local artifacts. Uploads are a separate, explicit step.
All fits use training-only labels; every evaluation departure remains scored.
"""

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline')]
from common import CATEGORICAL, RANKING, TRAIN_GLOB, connect
import modeling
from movement_data import load_phase
import overnight_models as models
from protocol import select_folds
from records import MODEL_FILES, clean, data_identity, runtime_identity, sha256, write_json
from registered_rules import apply_rule
import seasonal

PROTOCOL = 'overnight-cv-study-v1'
FREEZE = '2025-12-31'
SEASONAL_CONTROL = ROOT / 'VALIDATION/runs/seasonal_jan_jul'
SOURCES = sorted(set(MODEL_FILES + [
    'VALIDATION/overnight_run.py', 'VALIDATION/overnight_models.py',
    'VALIDATION/protocol.py', 'VALIDATION/seasonal.py', 'VALIDATION/records.py',
    'pipeline/registered_rules.py', 'pipeline/prediction_rules.py',
]))


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def schedule():
    """Six chronological origins plus three retrospective seasonal fits."""
    windows = {}
    folds, skipped = select_folds('main', months=6)
    if skipped:
        raise ValueError('Unexpected skipped original fold')
    for fold, start in folds:
        key = 'v3_' + fold.train_end[:7]
        if key not in windows:
            windows[key] = dict(id=key, scheme='v3',
                                train=pd.period_range(start, fold.train_end, freq='M').astype(str).tolist(),
                                start=str(start.date()), end=fold.train_end,
                                freeze=fold.freeze, evals=[])
        windows[key]['evals'].append(dict(fold=fold.name, month=fold.val,
                                         role=fold.role, gap=fold.gap))
    jobs = list(windows.values())
    for item in seasonal.schedule('both'):
        jobs.append(dict(id=item['id'], scheme=item['strategy'], train=item['train'],
                         start=item['train'][0] + '-01', end=FREEZE, freeze=FREEZE,
                         evals=[dict(fold=item['id'] + '_' + month, month=month,
                                     role='retrospective', gap=None) for month in item['val']]))
    return jobs


def model_key(job, recipe, seed):
    return f"{job['id']}__{recipe}__{seed}"


def prediction_key(job, evaluation, recipe, seed):
    return f"{model_key(job, recipe, seed)}__{evaluation['fold']}"


def expected_keys(jobs, recipes, seeds):
    fitted, predicted = [], []
    for job in jobs:
        for recipe in recipes:
            for seed in seeds:
                fitted.append(model_key(job, recipe, seed))
                predicted.extend(prediction_key(job, ev, recipe, seed) for ev in job['evals'])
    return fitted, predicted


def validation_complete(progress, jobs, recipes, seeds):
    fitted, predicted = expected_keys(jobs, recipes, seeds)
    return all(k in progress['models'] for k in fitted) and all(k in progress['predictions'] for k in predicted)


def split_job(dep, arr, job):
    return seasonal.split_frames(dep, arr, dict(train=job['train'],
                                               val=[ev['month'] for ev in job['evals']]))


def prediction_metadata(raw, matrix, job, evaluation):
    month = evaluation['month']
    if job['scheme'] == 'v3':
        frame = modeling.prediction_frame(raw, matrix, job['start'], job['end'], month)
        frame['raw_flight_id_missing'] = raw.FLIGHT_ID_mvt.isna().to_numpy()
    else:
        frame = seasonal.prediction_frame(raw, matrix, job, month)
    frame['fold'], frame['month'] = evaluation['fold'], month
    return frame


def artifact_hashes(directory, *paths):
    return {str(path.relative_to(directory)): sha256(path) for path in paths}


def check_artifacts(directory, progress):
    """Verify completed work before accepting a resume or creating a refit."""
    checked = set()
    for kind in ('models', 'predictions', 'refit_models', 'refit_predictions', 'refits'):
        for item in progress.get(kind, {}).values():
            if not item.get('artifacts'):
                raise ValueError(f'Missing artifact fingerprint in {kind}')
            for name, digest in item['artifacts'].items():
                path = directory / name
                if path.resolve().is_relative_to(directory.resolve()) is False:
                    raise ValueError('Artifact path escapes the run directory')
                identity = (name, digest)
                if identity not in checked:
                    if not path.is_file() or sha256(path) != digest:
                        raise ValueError(f'Missing/changed completed artifact: {path}')
                    checked.add(identity)


def save_frame(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    frame.to_parquet(temp, index=False)
    temp.replace(path)


def save_progress(directory, progress):
    write_json(directory / 'progress.json', progress)


def category_values(cats):
    return {name: values.tolist() for name, values in cats.items()}


def read_categories(path):
    return {name: pd.Index(values) for name, values in json.loads(path.read_text()).items()}


def prepare_training(td, ta, config):
    X, y, cats = modeling.training_matrix(td, ta, config)
    raw_kept = td.loc[td.taxi_sec.between(1, config['max_taxi_sec'])]
    if len(X) != len(raw_kept) or not X.index.equals(raw_kept.index):
        raise ValueError('Prepared training matrix no longer matches raw movement order')
    if len(X.columns) != 45:
        raise ValueError('Expected the shared 45-feature preparation matrix')
    np.testing.assert_array_equal(y, raw_kept.taxi_sec.to_numpy(float))
    return X, y, cats, raw_kept


def fit_and_save(directory, progress, bucket, key, job, recipe, seed, X, y,
                 raw_kept, cats_path, config):
    started = time.monotonic()
    booster, history = models.fit_model(X, y, raw_kept, recipe, seed,
                                         rounds=config['rounds'], threads=config['threads'])
    features = models.select_features(X, recipe).columns.tolist()
    if booster.feature_name() != features:
        raise ValueError('Fitted model feature names/order do not match its frozen recipe')
    path = directory / 'models' / f'{key}.txt'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.txt.tmp')
    booster.save_model(str(temporary))
    temporary.replace(path)
    loss_path = path.with_suffix('.loss.json')
    write_json(loss_path, history)
    # Keep one explicit raw-ID record per recipe/origin, shared by its seeds.
    keep = models.training_mask(raw_kept, recipe)
    ids = pd.DataFrame({'mvt_id': raw_kept.mvt_id.to_numpy(),
                        'mvt_time': raw_kept.MVT_TIME_UTC_mvt.to_numpy(),
                        'y': y, 'used_for_fit': keep})
    ids_path = directory / 'models' / f"{job['id']}__{recipe}.train_ids.parquet"
    if not ids_path.exists():
        save_frame(ids_path, ids)
    else:
        pd.testing.assert_frame_equal(pd.read_parquet(ids_path), ids)
    entry = dict(path=str(path.relative_to(directory)), recipe=recipe, seed=seed,
                 job=job['id'], rows=int(np.sum(keep)), available_train_rows=len(y),
                 minutes=(time.monotonic() - started) / 60, features=features,
                 params=models.model_params(recipe, seed, config['threads']),
                 categories=str(cats_path.relative_to(directory)),
                 train_ids=str(ids_path.relative_to(directory)), completed_at=utc_now(),
                 artifacts=artifact_hashes(directory, path, loss_path, ids_path, cats_path))
    progress[bucket][key] = entry
    save_progress(directory, progress)
    print(f'Completed fit {key}: {entry["rows"]:,} rows, {entry["minutes"]:.2f} min', flush=True)
    del booster, ids
    gc.collect()


def seasonal_control_reference(config, runtime, data):
    """Validate the already-completed seasonal control; never fit it again."""
    if config['smoke'] or 'control' not in config['recipes']:
        return None
    record = json.loads((SEASONAL_CONTROL / 'result.json').read_text())
    signature = record['signature']
    if (record['status'] != 'complete' or signature['protocol'] != seasonal.PROTOCOL
            or signature['jobs'] != seasonal.schedule('both')
            or signature['freeze'] != FREEZE or signature['data_identity'] != data
            or signature['versions'] != runtime['versions']):
        raise ValueError('Completed seasonal control has incompatible data, protocol or runtime')
    for key in ('rounds', 'sample', 'lag_train_days', 'max_taxi_sec', 'fpl_flag',
                'early_stopping_rounds', 'threads', 'prediction_threads', 'params'):
        if signature['config'][key] != config[key]:
            raise ValueError(f'Seasonal control config differs: {key}')
    if (signature['config']['features'] != models.ORIGINAL_FEATURES
            or not set(config['seeds']).issubset(signature['config']['seeds'])):
        raise ValueError('Seasonal control features/seeds differ')
    # Its historical wrapper intentionally differs from the shared current
    # wrapper. Record that source exactly, and verify every archived file.
    sources = dict(signature['model_source'], **signature['runner_source'])
    for name, digest in sources.items():
        if sha256(SEASONAL_CONTROL / 'source' / name) != digest:
            raise ValueError(f'Changed archived seasonal control source: {name}')
    for name in ('pipeline/common.py', 'VALIDATION/features_time.py',
                 'VALIDATION/movement_data.py', 'VALIDATION/protocol.py',
                 'pipeline/prediction_rules.py', 'pipeline/registered_rules.py'):
        if sha256(ROOT / name) != sources[name]:
            raise ValueError(f'Seasonal control common dependency changed: {name}')
    return dict(directory=str(SEASONAL_CONTROL.relative_to(ROOT)),
                result_sha256=sha256(SEASONAL_CONTROL / 'result.json'),
                progress_sha256=sha256(SEASONAL_CONTROL / 'progress.json'),
                source=sources, config=signature['config'])


def import_seasonal_control(dep, arr, jobs, config, directory, progress, reference):
    """Copy verified predictions/models and retain explicit historical provenance."""
    if reference is None:
        return
    previous = json.loads((SEASONAL_CONTROL / 'progress.json').read_text())
    checked = set()

    def verify(item):
        for name, digest in item['artifacts'].items():
            if name not in checked:
                if sha256(SEASONAL_CONTROL / name) != digest:
                    raise ValueError(f'Changed completed seasonal control artifact: {name}')
                checked.add(name)

    for job in jobs:
        if job['scheme'] == 'v3' or validation_complete(progress, [job], ['control'], config['seeds']):
            continue
        print(f"Importing completed seasonal control: {job['id']}", flush=True)
        td, ta, validation = split_job(dep, arr, job)
        kept = td.loc[td.taxi_sec.between(1, config['max_taxi_sec'])]
        ids = pd.DataFrame(dict(mvt_id=kept.mvt_id.to_numpy(),
                                 mvt_time=kept.MVT_TIME_UTC_mvt.to_numpy(),
                                 y=kept.taxi_sec.to_numpy(float), used_for_fit=True))
        ids_path = directory / 'models' / f"{job['id']}__control.train_ids.parquet"
        if not ids_path.exists():
            save_frame(ids_path, ids)
        else:
            pd.testing.assert_frame_equal(pd.read_parquet(ids_path), ids)
        cats_path = directory / 'models' / f"{job['id']}.categories.json"
        for seed in config['seeds']:
            key = model_key(job, 'control', seed)
            if key not in progress['models']:
                source_key = f"{job['id']}_baseline_{seed}"
                original = previous['models'][source_key]
                verify(original)
                booster = lgb.Booster(model_file=str(SEASONAL_CONTROL / original['path']))
                if (booster.feature_name() != models.ORIGINAL_FEATURES or original['rows'] != len(kept)
                        or booster.current_iteration() != config['rounds']):
                    raise ValueError('Completed seasonal control model identity differs')
                cats = dict(zip(CATEGORICAL, booster.pandas_categorical, strict=True))
                if cats_path.exists() and json.loads(cats_path.read_text()) != cats:
                    raise ValueError('Imported seasonal category vocabulary differs')
                write_json(cats_path, cats)
                path = directory / 'models' / f'{key}.txt'
                loss_path = path.with_suffix('.loss.json')
                shutil.copyfile(SEASONAL_CONTROL / original['path'], path)
                shutil.copyfile((SEASONAL_CONTROL / original['path']).with_suffix('.loss.json'), loss_path)
                provenance = dict(reference, model_key=source_key, artifacts=original['artifacts'])
                progress['models'][key] = dict(
                    path=str(path.relative_to(directory)), recipe='control', seed=seed,
                    job=job['id'], rows=len(kept), available_train_rows=len(kept),
                    minutes=0., original_fit_minutes=original['minutes'], features=models.ORIGINAL_FEATURES,
                    params=models.model_params('control', seed, config['threads']),
                    categories=str(cats_path.relative_to(directory)), train_ids=str(ids_path.relative_to(directory)),
                    imported_from=provenance, completed_at=utc_now(),
                    artifacts=artifact_hashes(directory, path, loss_path, ids_path, cats_path))
                save_progress(directory, progress)
                del booster
            for ev in job['evals']:
                key = prediction_key(job, ev, 'control', seed)
                if key in progress['predictions']:
                    continue
                source_key = f"{job['id']}_baseline_{ev['month']}_{seed}"
                original = previous['predictions'][source_key]
                verify(original)
                frame = pd.read_parquet(SEASONAL_CONTROL / original['path'])
                vd = validation[ev['month']][0]
                if (frame.mvt_id.isna().any() or frame.mvt_id.duplicated().any()
                        or set(frame.mvt_id) != set(vd.mvt_id)):
                    raise ValueError('Imported seasonal predictions have different evaluation IDs')
                frame = frame.set_index('mvt_id').reindex(vd.mvt_id).reset_index()
                np.testing.assert_array_equal(frame.y, vd.taxi_sec.to_numpy(float))
                expected, audit = apply_rule(frame.parent_pred.to_numpy(), vd, 'rome_4h')
                np.testing.assert_array_equal(frame['Q0-4h'], expected)
                np.testing.assert_array_equal(frame['Q0-4h_triggered'], audit.rule_triggered.to_numpy())
                columns = ['mvt_id', 'mvt_time', 'airport', 'y', 'has_flight_plan',
                           'arrival_history_available', 'arrival_history_days_available',
                           'raw_flight_id_missing']
                out = frame[columns].copy()
                out['seed'], out['fold'], out['month'] = seed, ev['fold'], ev['month']
                out['pred_raw'], out['pred'] = frame.parent_pred, expected
                out['rule_triggered'] = audit.rule_triggered.to_numpy()
                path = directory / 'predictions' / f'{key}.parquet'
                save_frame(path, out)
                progress['predictions'][key] = dict(
                    path=str(path.relative_to(directory)), seed=seed, recipe='control',
                    scheme=job['scheme'], fold=ev['fold'], month=ev['month'], job=job['id'],
                    role=ev['role'], gap=ev['gap'], rows=len(out), completed_at=utc_now(),
                    imported_from=dict(directory=reference['directory'], key=source_key,
                                       artifacts=original['artifacts']), artifacts=artifact_hashes(directory, path))
                save_progress(directory, progress)
        del td, ta, validation, kept, ids
        gc.collect()


def execute(dep, arr, jobs, recipes, config, directory, progress):
    for job in jobs:
        if validation_complete(progress, [job], recipes, config['seeds']):
            print(f"Reusing completed {job['id']}", flush=True)
            continue
        td, ta, validation = split_job(dep, arr, job)
        fitted, _ = expected_keys([job], recipes, config['seeds'])
        cats_path = directory / 'models' / f"{job['id']}.categories.json"
        if any(k not in progress['models'] for k in fitted):
            print(f"Preparing {job['id']}: {len(td):,} training departures", flush=True)
            X, y, cats, raw_kept = prepare_training(td, ta, config)
            if cats_path.exists():
                if json.loads(cats_path.read_text()) != clean(category_values(cats)):
                    raise ValueError('Training category vocabulary changed on resume')
            else:
                write_json(cats_path, category_values(cats))
            for recipe in recipes:
                for seed in config['seeds']:
                    key = model_key(job, recipe, seed)
                    if key not in progress['models']:
                        fit_and_save(directory, progress, 'models', key, job, recipe, seed,
                                     X, y, raw_kept, cats_path, config)
            del X, y, raw_kept
            gc.collect()
        else:
            cats = read_categories(cats_path)
        for ev in job['evals']:
            if all(prediction_key(job, ev, recipe, seed) in progress['predictions']
                   for recipe in recipes for seed in config['seeds']):
                continue
            vd, va = validation[ev['month']]
            print(f"Preparing {job['id']} / {ev['fold']}: {len(vd):,} validation departures", flush=True)
            V = modeling.validation_matrix(td, ta, vd, va, job['freeze'], cats, config)
            meta = prediction_metadata(vd, V, job, ev)
            for recipe in recipes:
                for seed in config['seeds']:
                    key = prediction_key(job, ev, recipe, seed)
                    if key in progress['predictions']:
                        continue
                    fitted = progress['models'][model_key(job, recipe, seed)]
                    booster = lgb.Booster(model_file=str(directory / fitted['path']))
                    pred = models.predict_model(booster, V, recipe, threads=config['prediction_threads'])
                    final, audit = apply_rule(pred, vd, 'rome_4h')
                    frame = meta.copy()
                    frame['seed'], frame['pred_raw'], frame['pred'] = seed, pred, final
                    frame['rule_triggered'] = audit.rule_triggered.to_numpy()
                    path = directory / 'predictions' / f'{key}.parquet'
                    save_frame(path, frame)
                    progress['predictions'][key] = dict(
                        path=str(path.relative_to(directory)), seed=seed, recipe=recipe,
                        scheme=job['scheme'], fold=ev['fold'], month=ev['month'], job=job['id'],
                        role=ev['role'], gap=ev['gap'], rows=len(frame), completed_at=utc_now(),
                        artifacts=artifact_hashes(directory, path))
                    save_progress(directory, progress)
                    del booster, frame
            del V, meta
            gc.collect()
        del td, ta, validation, cats
        gc.collect()


def execute_refit(dep, arr, rank, rank_arr, recipes, config, directory, progress, record):
    """Seed-42 full-year fit; ranking months are isolated context islands."""
    if config['smoke']:
        raise ValueError('Smoke validation cannot authorize a full-year refit')
    if not validation_complete(progress, record['jobs'], recipes, config['seeds']):
        raise ValueError('Complete all validation jobs for the requested recipes before refitting')
    if rank.mvt_id.isna().any() or rank.mvt_id.duplicated().any():
        raise ValueError('Ranking IDs must be unique and nonmissing')
    months = sorted(pd.to_datetime(rank.MVT_TIME_UTC_mvt).dt.strftime('%Y-%m').unique())
    if months != ['2026-01', '2026-07']:
        raise ValueError('Expected exactly the released January/July 2026 ranking islands')
    todo = [recipe for recipe in recipes if recipe not in progress['refits']]
    if not todo:
        print('Requested full-year refits are already complete.', flush=True)
        return
    job = dict(id='refit_2025', freeze=FREEZE)
    cats_path = directory / 'models/refit_2025.categories.json'
    if any(model_key(job, recipe, 42) not in progress['refit_models'] for recipe in todo):
        print(f'Preparing full-year training: {len(dep):,} departures', flush=True)
        X, y, cats, raw_kept = prepare_training(dep, arr, config)
        if cats_path.exists() and json.loads(cats_path.read_text()) != clean(category_values(cats)):
            raise ValueError('Full-year category vocabulary changed on resume')
        write_json(cats_path, category_values(cats))
        for recipe in todo:
            key = model_key(job, recipe, 42)
            if key not in progress['refit_models']:
                fit_and_save(directory, progress, 'refit_models', key, job, recipe, 42,
                             X, y, raw_kept, cats_path, config)
        del X, y, raw_kept
        gc.collect()
    else:
        cats = read_categories(cats_path)
    for month in months:
        if all(f'{recipe}__{month}' in progress['refit_predictions'] for recipe in todo):
            continue
        part = modeling.month_rows(rank, month)
        print(f'Preparing isolated ranking island {month}: {len(part):,} departures', flush=True)
        V = modeling.validation_matrix(dep, arr, part, modeling.month_rows(rank_arr, month),
                                       FREEZE, cats, config)
        for recipe in todo:
            key = f'{recipe}__{month}'
            if key in progress['refit_predictions']:
                continue
            fitted = progress['refit_models'][model_key(job, recipe, 42)]
            booster = lgb.Booster(model_file=str(directory / fitted['path']))
            pred = models.predict_model(booster, V, recipe, threads=config['prediction_threads'])
            final, audit = apply_rule(pred, part, 'rome_4h')
            frame = pd.DataFrame(dict(mvt_id=part.mvt_id.to_numpy(),
                                      mvt_time=part.MVT_TIME_UTC_mvt.to_numpy(),
                                      pred_raw=pred, pred=final,
                                      rule_triggered=audit.rule_triggered.to_numpy()))
            path = directory / 'refits' / f'{key}.parquet'
            save_frame(path, frame)
            progress['refit_predictions'][key] = dict(path=str(path.relative_to(directory)),
                recipe=recipe, month=month, rows=len(frame), artifacts=artifact_hashes(directory, path))
            save_progress(directory, progress)
            del booster, frame
        del V, part
        gc.collect()
    for recipe in todo:
        pieces = [pd.read_parquet(directory / progress['refit_predictions'][f'{recipe}__{month}']['path'])
                  for month in months]
        output = pd.concat(pieces, ignore_index=True).set_index('mvt_id').reindex(rank.mvt_id).reset_index()
        if len(output) != len(rank) or not np.isfinite(output[['pred_raw', 'pred']]).all().all():
            raise ValueError('Incomplete ranking predictions')
        path = directory / 'refits' / f'{recipe}.parquet'
        save_frame(path, output)
        fitted = progress['refit_models'][model_key(job, recipe, 42)]
        metadata = dict(recipe=recipe, seed=42, rounds=config['rounds'], freeze=FREEZE,
                        source_validated=True, signature_sha256=signature_digest(record['signature']),
                        validation_jobs=[j['id'] for j in record['jobs']],
                        validation_seeds=config['seeds'], config=config, model=fitted,
                        ranking_rows=len(output), ranking_months=months,
                        prediction_rule='rome_4h', predictions_sha256=sha256(path),
                        model_source=record['signature']['source'],
                        data_identity=record['signature']['data_identity'], completed_at=utc_now())
        meta_path = path.with_suffix('.meta.json')
        write_json(meta_path, metadata)
        progress['refits'][recipe] = dict(path=str(path.relative_to(directory)),
            metadata=str(meta_path.relative_to(directory)), recipe=recipe, rows=len(output),
            artifacts=artifact_hashes(directory, path, meta_path))
        save_progress(directory, progress)
        print(f'Completed full-year ranking predictions: {path}', flush=True)


def signature_digest(signature):
    return hashlib.sha256(json.dumps(clean(signature), sort_keys=True).encode()).hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['train', 'refit'], nargs='?', default='train')
    parser.add_argument('--recipes', help='Comma-separated ordered subset of the frozen recipe roster')
    parser.add_argument('--seeds', type=int, choices=[1, 3], default=3)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--prediction-threads', type=int, default=8)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'VALIDATION/runs/overnight_20260917')
    parser.add_argument('--smoke', action='store_true', help='1%% rows, five rounds, seed 42; no refits')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if min(args.threads, args.prediction_threads) < 1:
        parser.error('Thread counts must be positive')
    names = args.recipes.split(',') if args.recipes else list(models.RECIPES)
    args.recipes = [name.strip() for name in names]
    if not args.recipes or len(args.recipes) != len(set(args.recipes)) or any(r not in models.RECIPES for r in args.recipes):
        parser.error('Recipes must be a unique, nonempty subset of: ' + ','.join(models.RECIPES))
    if args.action == 'refit' and args.smoke:
        parser.error('Smoke validation cannot authorize a full-year refit')
    return args


def source_signature():
    return {name: sha256(ROOT / name) for name in SOURCES}


def snapshot_sources(directory, signature):
    for name, digest in signature.items():
        path = directory / 'source' / name
        if path.exists():
            if sha256(path) != digest:
                raise ValueError(f'Changed frozen source snapshot: {name}')
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT / name).read_bytes())
            if sha256(path) != digest:
                raise ValueError(f'Source changed while being snapshotted: {name}')


def load_inputs(sample, ranking=False):
    con = connect()
    try:
        con.execute('SET threads=2')
        dep = load_phase(con, TRAIN_GLOB, 'DEP', sample)
        arr = load_phase(con, TRAIN_GLOB, 'ARR', sample)
        if ranking:
            rank = load_phase(con, str(RANKING), 'DEP')
            rank_arr = load_phase(con, str(RANKING), 'ARR')
    finally:
        con.close()
    if dep.mvt_id.isna().any() or dep.mvt_id.duplicated().any() or not np.isfinite(dep.taxi_sec).all():
        raise ValueError('Training departure IDs/targets must be unique, present and finite')
    if dep.MVT_TIME_UTC_mvt.isna().any() or arr.MVT_TIME_UTC_mvt.isna().any():
        raise ValueError('Missing training movement timestamps')
    if any('BLOCK_TIME_UTC' in col for col in dep.columns):
        raise ValueError('Hidden block timestamps must not be loaded')
    print(f'Loaded {len(dep):,} departures / {len(arr):,} arrivals.', flush=True)
    return (dep, arr, rank, rank_arr) if ranking else (dep, arr)


def main(argv=None):
    args = parse_args(argv)
    jobs = schedule()
    seeds = [42] if args.smoke or args.seeds == 1 else [42, 142, 242]
    config = dict(models.BASE_CONFIG, recipes=args.recipes, seeds=seeds,
                  rounds=5 if args.smoke else 800, sample=.01 if args.smoke else 1.,
                  smoke=args.smoke, threads=args.threads, prediction_threads=args.prediction_threads)
    directory = args.output_dir.resolve()
    print(f'{args.action}: {len(jobs)} origins, {sum(len(j["evals"]) for j in jobs)} evaluations, '
          f'{len(args.recipes)} recipes, seeds {seeds}', flush=True)
    if args.dry_run:
        for job in jobs:
            print(f"{job['id']}: train {','.join(job['train'])}; "
                  f"evaluate {','.join(ev['fold'] + ':' + ev['month'] for ev in job['evals'])}")
        print('No models fitted and no data loaded.')
        return
    with seasonal.run_lock(directory):
        runtime = runtime_identity()
        paths = sorted((ROOT / 'data').glob('training_*.parquet')) + [RANKING]
        data = data_identity(paths)
        result_path = directory / 'result.json'
        record = json.loads(result_path.read_text()) if result_path.exists() else None
        if record:
            if not args.resume:
                raise ValueError('Existing run: add --resume or choose another --output-dir')
            if any(r not in record['config']['recipes'] for r in args.recipes):
                raise ValueError('Requested recipe was not in the frozen initial roster')
            config['recipes'] = record['config']['recipes']
        elif args.resume or args.action == 'refit':
            raise ValueError('No existing validation run to resume/refit')
        imported_control = seasonal_control_reference(config, runtime, data)
        signature = dict(protocol=PROTOCOL, jobs=jobs, config=config,
                         source=source_signature(), data_identity=data, runtime=runtime,
                         imported_control=imported_control)
        if record:
            if signature != record['signature']:
                raise ValueError('Cannot resume: configuration, source, data or runtime changed')
            progress = json.loads((directory / 'progress.json').read_text())
            check_artifacts(directory, progress)
        else:
            record = dict(protocol=PROTOCOL, signature=signature, config=config, jobs=jobs,
                          status='running', refit_status={}, started_at=utc_now())
            progress = dict(models={}, predictions={}, refit_models={}, refit_predictions={}, refits={})
            write_json(result_path, record)
            save_progress(directory, progress)
        snapshot_sources(directory, signature['source'])
        try:
            if args.action == 'train':
                if not validation_complete(progress, jobs, args.recipes, seeds):
                    dep, arr = load_inputs(config['sample'])
                    import_seasonal_control(dep, arr, jobs, config, directory, progress, imported_control)
                    execute(dep, arr, jobs, args.recipes, config, directory, progress)
                all_complete = validation_complete(progress, jobs, config['recipes'], seeds)
                record['status'] = ('execution_check_only' if args.smoke else 'complete') if all_complete else 'partial'
            else:
                if not validation_complete(progress, jobs, args.recipes, seeds):
                    raise ValueError('Complete all validation jobs for requested recipes before refitting')
                if any(r not in progress['refits'] for r in args.recipes):
                    dep, arr, rank, rank_arr = load_inputs(1., ranking=True)
                    execute_refit(dep, arr, rank, rank_arr, args.recipes, config, directory, progress, record)
                record['refit_status'] = {recipe: 'complete' for recipe in progress['refits']}
            record['updated_at'] = utc_now()
            record.pop('last_error', None)
            write_json(result_path, record)
        except Exception as exc:
            record['last_error'] = dict(type=type(exc).__name__, message=str(exc), at=utc_now())
            write_json(result_path, record)
            raise
    print(f'Saved {result_path}', flush=True)


if __name__ == '__main__':
    main()
