"""Refit an archived, validated model implementation without changing its code.

Only model predictions are produced. Rule variants and uploads are separate.
The archived submit.resolve_config source guards run inside a verified copy.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import sys
import time

WORKSPACE = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verified_copy(parent_path, destination):
    record = json.loads(parent_path.read_text())
    for name, expected in record['source']['files'].items():
        original = parent_path.parent / 'source' / name
        if digest(original) != expected:
            raise ValueError(f'Archived source changed: {name}')
        copied = destination / name
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, copied)
    data = destination / 'data'
    if not data.exists():
        data.symlink_to(WORKSPACE / 'data', target_is_directory=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    parent_path = WORKSPACE / 'VALIDATION/runs' / args.parent / 'result.json'
    frozen = WORKSPACE / 'tmp/leaderboard-frozen' / args.parent
    record = verified_copy(parent_path, frozen)
    sys.path[:0] = [str(frozen/'VALIDATION'), str(frozen/'pipeline')]

    import numpy as np
    import pandas as pd
    from common import RANKING, connect
    from harness import load
    from movement_data import load_phase
    from modeling import PARAMS, training_matrix, validation_matrix, fit_booster, month_rows, predict
    from records import MODEL_FILES, write_json, runtime_identity
    from submit import resolve_config, FREEZE

    config, seed, linked = resolve_config(argparse.Namespace(run_id=str(parent_path), seed=42))
    if (not linked['full_benchmark'] or config['rounds'] != 800
            or config['max_taxi_sec'] != 200000 or config['early_stopping_rounds'] != 0
            or config['threads'] != 6 or config['fpl_flag'] or config['lag_train_days'] != 0
            or config['recovery'] not in ['off', 'history']):
        raise ValueError('Outside the selected full-year model recipes')
    for item in record['data_identity']:
        if digest(WORKSPACE/'data'/item['file']) != item['sha256']:
            raise ValueError(f"Changed data: {item['file']}")
    actual_versions = runtime_identity()['versions']
    if record['runtime']['versions'] != actual_versions:
        raise ValueError('Library versions differ from validation')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'metadata.json').exists() or (args.output/'model.txt').exists():
        raise ValueError('Existing fit artifacts; do not repeat a completed/partial fit')
    print(f"Verified archived implementation {args.parent}; loading data", flush=True)
    con = connect()
    con.execute('SET threads=2')
    dep, arr = load(con, 1.0)
    rank, rank_arr = load_phase(con, str(RANKING), 'DEP'), load_phase(con, str(RANKING), 'ARR')
    con.close()
    prepared = []
    options = {'prepared_out': prepared} if 'prepared_out' in inspect.signature(training_matrix).parameters else {}
    print(f"Preparing full-year {config['recovery']} features", flush=True)
    X, y, cats = training_matrix(dep, arr, config, **options)
    if len(X.columns) != 27:
        raise ValueError('Expected the original 27 features')
    print(f'Fitting {len(y):,} rows / 27 features / 800 rounds / 6 threads', flush=True)
    history = {}
    booster = fit_booster(X, y, seed, 800, config, 'full-year '+config['recovery'], training_history=history)
    booster.save_model(str(args.output/'model.txt'))
    write_json(args.output/'training_loss.json', history)
    features, train_rows = X.columns.tolist(), len(y)
    del X, y
    output = rank[['mvt_id', 'ADEP_mvt', 'FLIGHT_ID_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt']].copy()
    output['pred'] = np.nan
    months = sorted(pd.to_datetime(rank.MVT_TIME_UTC_mvt).dt.strftime('%Y-%m').unique())
    equivalence = {}
    for month in months:
        print(f'Predicting isolated ranking month {month}', flush=True)
        part = month_rows(rank, month)
        options = {'prepared_training': prepared[0]} if prepared else {}
        V = validation_matrix(dep, arr, part, month_rows(rank_arr, month), FREEZE, cats, config, **options)
        if 'threads' in inspect.signature(predict).parameters:
            values = predict(booster, V, threads=20)
        else:
            # Historical B0's wrapper had no thread argument. Keep its chunking,
            # clipping and predictions, and verify thread equivalence explicitly.
            values = np.clip(np.concatenate([
                booster.predict(V.iloc[i:i+20000], num_threads=20)
                for i in range(0, len(V), 20000)]), 0, None)
            original = predict(booster, V.iloc[:512])
            np.testing.assert_array_equal(values[:512], original)
            equivalence[month] = '512 rows bit-identical to archived predict()'
        output.loc[part.index, 'pred'] = values
    if output.mvt_id.isna().any() or output.mvt_id.duplicated().any() or not np.isfinite(output.pred).all():
        raise ValueError('Incomplete ranking predictions')
    output.to_parquet(args.output/'predictions.parquet', index=False)
    metadata = {
        'parent': args.parent, 'parent_record_sha256': digest(parent_path),
        'config': config, 'seed': seed, 'rounds': 800, 'freeze': FREEZE,
        'train_rows': train_rows, 'features': features, 'params': PARAMS,
        'ranking_rows': len(output), 'ranking_months': months,
        'source_commit': record['source']['commit'],
        'model_source': {name: digest(frozen/name) for name in MODEL_FILES},
        'archived_source_verified': record['source']['files'],
        'runner_sha256': digest(__file__), 'data_identity': record['data_identity'],
        'runtime': runtime_identity(), 'execution_prediction_threads': 20,
        'prediction_thread_equivalence': equivalence,
        'model_sha256': digest(args.output/'model.txt'),
        'predictions_sha256': digest(args.output/'predictions.parquet'),
        'minutes': (time.monotonic()-started)/60,
    }
    write_json(args.output/'metadata.json', metadata)
    print(f"Completed {args.output}: {metadata['minutes']:.2f} minutes", flush=True)


if __name__ == '__main__':
    main()
