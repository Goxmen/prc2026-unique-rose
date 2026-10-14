"""Saved corrected-source v3 inference for v84; never trains or uploads.

Replay the selected old-input component before inferring all four final months.
The archived b04dd36 feature functions are copied byte for byte alongside this
file. Departure windows use 2025 labels only; arrivals and movements use the
full supplied final context. Training categories and missing values are fixed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import importlib.metadata
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'VALIDATION/runs/v3_20261014_v84_final/v3'
RECOVERED = ROOT / 'VALIDATION/runs/v3_20261014_exact_v80_recovery/tree_audit/v3_original_replay'
ORIGINAL = Path(__file__).resolve().parent / 'original_v3'
EXPECTED_MODEL = 'da634438f7027d06019f5254e34280a0727c251d56ed1dc63ce3fdf6ef6cca80'
SOURCES = {
    'pipeline/common.py':'1e9e1d04502ee75047a703142688c358f525bb900a7537a6d2a62f333ce771c4',
    'VALIDATION/features_time.py':'52d6190ec6b93f2077f2e2f907b65451d980dd5f1e629ab9b13754ce4a2b3f39',
    'VALIDATION/harness.py':'f3ed364a3437902ceb0e7e018520567b155bd132ab3554349f90bafa78bf0b03'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def art(path):
    return dict(path=str(Path(path).resolve().relative_to(ROOT)), sha256=sha(path))


def verify(record):
    p = ROOT / record['path']
    if sha(p) != record['sha256']:
        raise ValueError(f'Artifact changed: {p}')
    return p


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def freeze(threads):
    paths = [Path(__file__), RECOVERED/'model.txt', RECOVERED/'preflight.json',
             RECOVERED/'trained.json', RECOVERED/'identity.json', RECOVERED/'result.json',
             RECOVERED/'original_ranking.pkl', RECOVERED/'submissions/unique-rose_v3.parquet',
             ROOT/'data/final_ranking.parquet', ROOT/'data/final_submitting.parquet',
             ROOT/'data/submitting.parquet', *sorted((ROOT/'data').glob('training_2025-*.parquet'))]
    if len(list((ROOT/'data').glob('training_2025-*.parquet'))) != 12:
        raise ValueError('Expected twelve complete 2025 training files')
    for rel, expected in SOURCES.items():
        p = ORIGINAL/rel
        if sha(p) != expected:
            raise ValueError('Archived original feature source changed')
        paths.append(p)
    if sha(RECOVERED/'model.txt') != EXPECTED_MODEL:
        raise ValueError('Selected corrected-v3 saved model changed')
    for relative, expected in read(RECOVERED/'identity.json')['input_hashes'].items():
        if relative.startswith('data/training_2025-') and sha(ROOT/relative) != expected:
            raise ValueError('Original 2025 aggregate input changed')
    trained, preflight = read(RECOVERED/'trained.json'), read(RECOVERED/'preflight.json')
    if trained['model_sha256'] != EXPECTED_MODEL or sha(RECOVERED/'original_ranking.pkl') != preflight['ranking_sha256']:
        raise ValueError('Saved v3 model/input provenance changed')
    ident = dict(schema='v84_corrected_v3_saved_inference_v1', sources=[art(p) for p in paths],
                 model=art(RECOVERED/'model.txt'), original_source_git='b04dd36',
                 features=trained['features'], categories=preflight['categorical_values'],
                 threads=threads, chunk_rows=20000, freeze='2025-12-31',
                 runtime=dict(python=sys.version, packages={name:importlib.metadata.version(name)
                              for name in ['numpy','pandas','lightgbm','duckdb','pyarrow']}),
                 final_context='2025 plus all four months of final_ranking.parquet',
                 no_old_id_splicing=True, no_fit=True, no_upload=True)
    p = OUT/'identity.json'
    if p.exists():
        if read(p) != ident:
            raise ValueError('Preserve existing v3 inference identity')
    else:
        write(p, ident)
    return ident


def check_features(X, model, ident):
    import pandas as pd
    if list(X) != ident['features'] or list(X) != model.feature_name():
        raise ValueError('Model feature order differs')
    cats = [X[c].cat.categories.tolist() for c in X if isinstance(X[c].dtype, pd.CategoricalDtype)]
    if cats != model.pandas_categorical:
        raise ValueError('LightGBM training categories differ')


def infer(X, model, threads):
    import numpy as np
    return np.clip(np.concatenate([model.predict(X.iloc[i:i+20000], num_threads=threads)
                                  for i in range(0, len(X), 20000)]), 0, None)


def output(view, ids, pred, ident, extra):
    import numpy as np
    import pandas as pd
    template = pd.read_parquet(ROOT/'data'/('submitting.parquet' if view == 'old' else 'final_submitting.parquet'))
    if len(ids) != len(template) or len(set(ids)) != len(ids) or set(ids) != set(template.MVT_ID_mvt):
        raise ValueError('Incomplete or duplicated prediction IDs')
    frame = pd.DataFrame(dict(mvt_id=ids, raw=pred))
    frame = frame.set_index('mvt_id').loc[template.MVT_ID_mvt].reset_index()
    if not np.isfinite(frame.raw).all() or frame.raw.lt(0).any():
        raise ValueError('Invalid v3 prediction')
    if frame.raw.max() > np.iinfo(np.int32).max:
        raise ValueError('v3 prediction exceeds template integer range')
    frame['submitted'] = np.rint(frame.raw).astype(np.int32)
    if view == 'old':
        expected = pd.read_parquet(RECOVERED/'submissions/unique-rose_v3.parquet')
        np.testing.assert_array_equal(expected.MVT_ID_mvt, frame.mvt_id)
        np.testing.assert_array_equal(expected.TAXITIME_SEC_mvt, frame.submitted)
    destination = OUT/'components'/f'{view}.parquet'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError('Preserve existing unreceipted v3 predictions')
    frame.to_parquet(destination, index=False)
    write(OUT/f'{view}.json', dict(status='pass', identity=art(OUT/'identity.json'),
          prediction=art(destination), rows=len(frame), template=art(ROOT/'data'/('submitting.parquet' if view == 'old' else 'final_submitting.parquet')),
          old_selected_replay_exact=view == 'old', no_fit=True, no_upload=True, **extra))


def run(threads):
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    started = time.time()
    ident = freeze(threads)
    for rec in ident['sources']:
        verify(rec)
    model = lgb.Booster(model_file=str(verify(ident['model'])))
    if not (OUT/'old.json').exists():
        packet = pd.read_pickle(RECOVERED/'original_ranking.pkl')
        check_features(packet['X'], model, ident)
        pred = infer(packet['X'], model, threads)
        output('old', packet['ids'], pred, ident, dict(saved_old_inputs=art(RECOVERED/'original_ranking.pkl')))
        del packet, pred
        gc.collect()
    old = read(OUT/'old.json')
    if old['identity'] != art(OUT/'identity.json') or not old['old_selected_replay_exact']:
        raise ValueError('Selected corrected-v3 old replay failed')
    verify(old['prediction'])
    if (OUT/'final.json').exists():
        final = read(OUT/'final.json')
        if final['identity'] != art(OUT/'identity.json'):
            raise ValueError('Final inference identity changed')
        verify(final['prediction'])
        return
    sys.path[:0] = [str(ORIGINAL/'VALIDATION'), str(ORIGINAL/'pipeline')]
    import common
    import harness
    import features_time
    for module, rel in [(common,'pipeline/common.py'), (harness,'VALIDATION/harness.py'), (features_time,'VALIDATION/features_time.py')]:
        if Path(module.__file__).resolve() != (ORIGINAL/rel).resolve():
            raise ValueError('Original module import shadowed')
    con = common.connect()
    train_glob = str(ROOT/'data/training_2025-*.parquet')
    ranking = str(ROOT/'data/final_ranking.parquet')
    dep = con.sql(f"SELECT TAXITIME_SEC_mvt AS taxi_sec, {common.RAW_COLS} FROM read_parquet('{train_glob}') WHERE PHASE_mvt='DEP' AND TAXITIME_SEC_mvt IS NOT NULL").df()
    arr = con.sql(f"SELECT TAXITIME_SEC_mvt AS taxi_sec, ADES_mvt, MVT_TIME_UTC_mvt FROM read_parquet('{train_glob}') WHERE PHASE_mvt='ARR' AND TAXITIME_SEC_mvt IS NOT NULL").df()
    rank = con.sql(f"SELECT MVT_ID_mvt AS mvt_id, {common.RAW_COLS} FROM read_parquet('{ranking}') WHERE PHASE_mvt='DEP'").df()
    rank_arr = con.sql(f"SELECT TAXITIME_SEC_mvt AS taxi_sec, ADES_mvt, MVT_TIME_UTC_mvt FROM read_parquet('{ranking}') WHERE PHASE_mvt='ARR' AND TAXITIME_SEC_mvt IS NOT NULL").df()
    con.close()
    if len(rank) != 670790 or set(rank.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m')) != {'2026-01','2026-02','2026-06','2026-07'}:
        raise ValueError('Unexpected final departure months/population')
    if 'taxi_sec' in rank:
        raise ValueError('Final departure target entered features')
    arr_all = pd.concat([arr, rank_arr], ignore_index=True)
    tf = features_time.TimeFeatures(dep[dep.taxi_sec.between(common.MIN_TAXI_SEC, harness.TAIL_SEC)],
        arr_all[arr_all.taxi_sec.between(common.MIN_TAXI_SEC, harness.TAIL_SEC)],
        dep_events=pd.concat([dep,rank], ignore_index=True), arr_events=arr_all)
    X = harness.matrix(rank, tf, freeze_day=364, fpl_flag=False)
    for col, cats in ident['categories'].items():
        X[col] = pd.Categorical(X[col].where(X[col].isin(cats)), categories=cats)
    check_features(X, model, ident)
    packet_path = OUT/'final_inputs.pkl'
    if packet_path.exists():
        raise ValueError('Preserve unreceipted final v3 input packet')
    pd.to_pickle(dict(X=X, ids=rank.mvt_id.to_numpy()), packet_path)
    rank[['mvt_id','MVT_TIME_UTC_mvt','ADEP_mvt']].to_parquet(OUT/'final_row_metadata.parquet', index=False)
    counts = {str(k):int(v) for k,v in rank.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m').value_counts().items()}
    missing = {str(k):int(v) for k,v in X.isna().sum().items()}
    pred = infer(X, model, threads)
    reloaded = lgb.Booster(model_file=str(verify(ident['model'])))
    np.testing.assert_array_equal(pred, infer(X, reloaded, threads))
    output('final', rank.mvt_id.to_numpy(), pred, ident,
           dict(inputs=art(packet_path), row_metadata=art(OUT/'final_row_metadata.parquet'),
                month_counts=counts, feature_missing_counts=missing,
                independent_model_reload_float_exact=True, all_rows_fresh_inference=True,
                no_old_id_splicing=True, elapsed_seconds=time.time()-started))
    write(OUT/'result.json', dict(status='pass', identity=art(OUT/'identity.json'),
          old=art(OUT/'old.json'), final=art(OUT/'final.json'), model=ident['model'],
          no_fit=True, no_upload=True, pid=os.getpid(), elapsed_seconds=time.time()-started))
    print(json.dumps({'v3_final':'pass','rows':670790,'seconds':time.time()-started}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError('Positive thread count required')
    run(args.threads)
