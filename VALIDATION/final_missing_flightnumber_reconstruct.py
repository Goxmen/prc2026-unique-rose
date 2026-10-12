"""Recover the absent v56 classifier once using its retained original inputs.

Reuse the first68 unchanged columns of the saved full-year weather Rome packet,
append the original two flight-number features, and retain v54's unseen-number
fallback. No parent models are refitted. Exact old integers gate final inference.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

import callsign_submit as parent
from common import TRAIN_GLOB
import rome_fn_submit as original
from records import runtime_identity, sha256, write_json

OUT = ROOT / 'VALIDATION/runs/v3_20261012_final_missing_flightnumber_reconstruct'
OLD = original.OUT / 'submission_v56'
WEATHER = ROOT / 'VALIDATION/runs/v3_20261008_open_meteo_submit/work'
INPUTS = parent.fs.WORK
DESIGN = dict(version=56, rounds=1700, seed=42, threads=8, training_rows=160577,
    base_features=68, augmented_features=70, fallback_unseen=True,
    missing_fit='Only original R20+flight-number classifier; original script did not save its weights',
    input_reuse='First68 original columns of retained Rome weather packet; same immutable original builder',
    success='All344841 old v56 integer predictions exact before final inference',
    failure='Save model and discrepancies; close without final inference or hyperparameter/thread/library sweep',
    parent_final_provenance='Reuse already-audited October8 base-neural final parts; tiny old float refit discrepancy documented separately',
    no_cv_refit=True, no_parent_refit=True, no_upload=True)


def read(path):
    return json.loads(Path(path).read_text())


def art(path):
    return dict(path=str(Path(path).relative_to(ROOT)), sha256=sha256(path))


def freeze():
    assert not (OUT / 'identity.json').exists()
    prior = read(OLD / 'refit.json'); original_manifest = read(OLD / 'submission_v56.json')
    assert prior['rounds'] == 1700 and prior['fallback_unseen']
    assert prior['runtime']['versions'] == runtime_identity()['versions']
    paths = {Path(__file__), Path(parent.__file__), OLD / 'refit.json', OLD / 'submission_v56.json',
             WEATHER / 'rome.pkl', WEATHER / 'prepared.json'}
    modules = [original, original.rc, original.nis, original.cv, original.signals,
               original.rome_hybrid, original.ts, original.features, original.weather,
               original.cv.recipe, original.nis.v43s, original.nis.sf]
    for module in modules:
        path = Path(module.__file__)
        assert sha256(path) == prior['source']['files'][str(path.relative_to(ROOT))], path
        paths.add(path)
    assert sha256(WEATHER / 'rome.pkl') == read(WEATHER / 'prepared.json')['rome_sha256']
    cache = ROOT / 'data/derived/scored_submission_cache_20261011/manifest.json'
    ref = read(cache)['56']; target = ROOT / ref['file']
    assert sha256(target) == ref['sha256'] == original_manifest['sha256']
    paths.update([cache, target])
    for view in ['old', 'final']:
        paths.update([INPUTS / f'inputs_{view}.pkl', parent.fs.OUT / f'inputs_{view}.json'])
        assert sha256(INPUTS / f'inputs_{view}.pkl') == read(parent.fs.OUT / f'inputs_{view}.json')['sha256']
        if view == 'old':
            paths.update(original.V54 / f'ranking_{m}.parquet' for m in ['2026-01', '2026-07'])
        else:
            paths.update(INPUTS / 'parts' / f'{part}.parquet' for part in ['nn', 'trees', 'rome'])
        paths.add(ROOT / 'data' / ('submitting.parquet' if view == 'old' else 'final_submitting.parquet'))
    import glob
    paths.update(Path(p) for p in glob.glob(TRAIN_GLOB))
    params = dict(original.rome_hybrid.CLASSIFIER_PARAMS, **prior['settings'], seed=42,
                  bagging_seed=43, feature_fraction_seed=44, num_threads=8)
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / 'identity.json', dict(design=DESIGN, sources=[art(p) for p in sorted(paths)],
        runtime=runtime_identity(), params=params, original_source_bytes_exact=True,
        original_classifier_not_retained=True))
    print(json.dumps(dict(stage='frozen', sources=len(paths), fits_started=0)))


def guard():
    identity = read(OUT / 'identity.json')
    assert identity['design'] == DESIGN and identity['runtime']['versions'] == runtime_identity()['versions']
    for item in identity['sources']:
        assert sha256(ROOT / item['path']) == item['sha256'], item['path']
    return identity


def prepare():
    identity = guard(); assert not (OUT / 'preflight.json').exists()
    assert not (OUT / 'training.pkl').exists()
    cached = pd.read_pickle(WEATHER / 'rome.pkl')
    names = original.rome_hybrid.regressor_features('clock', original.cv.CLASSIFIER)
    assert len(names) == 68 and list(cached['X'])[:68] == names
    X = cached['X'][names].copy(); y = np.asarray(cached['y'], float)
    del cached; gc.collect()
    # A window keeps the original complete-DEP positional index before filters.
    with duckdb.connect(config={'threads': 1, 'memory_limit': '512MB'}) as con:
        con.execute("SET TimeZone='UTC'")
        raw = con.execute("""WITH dep AS (SELECT row_number() OVER()-1 original_index,
            MVT_ID_mvt mvt_id,ADEP_mvt,FLIGHT_ID_mvt,FLIGHT_mvt,MVT_TIME_UTC_mvt,SCHED_TIME_UTC_mvt,
            TAXITIME_SEC_mvt taxi_sec FROM read_parquet(?) WHERE PHASE_mvt='DEP')
            SELECT * FROM dep WHERE ADEP_mvt='LIRF' AND taxi_sec BETWEEN 1 AND 200000
            AND NOT (FLIGHT_ID_mvt IS NULL AND epoch(MVT_TIME_UTC_mvt-SCHED_TIME_UTC_mvt)>=14400)
            ORDER BY original_index""", [TRAIN_GLOB]).df().set_index('original_index')
        assert len(raw) == len(X) == 160577 and raw.index.equals(X.index)
        gap = (pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True)-pd.to_datetime(raw.SCHED_TIME_UTC_mvt, utc=True)).dt.total_seconds().to_numpy()
        np.testing.assert_array_equal(gap, X['mvt_minus_sched_sec'])
        expected_y = (abs(gap-raw.taxi_sec.to_numpy(float)) <= 60).astype(float)
        np.testing.assert_array_equal(y, expected_y)
        assert int(y.sum()) == 33205
        fn = raw.FLIGHT_mvt.fillna('').astype(str).to_numpy()
        month = pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True).dt.strftime('%Y-%m').to_numpy()
        rate, count = original.rc.copy_rate(fn, month, y, fn, month)
        history = pd.DataFrame(dict(mvt_id=raw.mvt_id.to_numpy(), fn=fn, month=month, copied=y))
        con.register('history', history)
        check = con.execute("""SELECT mvt_id,
            (sum(copied) OVER(PARTITION BY fn)-sum(copied) OVER(PARTITION BY fn,month)
                +10.0*avg(copied) OVER())/(count(*) OVER(PARTITION BY fn)-count(*) OVER(PARTITION BY fn,month)+10.0) rate,
            count(*) OVER(PARTITION BY fn)-count(*) OVER(PARTITION BY fn,month) n FROM history""").df().set_index('mvt_id').loc[raw.mvt_id]
        np.testing.assert_array_equal(rate, check.rate.to_numpy())
        np.testing.assert_array_equal(count, check.n.to_numpy())
        X['fn_copy_rate'], X['fn_copy_n'] = rate, count
    assert len(X.columns) == 70
    pd.to_pickle(dict(X=X, y=y, history=history), OUT / 'training.pkl')
    history.to_parquet(OUT / 'history.parquet', index=False)
    write_json(OUT / 'preflight.json', dict(identity=art(OUT / 'identity.json'), training=art(OUT / 'training.pkl'),
        history=art(OUT / 'history.parquet'), rows=len(X), columns=list(X), positives=int(y.sum()),
        original_ordered_LIRF_membership_and_raw_gaps_targets_exact=True,
        independent_SQL_leave_month_out_rate_and_counts_exact=True,
        base68_values_reused_without_change=True, source_equivalence_explicit=True, fits_started=0))
    print(json.dumps(dict(stage='preflight_pass', rows=len(X), features=len(X.columns), positives=int(y.sum()))))


def fit():
    identity = guard(); preflight = read(OUT / 'preflight.json')
    assert preflight['identity'] == art(OUT / 'identity.json')
    assert preflight['training'] == art(OUT / 'training.pkl')
    assert not (OUT / 'model.txt').exists() and not (OUT / 'trained.json').exists()
    data = pd.read_pickle(OUT / 'training.pkl'); started = time.monotonic()
    model = original.signals._train(identity['params'], data['X'], data['y'], 1700, 'missing v56')
    model.save_model(str(OUT / 'model.txt'))
    write_json(OUT / 'trained.json', dict(identity=art(OUT / 'identity.json'), model=art(OUT / 'model.txt'),
        preflight=art(OUT / 'preflight.json'), columns=model.feature_name(), trees=model.num_trees(),
        seconds=time.monotonic()-started, saved_before_inference=True, runtime=runtime_identity()))
    print(json.dumps(dict(stage='saved_model', seconds=time.monotonic()-started)))


def infer():
    identity = guard(); assert not (OUT / 'result.json').exists()
    trained = read(OUT / 'trained.json')
    assert trained['identity'] == art(OUT / 'identity.json') and trained['model'] == art(OUT / 'model.txt')
    model = lgb.Booster(model_file=str(OUT / 'model.txt')); reload = lgb.Booster(model_file=str(OUT / 'model.txt'))
    history = pd.read_parquet(OUT / 'history.parquet')
    results = {}; ref = read(ROOT / 'data/derived/scored_submission_cache_20261011/manifest.json')['56']
    for view in ['old', 'final']:
        assert not (OUT / f'{view}.json').exists(), 'Preserve completed inference; inspect before resuming'
        data = parent.ranking_data(view)
        neural, tree, old_p, sources = parent.parent_parts(view, data)
        frames = []; offset = 0; unseen_counts = {}
        for month, item in data['months'].items():
            V, raw = item['V'], item['raw']; stop = offset+len(raw); sl = slice(offset, stop)
            base = .6*neural[sl]+.4*tree[sl]
            use = raw.ADEP_mvt.eq('LIRF').to_numpy() & np.isfinite(V[original.rome_hybrid.GAP].to_numpy(float))
            M = original.signals.classifier_matrix(V.loc[use], raw.loc[use], 'clock', original.cv.CLASSIFIER)
            rate, count = original.rc.copy_rate(history.fn.to_numpy(), history.month.to_numpy(), history.copied.to_numpy(),
                raw.loc[use, 'FLIGHT_mvt'].fillna('').astype(str).to_numpy())
            M = M.assign(fn_copy_rate=rate, fn_copy_n=count)
            q = model.predict(M, num_threads=8); q2 = reload.predict(M, num_threads=8)
            np.testing.assert_array_equal(q, q2)
            p = np.zeros(len(raw)); unseen = np.zeros(len(raw), bool)
            p[use] = q; unseen[use] = count == 0
            p[unseen] = old_p[sl][unseen]
            frame, _ = original.nis.final(base, p, V, raw, data['parameters'], data['fitted'])
            frame['month'], frame['base'], frame['p'], frame['old_p'], frame['unseen'] = month, base, p, old_p[sl], unseen
            frames.append(frame); unseen_counts[month] = int(unseen.sum()); offset = stop
        if view == 'old':
            assert unseen_counts == read(OLD / 'refit.json')['unseen_kept_at_v54']
        template = ROOT / 'data' / ('submitting.parquet' if view == 'old' else 'final_submitting.parquet')
        ids = pq.read_table(template, columns=['MVT_ID_mvt'])['MVT_ID_mvt'].to_numpy()
        frame = pd.concat(frames, ignore_index=True).set_index('mvt_id').loc[ids].reset_index()
        frame['submitted'] = np.rint(np.maximum(frame.pred, 0))
        assert frame.mvt_id.is_unique and np.isfinite(frame[['pred', 'submitted', 'p']]).all().all()
        path = OUT / f'{view}_v56.parquet'; frame.to_parquet(path, index=False)
        result = dict(prediction=art(path), rows=len(frame), months=frame.month.value_counts().sort_index().to_dict(),
            parent_parts=sources, unseen_counts=unseen_counts, loaded_model_prediction_exact=True)
        if view == 'old':
            old = pd.read_parquet(ROOT / ref['file']).set_index('MVT_ID_mvt').loc[ids, 'TAXITIME_SEC_mvt'].to_numpy()
            delta = frame.submitted.to_numpy()-old
            result.update(exact_original_integer_replay=bool(np.all(delta == 0)), changed_rows=int(np.count_nonzero(delta)),
                rms_difference=float(np.sqrt(np.mean(delta**2))), maximum_absolute_difference=float(np.max(abs(delta))))
        results[view] = result; write_json(OUT / f'{view}.json', result)
        del data, neural, tree, frames, frame; gc.collect()
        if view == 'old' and not result['exact_original_integer_replay']:
            write_json(OUT / 'result.json', dict(state='closed_approximation_old_replay_failed', views=results,
                identity=art(OUT / 'identity.json'), no_final_inference=True, no_sweep=True, no_upload=True))
            print(json.dumps(dict(stage='closed_approximation', result=result)), flush=True); return
        print(json.dumps(dict(stage=view+'_complete', rows=result['rows'])), flush=True)
    write_json(OUT / 'result.json', dict(state='exact_old_replay_final_inference_complete', views=results,
        identity=art(OUT / 'identity.json'), no_upload=True, independent_final_audit_pending=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'prepare', 'fit', 'infer'])
    args = parser.parse_args()
    with threadpool_limits(8):
        globals()[args.stage]()
