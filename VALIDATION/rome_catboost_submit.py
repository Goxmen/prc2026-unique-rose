"""Original68-input CatBoost Rome component, prepared separately from callsign15.

Reuse the audited full-year R20 population. Select iterations by December2025
log loss for seed42 (maximum4000, patience150), then fit seeds42/142/242 on all
2025 rows for that same iteration count. Export the three-seed mean probability
for isolated original/final ranking views. No ordinary model, rule, blend weight,
calibration or upload is changed here. Heavy preparation/fits require scheduling.
"""
import argparse
import gc
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]
import catboost_jan_jul as cbj
import final_submit as fs
import rome_catboost_jan_jul as cv
from records import sha256, write_json

OUT = ROOT / 'VALIDATION/runs/v3_20261010_rome_catboost_submit'
VIEWS = ['old', 'final']
SEEDS = [42, 142, 242]
WEATHER_PREPARED = fs.ROME_MATRIX.parent / 'prepared.json'
WEATHER_AUDIT = fs.ROME_MATRIX.parent.parent / 'full_year_audit.json'
MEMBERSHIP = fs.nis.V46 / 'training_membership.parquet'
DESIGN = dict(parent='v58', part='Rome classifier', feature_count=68, categorical_count=11,
    new_features=[], population='unchanged original full-year R20 after_drop rows',
    max_iterations=4000, patience=150, stop_month='2025-12', selection_seed=42,
    full_fit_seeds=SEEDS, target='abs(takeoff-schedule-taxi_sec)<=60 at LIRF',
    criterion='natural unweighted logloss, no calibration',
    export='three-seed mean p; consumer uses fixed half p plus half R20',
    views=VIEWS, upload=False)


def read(path):
    return fs.ts.read(Path(path))


def artifact(path):
    return dict(path=str(Path(path).relative_to(ROOT)), sha256=sha256(path))


def verify(record):
    path = ROOT / record['path']
    if sha256(path) != record['sha256']:
        raise ValueError(f'Changed artifact: {path}')
    return path


def log(value):
    print(time.strftime('[%H:%M:%S] ') + str(value), flush=True)


def parent_spec():
    rows = [read(cv.OUT / 'predictions' / f'{month}_s42.json') for month in ['2025-01', '2025-07']]
    first = rows[0]
    for row in rows:
        if row['columns'] != first['columns'] or row['categoricals'] != first['categoricals'] or row['params'] != first['params']:
            raise ValueError('CV classifier specifications disagree')
        if row['source_sha256'] != sha256(Path(cv.__file__)):
            raise ValueError('Original CatBoost CV source changed')
        for source, digest in row['helper_sha256'].items():
            if sha256(ROOT / source) != digest:
                raise ValueError(f'Original CatBoost helper changed: {source}')
    expected = dict(loss_function='Logloss', eval_metric='Logloss', depth=6,
        learning_rate=.05, l2_leaf_reg=10., random_seed=42, task_type='GPU', devices='0',
        thread_count=2, gpu_ram_part=.65, allow_writing_files=False, verbose=500)
    if first['params'] != expected or len(first['columns']) != 68 or len(first['categoricals']) != 11:
        raise ValueError('Frozen CatBoost recipe changed')
    return first['columns'], first['categoricals'], expected


def identity():
    names, cats, params = parent_spec()
    paths = [cv.OUT / 'predictions' / f'{m}_s42.json' for m in ['2025-01', '2025-07']]
    paths += [ROOT / 'VALIDATION/runs/v3_20261010_rome_catboost_bagging_jan_jul/result.json',
              ROOT / 'VALIDATION/runs/v3_20261010_research_combo_jan_jul/result.json', WEATHER_PREPARED, WEATHER_AUDIT]
    return dict(design=DESIGN, columns=names, categoricals=cats, params=params,
        source_sha256=sha256(Path(__file__)),
        helpers={str(Path(m.__file__).relative_to(ROOT)): sha256(Path(m.__file__))
                 for m in [cbj, cv, fs, fs.signals, fs.rs]},
        evidence=[artifact(p) for p in paths])


def guard():
    current = identity()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'design.json'
    if path.exists():
        if read(path) != current:
            raise ValueError('Production CatBoost design/source/evidence changed')
    else:
        if list(OUT.glob('**/*.parquet')) or list(OUT.glob('**/*.cbm')):
            raise ValueError('Unrecorded production artifacts require inspection')
        write_json(path, current)
    return current


def save_pickle(path, value):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.pkl.tmp')
    pd.to_pickle(value, temp)
    temp.replace(path)


def save_frame(path, frame):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.parquet.tmp')
    frame.to_parquet(temp, index=False)
    temp.replace(path)


def original_training(source, names):
    """Verify source row positions against raw ID/label semantics and membership."""
    X = source['X'].drop(columns=fs.wx.WEATHER)
    y = np.asarray(source['y'], float)
    if list(X) != names or X.shape != (160577, 68) or len(y) != len(X):
        raise ValueError('Original R20 full-year shape/columns changed')
    if not set(np.unique(y)) == {0., 1.} or int(y.sum()) != 33205:
        raise ValueError('Original R20 full-year target population changed')
    with duckdb.connect(config={'threads': 2}) as con:
        con.execute("SET TimeZone='UTC'")
        raw = con.execute('''SELECT MVT_ID_mvt::BIGINT mvt_id,TAXITIME_SEC_mvt taxi_sec,
            ADEP_mvt airport,MVT_TIME_UTC_mvt t,SCHED_TIME_UTC_mvt schedule
            FROM read_parquet(?) WHERE PHASE_mvt='DEP' ''',
            [str(ROOT / 'data/training_2025-*.parquet')]).df()
    selected = raw.loc[X.index]
    if not selected.airport.eq('LIRF').all():
        raise ValueError('Training cache includes non-Rome rows')
    months = selected.t.dt.strftime('%Y-%m').to_numpy()
    if set(months) != {f'2025-{m:02d}' for m in range(1, 13)}:
        raise ValueError('Training does not span exactly2025')
    gap = (selected.t - selected.schedule).dt.total_seconds().to_numpy()
    np.testing.assert_array_equal(gap, X.mvt_minus_sched_sec.to_numpy(float))
    np.testing.assert_array_equal(y, (np.abs(gap-selected.taxi_sec.to_numpy(float)) <= 60).astype(float))
    np.testing.assert_array_equal(X.month.to_numpy(), selected.t.dt.month.to_numpy())
    parent = read(fs.nis.V46 / 'submission_v46.json')
    if sha256(MEMBERSHIP) != parent['training']['artifacts']['training_membership.parquet']:
        raise ValueError('Original R20 membership changed')
    membership = pd.read_parquet(MEMBERSHIP)
    ids = selected.mvt_id.to_numpy(np.int64)
    np.testing.assert_array_equal(ids, membership.loc[membership.used_by_classifier, 'mvt_id'].to_numpy(np.int64))
    return dict(X=X, y=y, ids=ids, months=months, original_index=X.index.to_numpy())


def ranking_packet(data, names, cached_old=None):
    matrices, masks, identifiers = [], [], []
    for month, item in data['months'].items():
        X, raw = item['V'], item['raw']
        use = raw.ADEP_mvt.eq('LIRF').to_numpy() & np.isfinite(X[fs.rs.rome_hybrid.GAP].to_numpy(float))
        matrix = fs.signals.classifier_matrix(X.loc[use], raw.loc[use], fs.rs.KIND, fs.cv.CLASSIFIER)
        if list(matrix) != names:
            raise ValueError('Ranking Rome features differ from original68')
        if cached_old is not None:
            original = cached_old['months'][month]
            np.testing.assert_array_equal(use, original['use'])
            np.testing.assert_array_equal(raw.mvt_id.to_numpy(), original['ids'])
            pd.testing.assert_frame_equal(matrix, original['X'].drop(columns=fs.wx.WEATHER), check_exact=True)
        matrices.append(matrix)
        masks.append(use)
        identifiers.append(raw.mvt_id.to_numpy())
    ids, use = np.concatenate(identifiers), np.concatenate(masks)
    np.testing.assert_array_equal(ids, data['rank_ids'])
    if pd.Index(ids).has_duplicates:
        raise ValueError('Duplicate movement IDs within ranking view')
    return dict(X=pd.concat(matrices, ignore_index=True), ids=ids, use=use)


def prepare():
    recipe = guard()
    if (OUT / 'prepared.json').exists():
        validate_prepared()
        portable_files()
        log('Prepared CatBoost packets verified; skip')
        return
    weather = read(WEATHER_PREPARED)
    if sha256(fs.ROME_MATRIX) != weather['rome_sha256']:
        raise ValueError('Audited full-year Rome matrix changed')
    audit = read(WEATHER_AUDIT)['population']
    if not audit['every_eligible_row_accounted_for'] or audit['rome_rows'] != 160577 or audit['rome_positive_rows'] != 33205:
        raise ValueError('Full-year population audit does not match')
    source = pd.read_pickle(fs.ROME_MATRIX)
    train = original_training(source, recipe['columns'])
    train_path = OUT / 'inputs/train.pkl'
    save_pickle(train_path, train)
    frames = {}
    for view in VIEWS:
        path = fs.WORK / f'inputs_{view}.pkl'
        if sha256(path) != read(fs.OUT / f'inputs_{view}.json')['sha256']:
            raise ValueError('Prepared ranking matrix changed')
        data = pd.read_pickle(path)
        if list(data['months']) != fs.MONTHS[view]:
            raise ValueError('Ranking month scope changed')
        packet = ranking_packet(data, recipe['columns'], source if view == 'old' else None)
        if np.intersect1d(train['ids'], packet['ids']).size:
            raise ValueError('Training/ranking IDs overlap')
        dest = OUT / 'inputs' / f'{view}.pkl'
        save_pickle(dest, packet)
        frames[view] = dict(packet=artifact(dest), original_inputs=artifact(path),
                           rows=len(packet['ids']), eligible_rows=int(packet['use'].sum()))
        del data, packet
        gc.collect()
    write_json(OUT / 'prepared.json', dict(design_sha256=sha256(OUT / 'design.json'),
        train=artifact(train_path), original_matrix=artifact(fs.ROME_MATRIX), membership=artifact(MEMBERSHIP),
        source_movements=[artifact(p) for p in sorted((ROOT / 'data').glob('training_2025-*.parquet'))],
        views=frames, training_rows=len(train['y']), positives=int(train['y'].sum()),
        stop_month=DESIGN['stop_month'], stop_rows=int(np.sum(train['months'] == DESIGN['stop_month'])),
        exact_original_membership_and_targets_verified=True))
    portable_files()
    log('Original68 features,160577 rows and33205 positives verified; two ranking packets ready')


def validate_prepared(portable=False):
    guard()
    prepared = read(OUT / 'prepared.json')
    if prepared['design_sha256'] != sha256(OUT / 'design.json'):
        raise ValueError('Prepared design changed')
    # Always verify every transferred packet and all source/helper/evidence
    # identities in guard(). Only original large source files may be deferred.
    verify(prepared['train'])
    for view in VIEWS:
        verify(prepared['views'][view]['packet'])
    if portable:
        transfer = read(OUT / 'portable_files.json')
        if (transfer['prepared_sha256'] != sha256(OUT / 'prepared.json')
                or transfer['design_sha256'] != sha256(OUT / 'design.json')):
            raise ValueError('Portable transfer manifest identity changed')
        for entry in transfer['files']:
            verify(entry)
    else:
        for entry in [prepared['original_matrix'], prepared['membership'], *prepared['source_movements']]:
            verify(entry)
        for view in VIEWS:
            verify(prepared['views'][view]['original_inputs'])
    return prepared


def portable_files():
    """Compact transfer allowlist; this is produced after full desktop checks."""
    recipe, prepared = read(OUT / 'design.json'), read(OUT / 'prepared.json')
    paths = {OUT / 'design.json', OUT / 'prepared.json', Path(__file__)}
    paths.update(ROOT / item['path'] for item in recipe['evidence'])
    paths.update(ROOT / name for name in recipe['helpers'])
    paths.add(ROOT / prepared['train']['path'])
    paths.update(ROOT / prepared['views'][view]['packet']['path'] for view in VIEWS)
    for month in ['2025-01', '2025-07']:
        record = read(cv.OUT / 'predictions' / f'{month}_s42.json')
        paths.update(ROOT / name for name in record['helper_sha256'])
    report = dict(prepared_sha256=sha256(OUT / 'prepared.json'), design_sha256=sha256(OUT / 'design.json'),
        files=[artifact(path) for path in sorted(paths)],
        source_check_scope='Desktop prepare verifies original matrices, membership and all12 training files. '
            'Portable fit verifies packets, recipe, source/helper files and evidence JSON only. '
            'Desktop export must verify original sources again; portable fit never exports hook records.')
    path = OUT / 'portable_files.json'
    if path.exists() and read(path) != report:
        raise ValueError('Portable transfer allowlist changed')
    write_json(path, report)


def reconstruct_probabilities(ids, use, values):
    ids, use, values = np.asarray(ids), np.asarray(use), np.asarray(values, float)
    if use.dtype != np.dtype(bool) or len(ids) != len(use) or use.sum() != len(values) or pd.Index(ids).has_duplicates:
        raise ValueError('Classifier prediction row/mask mismatch')
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Invalid classifier probability')
    p = np.zeros(len(ids), float)
    p[use] = values
    return pd.DataFrame(dict(mvt_id=ids, p=p))


def completed_seed(seed, rounds):
    path = OUT / 'models' / f's{seed}.json'
    model = OUT / 'models' / f's{seed}.cbm'
    if not path.exists():
        if model.exists() or any((OUT / 'predictions' / f'{view}_s{seed}.parquet').exists() for view in VIEWS):
            raise ValueError('Unrecorded CatBoost fit artifact requires inspection')
        return None
    record = read(path)
    names, cats, parent_params = parent_spec()
    expected_params = dict(parent_params, random_seed=seed)
    if (record['params'] != expected_params or record['rounds'] != rounds
            or record['columns'] != names or record['categoricals'] != cats
            or record['prepared_sha256'] != sha256(OUT / 'prepared.json')
            or record['design_sha256'] != sha256(OUT / 'design.json')):
        raise ValueError('Completed CatBoost fit provenance changed')
    for item in [record['model'], record['selection'], *record['predictions'].values()]:
        verify(item)
    return record


def fit(portable=False):
    from catboost import CatBoostClassifier, Pool, __version__ as catboost_version
    prepared = validate_prepared(portable=portable)
    scope = ('portable: exact packets/code/helpers/evidence; original large sources verified on desktop prepare, '
             'deferred until desktop export' if portable else 'desktop: complete original source and packet verification')
    names, cats, params = parent_spec()
    train = pd.read_pickle(verify(prepared['train']))
    X = cbj.frame(train['X'][names], cats)
    y = train['y']
    hold = train['months'] == DESIGN['stop_month']
    if not hold.any() or not (~hold).any() or any(len(np.unique(y[mask])) != 2 for mask in [hold, ~hold]):
        raise ValueError('December stopping split lacks either class')
    select_path = OUT / 'selection.json'
    inner_path = OUT / 'models/inner_s42.cbm'
    if select_path.exists():
        selected = read(select_path)
        if selected['prepared_sha256'] != sha256(OUT / 'prepared.json') or selected['params'] != params:
            raise ValueError('Recorded December selection changed')
        verify(selected['model'])
        rounds = selected['rounds']
    else:
        if inner_path.exists():
            raise ValueError('Unrecorded stopping fit requires inspection')
        started = time.monotonic()
        inner = CatBoostClassifier(iterations=4000, **params)
        inner.fit(Pool(X.loc[~hold], y[~hold], cat_features=cats),
            eval_set=Pool(X.loc[hold], y[hold], cat_features=cats),
            early_stopping_rounds=150, use_best_model=True)
        rounds = int(inner.get_best_iteration()) + 1
        if not 1 <= rounds <= 4000:
            raise ValueError('Invalid training-selected stopping round')
        inner_path.parent.mkdir(parents=True, exist_ok=True)
        inner.save_model(str(inner_path))
        write_json(select_path, dict(prepared_sha256=sha256(OUT / 'prepared.json'), params=params,
            rounds=rounds, stop_month=DESIGN['stop_month'], inner_rows=int((~hold).sum()), held_rows=int(hold.sum()),
            model=artifact(inner_path), inner_curve=inner.get_evals_result(), source_check_scope=scope,
            minutes=(time.monotonic()-started)/60))
        del inner
        gc.collect()
    for seed in SEEDS:
        if completed_seed(seed, rounds) is not None:
            continue
        started = time.monotonic()
        current_params = dict(params, random_seed=seed)
        model = CatBoostClassifier(iterations=rounds, **current_params)
        model.fit(Pool(X, y, cat_features=cats))
        model_path = OUT / 'models' / f's{seed}.cbm'
        model.save_model(str(model_path))
        predictions = {}
        for view in VIEWS:
            packet = pd.read_pickle(verify(prepared['views'][view]['packet']))
            query = cbj.frame(packet['X'][names], cats)
            probability = model.predict_proba(Pool(query, cat_features=cats))[:, 1]
            frame = reconstruct_probabilities(packet['ids'], packet['use'], probability)
            dest = OUT / 'predictions' / f'{view}_s{seed}.parquet'
            save_frame(dest, frame)
            predictions[view] = artifact(dest)
        write_json(OUT / 'models' / f's{seed}.json', dict(params=current_params, rounds=rounds,
            prepared_sha256=sha256(OUT / 'prepared.json'), design_sha256=sha256(OUT / 'design.json'),
            selection=artifact(select_path), columns=names, categoricals=cats,
            model=artifact(model_path), predictions=predictions, catboost_version=catboost_version,
            source_check_scope=scope, minutes=(time.monotonic()-started)/60))
        del model
        gc.collect()
        log(f'Full-year Rome CatBoost seed{seed}, {rounds} rounds complete')
    if portable:
        log('Portable fits complete. Copy models/selection/predictions back; desktop export must reverify original sources.')
    else:
        export()


def export():
    prepared = validate_prepared()
    rounds = read(OUT / 'selection.json')['rounds']
    records = [completed_seed(seed, rounds) for seed in SEEDS]
    if any(record is None for record in records):
        raise ValueError('All three classifier seeds must complete')
    for view in VIEWS:
        frames = [pd.read_parquet(verify(record['predictions'][view])) for record in records]
        for frame in frames[1:]:
            np.testing.assert_array_equal(frame.mvt_id, frames[0].mvt_id)
        mean = pd.DataFrame(dict(mvt_id=frames[0].mvt_id, p=np.mean([f.p.to_numpy(float) for f in frames], axis=0)))
        dest = OUT / 'probabilities' / f'{view}.parquet'
        if dest.exists():
            old = read(OUT / f'{view}.json')
            verify(old['prediction'])
            pd.testing.assert_frame_equal(pd.read_parquet(dest), mean, check_exact=True)
        else:
            save_frame(dest, mean)
        write_json(OUT / f'{view}.json', dict(view=view, prediction=artifact(dest), source=artifact(Path(__file__)),
            seeds=SEEDS, probability_column='p', prepared=artifact(OUT / 'prepared.json'),
            selection=artifact(OUT / 'selection.json'),
            seed_records=[artifact(OUT / 'models' / f's{seed}.json') for seed in SEEDS],
            rows=len(mean), positive_probability_rows=int(mean.p.gt(0).sum()),
            consumer='fixed0.5 exported mean +0.5 original R20; no calibration or upload'))
    log('Old/final three-seed mean probability hook records exported')


def check():
    parent_spec()
    frame = reconstruct_probabilities([8, 4, 2, 9], np.array([False, True, False, True]), [.2, .8])
    np.testing.assert_array_equal(frame.mvt_id, [8, 4, 2, 9])
    np.testing.assert_array_equal(frame.p, [0., .2, 0., .8])
    for invalid in [[-.1, .8], [np.nan, .8], [1.1, .8]]:
        try:
            reconstruct_probabilities([8, 4], np.array([True, True]), invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid probabilities were accepted')
    log('Original68/11categoricals/params and exact full-ID mask reconstruction checks pass')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['check', 'prepare', 'fit', 'export'])
    parser.add_argument('--portable', action='store_true',
                        help='Fit only from guarded transferred packets; desktop-only full-source export remains required')
    args = parser.parse_args()
    if args.portable and args.stage != 'fit':
        parser.error('--portable is available only for fit; prepare/export require complete original sources')
    with threadpool_limits(limits=1, user_api='blas'):
        if args.stage == 'fit':
            fit(portable=args.portable)
        else:
            globals()[args.stage]()
