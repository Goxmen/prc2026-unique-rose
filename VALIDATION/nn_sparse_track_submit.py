"""Prepare and fit the frozen three-member sparse fusion on the full year.

Execution requires a separately allocated desktop slot. This runner has no
submission construction or upload stage. It preserves original156 inputs and
the full-year N09 group42 stop of15 epochs, with seeds42/1042/2042. Training,
old-ranking and final-ranking tracks have independently verified namespaces.
"""
import argparse
import gc
import os
from pathlib import Path
import tempfile
import time

os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import nn_sparse_track_jan_jul as cv
import nn_sparse_track_pool3 as pool
import build_sparse_track_submit as ranking_cache
import callsign_submit_portable as packets
import chunked_direct_data as adapter
import fixed_epoch_sparse_train as fast
import sparse_track_combo_score as combination

parent = packets.parent
ROOT = cv.ROOT
OUT = ROOT / 'VALIDATION/runs/v3_20261010_nn_sparse_track_submit'
IDENTITY = OUT / 'identity.json'
VIEWS = ['old', 'final']
SEEDS = [42, 1042, 2042]
EPOCHS = 15
DECISION = OUT / 'chosen_recipe.json'
CHOSEN_RECIPE = 'half_sparse3_combination'
DESIGN = dict(original_columns=156, appended_tabular_columns=[], seeds=SEEDS,
    fixed_epochs=EPOCHS, full_year_stop_source='original N09 group42 full-year stop',
    model=cv.DESIGN['model'], sequence=cv.DESIGN['sequence'],
    target=cv.DESIGN['target'], population='unchanged original ordinary full-year rows',
    sequence_namespaces=['train', 'old', 'final'], execution=fast.SETTINGS,
    encoder=adapter.SETTINGS, output='three standalone ordinary-network predictions and equal mean',
    no_new_stopping_or_search=True, no_submission_build_or_upload=True)


def recipe_evidence():
    score_path = cv.OUT / 'score_pool3.json'
    combined_path = combination.OUT / 'result.json'
    pooled = parent.read(score_path)
    combined = parent.read(combined_path)
    if (pooled['design'] != pool.DESIGN or combined['design'] != combination.DESIGN
            or combined['source_sha256'] != cv.sha256(Path(combination.__file__))):
        raise ValueError('Completed pool3 or combination score design/source changed')
    for name, digest in combined['inputs'].items():
        if cv.sha256(ROOT / name) != digest:
            raise ValueError('Combination score input changed: ' + name)
    return dict(chosen_recipe=CHOSEN_RECIPE, recipe=combination.DESIGN,
        pool3_score=parent.artifact(score_path), combination_score=parent.artifact(combined_path),
        combination_predictions=parent.artifact(combination.OUT / 'scored_errors.parquet'),
        selected_cv=combined['scores'][CHOSEN_RECIPE],
        selection='explicit production recipe choice after reused Jan/Jul development evidence')


def prepare(recipe):
    if recipe != CHOSEN_RECIPE:
        raise ValueError('Explicit predeclared combined recipe is required')
    evidence = recipe_evidence()
    OUT.mkdir(parents=True, exist_ok=True)
    if DECISION.exists() and parent.read(DECISION) != evidence:
        raise ValueError('Chosen production recipe or its evidence changed')
    parent.write_json(DECISION, evidence)
    return guard()


def ranking_identity(view, prepared):
    """Tie bytes and metadata to the explicit view, never just movement IDs."""
    target = ranking_cache.OUT / view
    manifest_path = target / 'full.json'
    manifest = parent.read(manifest_path)
    expected = ranking_cache.view_design(ranking_cache.OUT, view, prepared)
    if (any(manifest.get(k) != value for k, value in expected.items())
            or manifest['settings']['ranking_view'] != view
            or manifest['rows'] != ranking_cache.ROWS[view]
            or manifest['point_dtype'] != '<i4'
            or manifest['point_shape'] != [manifest['stored_points'], 6]
            or set(manifest['outputs']) != {'full_index.parquet', 'full_points.bin'}):
        raise ValueError('Production sequence view/source/schema differs: ' + view)
    if parent.read(target / 'full_design.json') != expected:
        raise ValueError('Production sequence design changed: ' + view)
    for name, digest in manifest['outputs'].items():
        if cv.sha256(target / name) != digest:
            raise ValueError('Production sequence bytes changed: ' + view + '/' + name)
    return dict(ranking_view=view, manifest=parent.artifact(manifest_path),
        design=parent.artifact(target / 'full_design.json'), outputs=manifest['outputs'],
        rows=manifest['rows'], stored_points=manifest['stored_points'])


def guard():
    """Full source and saved-evidence validation; run only in allocated slot."""
    chosen = recipe_evidence()
    if not DECISION.exists() or parent.read(DECISION) != chosen:
        raise ValueError('Run prepare with an explicit chosen recipe after reviewing pool3 and combination scores')
    manifest, prepared = packets.validate_portable()
    parent.validate_prepared()
    cfg = cv.shared.recipe()
    historical = parent.read(parent.PARENT_NN)
    if historical['config'] != cfg or historical['groups']['42']['stop_epoch'] != EPOCHS:
        raise ValueError('Frozen full-year N09 stop/configuration changed')
    if len(manifest['columns']) != 156 or len(set(manifest['columns'])) != 156:
        raise ValueError('Original156 column contract changed')
    training = cv.cache_identity()
    pool.guard(training)
    evidence = []
    for month in cv.STOP:
        for k in range(3):
            if pool.validate_member(month, k, training) is None:
                raise ValueError('Complete reviewed three-member CV evidence is required')
            record_path = cv.OUT / 'networks' / f'{month}_s42.json' if k == 0 else pool.paths(month, k)[2]
            evidence.append(parent.artifact(record_path))
    for name in ['score_pool3.json', 'scored_pool3.parquet', 'sparse_pool3_manifest.json']:
        evidence.append(parent.artifact(cv.OUT / name))
    ranking_prepared = ranking_cache.validate(portable=False)
    ranks = {view: ranking_identity(view, ranking_prepared) for view in VIEWS}
    modules = [cv, pool, cv.prototype, cv.seq, cv.tensor_seq, ranking_cache, packets,
        adapter, fast, combination, cv.shared, cv.sensor, cv.shared.nni, cv.shared.nnj,
        cv.preprocess_guard, parent]
    identity = dict(design=DESIGN, chosen_recipe=parent.artifact(DECISION), source=parent.artifact(Path(__file__)),
        helpers={str(Path(m.__file__).relative_to(ROOT)): cv.sha256(Path(m.__file__)) for m in modules},
        config=cfg, original_input_packet=parent.artifact(packets.MANIFEST),
        prepared_original_inputs=parent.artifact(parent.OUT / 'prepared.json'),
        training_cache=training, ranking_caches=ranks, cv_evidence=evidence,
        historical_full_year=parent.artifact(parent.PARENT_NN),
        parity_test=parent.artifact(ROOT / 'VALIDATION/tests/test_fixed_epoch_sparse_train.py'),
        namespace_test=parent.artifact(ROOT / 'VALIDATION/tests/test_nn_sparse_track_submit.py'))
    OUT.mkdir(parents=True, exist_ok=True)
    if IDENTITY.exists():
        if parent.read(IDENTITY) != identity:
            raise ValueError('Production fusion implementation/source/evidence changed')
    else:
        if list(OUT.glob('**/*.pt')) or list(OUT.glob('**/*.parquet')):
            raise ValueError('Unrecorded production artifacts require inspection')
        parent.write_json(IDENTITY, identity)
    return manifest, prepared, identity


def load_tracks(directory, identity, device, view=None):
    directory = Path(directory)
    if view is not None and identity.get('ranking_view') != view:
        raise ValueError('Wrong explicit ranking_view for sequence store')
    index = pd.read_parquet(directory / 'full_index.parquet')
    points = (np.memmap(directory / 'full_points.bin', mode='r', dtype='<i4').reshape(-1, 6)
              if identity['stored_points'] else np.empty((0, 6), dtype=np.int32))
    if len(index) != identity['rows'] or len(points) != identity['stored_points']:
        raise ValueError('Sequence shape differs from the verified manifest')
    store = cv.tensor_seq.RaggedTracks(index, points, device)
    # identity was verified against view_design and the manifest before loading.
    store.ranking_view = view
    store.manifest_identity = identity
    return store


def member_paths(k):
    return OUT / 'models' / f's42_k{k}.pt', OUT / 'networks' / f's42_k{k}.json'


def training_path(k):
    return OUT / 'trained' / f's42_k{k}.json'


def trained_member(k, identity):
    model, _ = member_paths(k)
    path = training_path(k)
    if not path.exists():
        if model.exists():
            raise ValueError('Unrecorded trained weights require inspection; do not refit')
        return None
    result = parent.read(path)
    if (result['identity'] != parent.artifact(IDENTITY) or result['seed'] != SEEDS[k]
            or result['epochs'] != EPOCHS or result['config'] != identity['config']
            or result['model'] != parent.artifact(model)
            or result['preprocessing'] != parent.artifact(OUT / 'preprocessing/full_year.pkl')
            or len(result['training_curve']) != EPOCHS
            or result['training_curve'][-1]['epoch'] != EPOCHS):
        raise ValueError('Saved completed training checkpoint changed')
    return result


def completed_member(k, identity):
    model, metadata = member_paths(k)
    predictions = {view: OUT / 'networks' / f'{view}_s42_k{k}.parquet' for view in VIEWS}
    trained = trained_member(k, identity)
    if not metadata.exists():
        if trained is None and any(path.exists() for path in predictions.values()):
            raise ValueError('Partial unrecorded production member requires inspection')
        return None
    result = parent.read(metadata)
    if (result['identity'] != parent.artifact(IDENTITY) or result['seed'] != SEEDS[k]
            or trained is None or result['training_checkpoint'] != parent.artifact(training_path(k))
            or result['epochs'] != EPOCHS or result['config'] != identity['config']
            or result['model'] != parent.artifact(model)
            or result['preprocessing'] != parent.artifact(OUT / 'preprocessing/full_year.pkl')
            or result['predictions'] != {view: parent.artifact(path) for view, path in predictions.items()}):
        raise ValueError('Completed member provenance/artifacts changed')
    return result


def fit():
    manifest, prepared, identity = guard()
    if not torch.cuda.is_available():
        raise RuntimeError('Allocated desktop GPU required')
    if all(completed_member(k, identity) is not None for k in range(3)):
        parent.log('All three production sparse members complete; skipped')
        return
    data = pd.read_pickle(packets.verify(manifest['cached_training_inputs']))
    saved_ids = pd.read_parquet(packets.verify(manifest['training_ids']))
    np.testing.assert_array_equal(saved_ids.original_index, data['Xn'].index)
    train_ids = saved_ids.mvt_id.to_numpy()
    if len(train_ids) != prepared['training_rows'] or len(train_ids) != len(data['yn']):
        raise ValueError('Original ordinary training ID/label population changed')
    names = manifest['columns']
    X = data['Xn'].loc[:, names]
    held, view_ids = [], {}
    for view in VIEWS:
        packet = packets.load_packet(view, manifest, data=data)
        held.append(packet['Vn'])
        view_ids[view] = np.asarray(packet['rank_ids']).copy()
    V = pd.concat(parent.fs.widen(held), ignore_index=True)
    if list(V) != names or list(X) != names:
        raise ValueError('Original156 column order changed')
    parent.cv.ts.FOLDS.setdefault(parent.hs.ids.FULL, parent.features.TRAIN_MONTHS)
    item = dict(X=X, Vn=V, y=data['yn'], tmonth=data['tmonth'],
        meta=dict(month=parent.hs.ids.FULL), raw=pd.DataFrame({'taxi_sec': np.full(len(V), np.nan)}))
    del data, saved_ids, held, packet
    gc.collect()
    with tempfile.TemporaryDirectory(prefix='encoding_', dir=OUT) as scratch:
        d = adapter.ChunkedDirectData(item, 'cuda', scratch)
    if np.any(d.offset) or np.any(d.val_offset) or d.info['inner_month'] != '2025-12':
        raise ValueError('Original direct target/full-year preprocessing changed')
    pre = OUT / 'preprocessing/full_year.pkl'
    state = d.preprocessing()
    state['design'] = DESIGN
    if pre.exists():
        cv.preprocess_guard.retain_saved_preprocessing(pre, state)
    else:
        cv.shared.atomic_pickle(pre, state)
    pre_metadata = dict(identity=parent.artifact(IDENTITY),
        artifact=parent.artifact(pre), encoding=d.info, columns=names,
        view_order=VIEWS, view_rows={view: len(ids) for view, ids in view_ids.items()})
    if pre.with_suffix('.json').exists() and parent.read(pre.with_suffix('.json')) != pre_metadata:
        raise ValueError('Saved preprocessing metadata changed')
    parent.write_json(pre.with_suffix('.json'), pre_metadata)
    del X, V, item, state
    gc.collect()
    train_tracks = load_tracks(cv.seq.OUT, identity['training_cache'], 'cuda')
    stores = {view: load_tracks(ranking_cache.OUT / view, identity['ranking_caches'][view], 'cuda', view)
              for view in VIEWS}
    train_rows = train_tracks.align(train_ids)
    views = fast.ranking_views(d, view_ids, stores)
    for k, seed in enumerate(SEEDS):
        if completed_member(k, identity) is not None:
            continue
        started = time.monotonic()
        model_path, record_path = member_paths(k)
        trained = trained_member(k, identity)
        if trained is None:
            initial = pool.member_initial_check(d, seed)
            model, curve = fast.train_updates(d, train_tracks, train_rows, views, seed, EPOCHS)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = model_path.with_suffix('.pt.tmp')
            torch.save(dict(state_dict=model.state_dict(), config=identity['config'], design=DESIGN,
                            preprocessing=str(pre.relative_to(ROOT))), temporary)
            temporary.replace(model_path)
            parent.write_json(training_path(k), dict(identity=parent.artifact(IDENTITY), seed=seed,
                epochs=EPOCHS, config=identity['config'], model=parent.artifact(model_path),
                preprocessing=parent.artifact(pre), initial_check=initial, training_curve=curve,
                minutes=(time.monotonic()-started)/60, device=torch.cuda.get_device_name(),
                torch=str(torch.__version__), inference_complete=False))
            trained = trained_member(k, identity)
            parent.log(f'Production sparse k{k}: fixed updates saved before full inference')
        else:
            model = cv.prototype.SparseFusion(d.cards, d.bins, d.train[2].shape[1]).to(d.device)
            model.load_state_dict(torch.load(model_path, map_location=d.device, weights_only=False)['state_dict'])
            parent.log(f'Production sparse k{k}: completed weights reused; inference only')
        predictions, inference_seconds = fast.infer_views(model, d, views)
        # Portable state-dict reload plus independently sliced per-view prediction.
        reloaded = cv.prototype.SparseFusion(d.cards, d.bins, d.train[2].shape[1]).to(d.device)
        reloaded.load_state_dict(torch.load(model_path, map_location=d.device, weights_only=False)['state_dict'])
        artifacts = {}
        for view in views:
            prediction = predictions[view.name]
            np.testing.assert_array_equal(fast.predict_view(reloaded, d, view, limit=fast.INFERENCE_CHUNK),
                                          prediction[:fast.INFERENCE_CHUNK])
            path = OUT / 'networks' / f'{view.name}_s42_k{k}.parquet'
            parent.save_frame(path, pd.DataFrame(dict(mvt_id=view_ids[view.name],
                base=prediction.astype(np.float32), sequence_count=view.tracks.counts[view.rows].astype(np.int16))))
            artifacts[view.name] = parent.artifact(path)
        parent.write_json(record_path, dict(identity=parent.artifact(IDENTITY), seed=seed,
            epochs=EPOCHS, config=identity['config'], model=parent.artifact(model_path),
            preprocessing=parent.artifact(pre), predictions=artifacts,
            training_checkpoint=parent.artifact(training_path(k)), inference_seconds=inference_seconds,
            portable_reload_rows_per_view=fast.INFERENCE_CHUNK,
            minutes=(time.monotonic()-started)/60, device=torch.cuda.get_device_name(), torch=str(torch.__version__)))
        del model, reloaded, predictions
        torch.cuda.empty_cache()
        parent.log(f'Production sparse member k{k}, seed{seed}, fixed15 epochs saved')


def export():
    _, _, identity = guard()
    members = [completed_member(k, identity) for k in range(3)]
    if any(member is None for member in members):
        raise ValueError('All three production networks must complete')
    outputs = {}
    for view in VIEWS:
        frames = [pd.read_parquet(ROOT / member['predictions'][view]['path']) for member in members]
        for frame in frames[1:]:
            np.testing.assert_array_equal(frame.mvt_id, frames[0].mvt_id)
            np.testing.assert_array_equal(frame.sequence_count, frames[0].sequence_count)
        mean = np.mean([frame.base.to_numpy(float) for frame in frames], axis=0)
        path = OUT / 'pooled' / f'{view}.parquet'
        parent.save_frame(path, pd.DataFrame(dict(mvt_id=frames[0].mvt_id, base=mean,
                                                sequence_count=frames[0].sequence_count)))
        outputs[view] = parent.artifact(path)
    parent.write_json(OUT / 'pooled.json', dict(identity=parent.artifact(IDENTITY),
        rule='equal mean of three float32 saved member predictions', members=SEEDS,
        models=[parent.artifact(member_paths(k)[1]) for k in range(3)], outputs=outputs,
        no_submission_build_or_upload=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'check', 'fit', 'export'])
    parser.add_argument('--recipe', choices=[CHOSEN_RECIPE])
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(2):
        if args.stage == 'prepare':
            prepare(args.recipe)
        else:
            if args.recipe is not None:
                parser.error('--recipe is accepted only by explicit prepare')
            {'check': guard, 'fit': fit, 'export': export}[args.stage]()
