"""Portable callsign15 fitting without changing the frozen production runner.

Desktop prepare verifies the complete production inputs, then exports only the
final ranking view's original156 matrix and IDs. The old ranking matrix already
lives inside the hash-verified full-year training cache. The worker reuses that
full-year training cache, verifies every transferred feature/code/recipe packet,
and follows the original DirectData, preprocessing, seed and training contract.
Original source verification is repeated on desktop before final construction.
No guard monkeypatch, alternative training recipe, upload or submission stage.
"""
import argparse
import gc
import gzip
import os
from pathlib import Path
import pickle
import sys
import time

os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'VALIDATION'))
import callsign_submit as parent

OUT = parent.OUT / 'portable_worker'
MANIFEST = OUT / 'manifest.json'
SCOPE = ('Portable worker: exact cached original training matrix, transferred ranking matrices, '
         'feature tables, training IDs, recipe, source and helper hashes. Original ranking/raw '
         'sources fully checked on desktop prepare; desktop audit/build checks them again.')
DESIGN = dict(parent_runner='VALIDATION/callsign_submit.py', original_columns=156,
    packet_schema=['view', 'Vn', 'rank_ids'], view_order=parent.VIEWS,
    old_view='reuse original Vn/rank_ids in verified cached hs.INPUTS',
    groups=parent.GROUPS, nets=parent.NETS, epochs={str(k): v for k, v in parent.EPOCHS.items()},
    target='unchanged clipped direct mean', training='original shared.train; no inner fit',
    portable_scope=SCOPE, original_runner_unchanged=True)


def verify(entry):
    return parent.features.verify_artifact(entry)


def read(path):
    return parent.read(path)


def packet_path(view):
    if view not in parent.VIEWS:
        raise ValueError('Unknown ranking view')
    return OUT / f'inputs_{view}.pkl.gz'


def load_packet(view, manifest, data=None):
    if view == 'old':
        if data is None:
            data = pd.read_pickle(verify(manifest['cached_training_inputs']))
        packet = dict(view=view, Vn=data['Vn'].loc[:, manifest['columns']],
                      rank_ids=np.asarray(data['rank_ids']).copy())
    else:
        path = verify(manifest['views'][view]['packet'])
        packet = pd.read_pickle(path, compression='gzip')
    if (list(packet) != DESIGN['packet_schema'] or packet['view'] != view
            or list(packet['Vn']) != manifest['columns']
            or len(packet['Vn']) != len(packet['rank_ids'])
            or len(packet['Vn']) != manifest['views'][view]['rows']
            or pd.Index(packet['rank_ids']).has_duplicates):
        raise ValueError('Ranking packet contract changed')
    return packet


def ranking_packet(view, data=None):
    """Shared API for production tree fitting; old data may reuse an open cache."""
    manifest, _ = validate_portable()
    return load_packet(view, manifest, data=data)


def validate_portable():
    manifest = read(MANIFEST)
    if manifest['design'] != DESIGN or manifest['worker_source_sha256'] != parent.sha256(Path(__file__)):
        raise ValueError('Portable worker implementation/design changed')
    for entry in manifest['files']:
        verify(entry)
    verify(manifest['cached_training_inputs'])
    prepared = read(parent.OUT / 'prepared.json')
    identity = read(parent.OUT / 'design.json')
    if (prepared['identity'] != identity or prepared['design_sha256'] != parent.sha256(parent.OUT / 'design.json')
            or identity['source_sha256'] != parent.sha256(Path(parent.__file__))
            or identity['design'] != parent.DESIGN
            or prepared['original_columns'] != manifest['columns']
            or prepared['original_inputs'] != manifest['cached_training_inputs']):
        raise ValueError('Original production source/recipe/input identity changed')
    expected_features = {'train': prepared['train_features']}
    expected_features.update({view: prepared['views'][view]['features'] for view in parent.VIEWS})
    if (manifest['training_ids'] != prepared['training_ids']
            or manifest['feature_records'] != expected_features
            or manifest['full_desktop_source_validation_complete'] is not True
            or any(manifest['views'][view]['original_source'] != prepared['views'][view]['inputs']
                   for view in parent.VIEWS)):
        raise ValueError('Portable feature/row packets differ from the original prepared manifest')
    for name, digest in identity['helpers'].items():
        if parent.sha256(ROOT / name) != digest:
            raise ValueError('Original production helper changed: ' + name)
    if parent.read(parent.PARENT_NN)['config'] != parent.cv.shared.recipe():
        raise ValueError('Original N09 training configuration changed')
    return manifest, prepared


def prepare():
    """Only run on desktop with the complete original input/source cache."""
    prepared = parent.validate_prepared()
    if MANIFEST.exists():
        validate_portable()
        parent.log('Existing portable callsign packets verified; skipped')
        return
    OUT.mkdir(parents=True, exist_ok=True)
    if list(OUT.glob('*.pkl.gz')):
        raise ValueError('Unrecorded portable ranking packet; preserve and inspect it')
    views = {}
    names = prepared['original_columns']
    for view in parent.VIEWS:
        if view == 'old':
            # Frozen parent.prepare checked old Vn156/IDs exactly against hs.INPUTS.
            views[view] = dict(storage='cached_training_inputs', rows=parent.features.EXPECTED_ROWS[view],
                original_source=prepared['views'][view]['inputs'],
                exact_matrix_and_ids_verified_by_parent_prepare=True)
            continue
        ranking = parent.ranking_data(view)
        packet = dict(view=view, Vn=ranking['Vn'].loc[:, names].copy(),
                      rank_ids=np.asarray(ranking['rank_ids']).copy())
        pd.testing.assert_frame_equal(packet['Vn'], ranking['Vn'][names], check_exact=True)
        path = packet_path(view)
        temp = path.with_suffix('.tmp')
        with gzip.open(temp, 'wb', compresslevel=1) as stream:
            pickle.dump(packet, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temp.replace(path)
        # A serialized round trip must preserve categorical order and DataFrame index.
        restored = pd.read_pickle(path, compression='gzip')
        pd.testing.assert_frame_equal(restored['Vn'], packet['Vn'], check_exact=True)
        np.testing.assert_array_equal(restored['rank_ids'], packet['rank_ids'])
        views[view] = dict(storage='portable_pickle', packet=parent.artifact(path), rows=len(packet['rank_ids']),
                           original_source=prepared['views'][view]['inputs'],
                           exact_matrix_and_ids_verified=True)
        parent.log(f'{view}: original156/IDs packet {path.stat().st_size/1e6:.2f} MB verified')
        del ranking, packet, restored
        gc.collect()
    files = {Path(__file__), Path(parent.__file__), parent.OUT / 'prepared.json', parent.OUT / 'design.json',
             ROOT / prepared['training_ids']['path'], ROOT / prepared['feature_design']['path'], parent.PARENT_NN}
    files.update(ROOT / name for name in prepared['identity']['helpers'])
    files.update(ROOT / item['path'] for item in prepared['identity']['evidence'].values())
    feature_records = {'train': prepared['train_features']}
    feature_records.update({view: prepared['views'][view]['features'] for view in parent.VIEWS})
    files.update(ROOT / entry['artifact']['path'] for entry in feature_records.values())
    files.update(ROOT / entry['packet']['path'] for entry in views.values() if 'packet' in entry)
    manifest = dict(design=DESIGN, worker_source_sha256=parent.sha256(Path(__file__)),
        cached_training_inputs=prepared['original_inputs'], columns=names, views=views,
        training_ids=prepared['training_ids'], feature_records=feature_records,
        files=[parent.artifact(path) for path in sorted(files)],
        full_desktop_source_validation_complete=True)
    parent.write_json(MANIFEST, manifest)
    validate_portable()
    parent.log('Portable callsign manifest ready; transfer listed files plus manifest.json, reuse cached hs.INPUTS')


def worker_record():
    return dict(source=parent.artifact(Path(__file__)), manifest=parent.artifact(MANIFEST), source_check_scope=SCOPE)


def completed_member(group, k):
    record = parent.completed_member(group, k)
    if record is not None:
        if record.get('portable_worker') != worker_record() or record['config'] != parent.cv.shared.recipe():
            raise ValueError('Saved member differs from this portable worker/recipe')
    return record


def load_feature_table(view, manifest):
    table = parent.cv.load_features(verify(manifest['feature_records'][view]['artifact']))
    return table


def assemble_inputs(data, ids, names, tables, packets):
    """Same joins, concatenation and widening as the frozen parent fit()."""
    X = parent.cv.add_features(data['Xn'][names], ids, tables['train'])
    pd.testing.assert_frame_equal(X[names], data['Xn'][names], check_exact=True)
    held, view_ids, view_sizes = [], {}, {}
    for view in parent.VIEWS:
        packet = packets[view]
        value = parent.cv.add_features(packet['Vn'][names], packet['rank_ids'], tables[view])
        pd.testing.assert_frame_equal(value[names], packet['Vn'][names], check_exact=True)
        held.append(value)
        view_ids[view] = np.asarray(packet['rank_ids']).copy()
        view_sizes[view] = len(value)
    V = pd.concat(parent.fs.widen(held), ignore_index=True)
    return X, V, view_ids, view_sizes


def fit():
    manifest, prepared = validate_portable()
    if not torch.cuda.is_available():
        raise RuntimeError('Portable callsign fits require the allocated GPU')
    if all(completed_member(g, k) is not None for g in parent.GROUPS for k in parent.NETS):
        parent.log('All15 portable callsign members complete; skipped')
        return
    data = pd.read_pickle(verify(manifest['cached_training_inputs']))
    saved_ids = pd.read_parquet(verify(manifest['training_ids']))
    np.testing.assert_array_equal(saved_ids.original_index, data['Xn'].index)
    ids = saved_ids.mvt_id.to_numpy()
    if len(ids) != prepared['training_rows'] or len(ids) != len(data['yn']):
        raise ValueError('Original full-year training alignment changed')
    tables = {view: load_feature_table(view, manifest) for view in ['train'] + parent.VIEWS}
    packets = {view: load_packet(view, manifest, data=data) for view in parent.VIEWS}
    X, V, view_ids, view_sizes = assemble_inputs(data, ids, manifest['columns'], tables, packets)
    parent.cv.ts.FOLDS.setdefault(parent.hs.ids.FULL, parent.features.TRAIN_MONTHS)
    item = dict(X=X, y=data['yn'], Vn=V, tmonth=data['tmonth'], meta=dict(month=parent.hs.ids.FULL),
                raw=pd.DataFrame({'taxi_sec': np.full(len(V), np.nan)}))
    cfg = parent.cv.shared.recipe()
    # item retains the exact y/tmonth arrays. Release unused original Xn/Vn and
    # raw month frames before encoding to bound peak memory on the16GB laptop.
    del data, tables, packets, saved_ids
    gc.collect()
    d = parent.cv.DirectData(item, 'cuda')
    if np.any(d.offset) or np.any(d.val_offset) or d.info['inner_month'] != '2025-12':
        raise ValueError('Direct-target/full-year preprocessing contract changed')
    pre = parent.OUT / 'preprocessing/full_year.pkl'
    pre.parent.mkdir(parents=True, exist_ok=True)
    state = d.preprocessing()
    state['design'] = parent.DESIGN
    if pre.exists():
        parent.resume.retain_saved_preprocessing(pre, state)
    else:
        parent.cv.shared.atomic_pickle(pre, state)
    record = dict(design=parent.DESIGN, config=cfg, encoding=d.info, columns=list(X),
        preprocessing=parent.artifact(pre), prepared_sha256=parent.sha256(parent.OUT / 'prepared.json'),
        view_rows=view_sizes, view_order=parent.VIEWS, label_monitor='ranking labels absent; no epoch selection')
    pre_meta = pre.with_suffix('.json')
    if pre_meta.exists() and read(pre_meta) != record:
        raise ValueError('Saved preprocessing metadata changed')
    parent.write_json(pre_meta, record)
    del X, V, item
    gc.collect()
    for group in parent.GROUPS:
        for k in parent.NETS:
            if completed_member(group, k) is not None:
                continue
            started = time.monotonic()
            model, pred, curve = parent.cv.shared.train(d, np.arange(len(d.y)), group + 1000*k,
                parent.EPOCHS[group], (d.held, d.yv, d.val_offset))
            if not np.isfinite(pred).all() or (pred < 0).any():
                raise ValueError('Invalid portable ranking predictions')
            model_path, record_path = parent.member_paths(group, k)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            temp = model_path.with_suffix('.pt.tmp')
            torch.save(dict(state_dict=model.state_dict(), config=cfg, design=parent.DESIGN,
                            preprocessing=str(pre.relative_to(ROOT))), temp)
            temp.replace(model_path)
            predictions, offset = {}, 0
            for view in parent.VIEWS:
                n = view_sizes[view]
                dest = parent.OUT / 'networks' / f'{view}_s{group}_k{k}.parquet'
                parent.save_frame(dest, pd.DataFrame(dict(mvt_id=view_ids[view], base=pred[offset:offset+n].astype(np.float32))))
                predictions[view] = parent.artifact(dest)
                offset += n
            if offset != len(pred):
                raise ValueError('Ranking view slices did not cover the inference matrix')
            curve = [{name: value for name, value in row.items() if name != 'rmse'} for row in curve]
            parent.write_json(record_path, dict(design_sha256=parent.sha256(parent.OUT / 'design.json'),
                prepared_sha256=parent.sha256(parent.OUT / 'prepared.json'), seed=group+1000*k,
                epochs=parent.EPOCHS[group], model=parent.artifact(model_path), preprocessing=parent.artifact(pre),
                predictions=predictions, config=cfg, training_curve=curve,
                minutes=(time.monotonic()-started)/60, device=torch.cuda.get_device_name(),
                torch=str(torch.__version__), portable_worker=worker_record()))
            parent.log(f'Portable full-year callsign group{group} k{k}, epoch{parent.EPOCHS[group]} saved')
            del model
            torch.cuda.empty_cache()


def audit():
    parent.validate_prepared()
    validate_portable()
    if not all(completed_member(g, k) is not None for g in parent.GROUPS for k in parent.NETS):
        raise ValueError('All15 portable members must complete before desktop audit')
    parent.log('Portable source/artifacts and original desktop sources verified; original build may proceed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'fit', 'audit'])
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(2):
        {'prepare': prepare, 'fit': fit, 'audit': audit}[args.stage]()


if __name__ == '__main__':
    main()
