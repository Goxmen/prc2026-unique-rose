"""Fixed frozen-track representation for Rome, with same-laptop Cat controls.

Extract only each outer fold's saved track encoder: 64 pooled activations and
nine input-quality values. No encoder updates, inner stopping, threshold search,
production fitting or upload. Original68 classifier rows/labels remain intact.
"""
import argparse
import gc
import os
from pathlib import Path
import socket
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import nn_sparse_track_jan_jul as tracks
import rome_own_arrival_jan_jul as packets
import rome_catboost_jan_jul as parent
import rome_stand_arrival_jan_jul as context

ROOT = packets.ROOT
OUT = ROOT / 'VALIDATION/runs/v3_20261010_rome_frozen_track_jan_jul'
FEATURES = [f'rt_embedding_{i:02d}' for i in range(64)] + [
    f'rt_quality_{i:02d}' for i in range(9)]
ROUNDS = {'2025-01': 1182, '2025-07': 1401}
DESIGN = dict(parent='selected v64 ordinary stack; fixed half CatBoost/R20 Rome',
    original_features=68, added_features=FEATURES, encoder='own outer-fold sparse156 seed42 k0 track.* only',
    encoder_updates=False, missing='64 activations NaN for fewer than two points; quality retained',
    rows='unchanged original R20 classifier rows including copy positives',
    rounds=ROUNDS, seed=42, controls='original68 and candidate141 on same laptop GPU',
    stopping='fixed original Cat42 rounds; no new inner stop after supervised encoder',
    primary='half new Cat42/half R20 versus half matched original68 Cat42/half R20',
    selected_reference='v64 half historical Cat3/half R20; different classifier pool size explicit',
    ordinary='fixed half sparse3/half calls15 plus callsign main tree; unchanged corrections',
    exploratory_reused_development_months=True, no_weight_search=True,
    no_full_year_fit=True, no_upload=True)


def artifact(path):
    path = Path(path)
    return dict(path=str(path.relative_to(ROOT)), sha256=packets.sha(path))


def verify(record):
    path = ROOT / record['path']
    if packets.sha(path) != record['sha256']:
        raise ValueError('Artifact changed: ' + str(path))
    return path


def identity():
    modules = [packets, parent, context, tracks, tracks.prototype, tracks.tensor_seq,
               tracks.seq, parent.cbj]
    return dict(design=DESIGN, source=artifact(Path(__file__)),
                helpers=[artifact(Path(module.__file__)) for module in modules])


def append(X, ids, table):
    if X.shape[1] != 68 or not table.index.is_unique or len(X) != len(ids):
        raise ValueError('Classifier shape/identity differs')
    if not pd.Index(ids).isin(table.index).all() or set(X) & set(FEATURES):
        raise ValueError('Missing identity or feature overlap')
    result = X.copy()
    for name in FEATURES:
        result[name] = table.loc[ids, name].to_numpy(np.float32)
    pd.testing.assert_frame_equal(result[X.columns], X, check_exact=True)
    if list(result) != list(X) + FEATURES:
        raise ValueError('Feature order differs')
    return result


def encode(encoder, bank, rows, chunk=2048):
    values = []
    encoder.eval()
    with torch.no_grad():
        for start in range(0, len(rows), chunk):
            points, mask, quality = bank.batch(rows[start:start + chunk])
            pooled = encoder(points, mask)
            pooled[mask.sum(1) < 2] = torch.nan
            values.append(torch.cat([pooled, quality], 1).cpu().numpy())
    result = np.concatenate(values)
    if result.shape != (len(rows), 73) or np.isinf(result).any():
        raise ValueError('Invalid encoder shape/values')
    np.testing.assert_array_equal(np.isnan(result[:, :64]).all(1), bank.counts[rows] < 2)
    if not np.isfinite(result[:, 64:]).all():
        raise ValueError('Input quality must be finite')
    return result


def prepare():
    if (OUT / 'prepared.json').exists():
        load_manifest()
        packets.log('Existing frozen feature packet reused')
        return
    started = time.monotonic()
    cache = tracks.cache_identity()
    tracks.guard(cache)
    selected = context.load_manifest()['scoring']
    index = pd.read_parquet(tracks.seq.OUT / 'full_index.parquet')
    points = np.memmap(tracks.seq.OUT / 'full_points.bin', mode='r', dtype=np.int32).reshape(-1, 6)
    bank = tracks.tensor_seq.RaggedTracks(index, points, 'cuda')
    record = dict(identity=identity(), cache=cache, scoring=selected, folds={})
    for month in packets.MONTHS:
        encoder_record = tracks.validate_completed(month, cache)
        if encoder_record is None or encoder_record['month'] != month:
            raise ValueError('Matching outer-fold encoder required')
        checkpoint_path = tracks.OUT / 'models' / f'{month}_s42_k0.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state = {name[len('track.'):]: value for name, value in checkpoint['state_dict'].items()
                 if name.startswith('track.')}
        encoder = tracks.prototype.TrackEncoder().cuda().eval()
        encoder.load_state_dict(state, strict=True)
        encoder.requires_grad_(False)
        packet, meta = packets.load_packet(month)
        source_record_path = parent.OUT / 'predictions' / f'{month}_s42.json'
        source_record = packets.read(source_record_path)
        names = meta['exact_original_columns_verified']
        if (names != source_record['columns'] or packet['categoricals'] != source_record['categoricals']
                or len(packet['A']) != source_record['training_rows']
                or int(packet['target'].sum()) != source_record['positives']
                or source_record['rounds'] != ROUNDS[month]):
            raise ValueError('Original Cat classifier contract differs')
        entries = {}
        for part, key, ids in [('train', 'A', packet['train_ids']),
                               ('held', 'B', packet['ids'][packet['use']])]:
            rows = bank.align(ids)
            scope = index.iloc[rows]
            months = pd.to_datetime(scope.to_ms, unit='ms', utc=True).dt.strftime('%Y-%m')
            expected = set(packets.FOLDS[month]) if part == 'train' else {month}
            if set(months) != expected or not scope.airport.eq('LIRF').all():
                raise ValueError('Classifier fold/airport differs')
            value = encode(encoder, bank, rows)
            # Independent scalar raw-to-input path, on 32 fixed keyed rows per part.
            positions = np.linspace(0, len(rows) - 1, min(32, len(rows)), dtype=int)
            raw_points, raw_mask, raw_quality = tracks.seq.encoded_batch(index.iloc[rows[positions]], points)
            with torch.no_grad():
                expected_embedding = encoder(torch.from_numpy(raw_points).cuda(),
                                             torch.from_numpy(raw_mask).cuda()).cpu().numpy()
            expected_embedding[raw_mask.sum(1) < 2] = np.nan
            np.testing.assert_allclose(value[positions, :64], expected_embedding,
                                       rtol=2e-5, atol=2e-5, equal_nan=True)
            np.testing.assert_allclose(value[positions, 64:], raw_quality, rtol=2e-6, atol=2e-7)
            table = pd.DataFrame(value, columns=FEATURES)
            table.insert(0, 'mvt_id', ids)
            append(packet[key][names], ids, table.set_index('mvt_id'))
            path = OUT / 'inputs' / f'{month}_{part}.parquet'
            path.parent.mkdir(parents=True, exist_ok=True)
            table.to_parquet(path, index=False)
            entries[part] = dict(artifact=artifact(path), rows=len(rows),
                                 usable=int((bank.counts[rows] >= 2).sum()), scalar_checks=len(positions),
                                 query_months=sorted(expected))
        record['folds'][month] = dict(features=entries, columns=names, categoricals=packet['categoricals'],
            packet=artifact(packets.OUT / 'inputs' / f'{month}.pkl.gz'),
            packet_meta=artifact(packets.OUT / 'inputs' / f'{month}.pkl.json'),
            encoder=artifact(checkpoint_path), encoder_record=artifact(tracks.OUT / 'networks' / f'{month}_s42.json'),
            parent_cat_record=artifact(source_record_path), params=source_record['params'])
        del encoder, checkpoint, packet, value, table, state
        gc.collect()
        torch.cuda.empty_cache()
    record['seconds'] = time.monotonic() - started
    packets.write(OUT / 'prepared.json', record)
    packets.log(f'Frozen track classifier inputs prepared in {record["seconds"]:.1f}s')


def load_manifest():
    value = packets.read(OUT / 'prepared.json')
    if value['identity'] != identity():
        raise ValueError('Frozen source/helper/design changed')
    for month, fold in value['folds'].items():
        verify(fold['packet']); verify(fold['packet_meta'])
        for part, entry in fold['features'].items():
            verify(entry['artifact'])
            expected = sorted(packets.FOLDS[month]) if part == 'train' else [month]
            if entry['query_months'] != expected:
                raise ValueError('Fold scope changed')
    for entry in value['scoring']['views'].values():
        verify(entry['artifact'])
    return value


def fit():
    from catboost import CatBoostClassifier, Pool
    manifest = load_manifest()
    for path in (OUT / 'predictions').glob('*.json'):
        saved = packets.read(path)
        if saved['host'] != socket.gethostname() or saved['gpu'] != torch.cuda.get_device_name():
            raise ValueError('All matched controls and candidates must stay on one machine/GPU')
    for month in packets.MONTHS:
        packet, _ = packets.load_packet(month)
        fold = manifest['folds'][month]
        frames = {}
        for part, key, ids in [('train', 'A', packet['train_ids']),
                               ('held', 'B', packet['ids'][packet['use']])]:
            table = pd.read_parquet(verify(fold['features'][part]['artifact'])).set_index('mvt_id')
            np.testing.assert_array_equal(table.index, ids)
            frames[key] = append(packet[key][fold['columns']], ids, table)
        for arm in ['matched', 'candidate']:
            destination = OUT / 'predictions' / f'{month}_{arm}.parquet'
            model_path = OUT / 'models' / f'{month}_{arm}.cbm'
            record_path = destination.with_suffix('.json')
            if record_path.exists():
                saved = packets.read(record_path)
                if saved['identity'] != identity() or saved['prepared'] != artifact(OUT / 'prepared.json'):
                    raise ValueError('Completed model contract changed')
                verify(saved['model']); verify(saved['prediction'])
                continue
            if destination.exists() or model_path.exists():
                raise ValueError('Unrecorded model/output; inspect rather than repeat')
            columns = fold['columns'] + (FEATURES if arm == 'candidate' else [])
            X, V = [parent.cbj.frame(frames[key][columns], fold['categoricals']) for key in ['A', 'B']]
            started = time.monotonic()
            model = CatBoostClassifier(iterations=ROUNDS[month], **fold['params'])
            packets.log(f'{month} {arm}: {len(X)} rows/{len(columns)} inputs/{ROUNDS[month]} rounds')
            model.fit(Pool(X, packet['target'], cat_features=fold['categoricals']))
            probability = np.zeros(len(packet['ids']))
            probability[packet['use']] = model.predict_proba(Pool(V, cat_features=fold['categoricals']))[:, 1]
            if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
                raise ValueError('Invalid probabilities')
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save_model(str(model_path))
            destination.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(dict(mvt_id=packet['ids'], p=probability)).to_parquet(destination, index=False)
            packets.write(record_path, dict(identity=identity(), prepared=artifact(OUT / 'prepared.json'),
                month=month, arm=arm, columns=columns, train_rows=len(X), positives=int(packet['target'].sum()),
                params=fold['params'], rounds=ROUNDS[month], model=artifact(model_path),
                prediction=artifact(destination), host=socket.gethostname(), gpu=torch.cuda.get_device_name(),
                seconds=time.monotonic() - started))
            del model, X, V
            gc.collect()
        del packet, frames
        gc.collect()


def score():
    manifest = load_manifest()
    sc, rome, _, trees = parent.shared.baseline.baseline()
    errors = {arm: {} for arm in ['selected_cat3', 'matched_single_cat', 'candidate_single_cat']}
    rows = []
    for month in packets.MONTHS:
        ids = sc.d[month]['ids']
        selected = pd.read_parquet(verify(manifest['scoring']['views'][month]['artifact']))
        np.testing.assert_array_equal(selected.mvt_id, ids)
        probabilities = {'selected_cat3': selected.cat.to_numpy(float)}
        for arm in ['matched', 'candidate']:
            record = packets.read(OUT / 'predictions' / f'{month}_{arm}.json')
            if record['identity'] != identity() or record['prepared'] != artifact(OUT / 'prepared.json'):
                raise ValueError('Completed model contract changed')
            frame = pd.read_parquet(verify(record['prediction']))
            np.testing.assert_array_equal(frame.mvt_id, ids)
            probabilities[arm + '_single_cat'] = frame.p.to_numpy(float)
        members = {parent.shared.hpo.MEMBER_OF[part]: trees[part][42][month] for part in trees}
        members[parent.shared.hpo.MEMBER_OF['lgb']] = selected.main.to_numpy(float)
        for arm, probability in probabilities.items():
            errors[arm][month] = sc.se(month, 42, nn=selected.nn.to_numpy(float),
                                      p=.5 * rome[month] + .5 * probability, **members)
        np.testing.assert_array_equal(errors['selected_cat3'][month], selected.expected_se.to_numpy(float))
        rows.append(pd.DataFrame(dict(mvt_id=ids, month=month, day=sc.d[month]['day'],
                                      **{arm: value[month] for arm, value in errors.items()})))
    frame = pd.concat(rows, ignore_index=True)
    scores = {arm: sc.cv(value) for arm, value in errors.items()}
    if scores['selected_cat3']['cv'] != manifest['scoring']['expected_cv']:
        raise ValueError('Selected recipe does not reproduce')
    comparisons = {name: parent.shared.ts.paired_bootstrap(frame.day, frame.month,
        frame[name], frame.candidate_single_cat) for name in ['matched_single_cat', 'selected_cat3']}
    frame.to_parquet(OUT / 'scored_errors.parquet', index=False)
    packets.write(OUT / 'result.json', dict(identity=identity(), prepared=artifact(OUT / 'prepared.json'),
        scores=scores, comparisons=comparisons,
        caution='Matched single classifier comparison; actual selected reference has three Cat members. Reused development months.'))
    packets.log(str(scores))


def check():
    X = pd.DataFrame({f'x{i}': np.arange(4, dtype=float) for i in range(67)}, index=[3, 1, 4, 2])
    X['category'] = pd.Categorical(['b', None, 'a', 'c'], categories=['c', 'b', 'a', 'unused'])
    table = pd.DataFrame(np.arange(4 * 73, dtype=np.float32).reshape(4, 73), columns=FEATURES,
                         index=[40, 20, 10, 30])
    table.iloc[1, :64] = np.nan
    result = append(X, [10, 20, 30, 40], table)
    pd.testing.assert_frame_equal(result[X.columns], X, check_exact=True)
    np.testing.assert_array_equal(result[FEATURES].to_numpy(), table.loc[[10, 20, 30, 40]].to_numpy())
    if not all(result[name].dtype == np.float32 for name in FEATURES):
        raise ValueError('Feature precision changed')
    for invalid in [table.iloc[:3], pd.concat([table, table.iloc[:1]])]:
        try:
            append(X, [10, 20, 30, 40], invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Missing or duplicate feature identities were accepted')
    packets.log('Original68 categories/index/values and keyed73 feature order/missingness pass')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['check', 'prepare', 'fit', 'score'])
    torch.set_num_threads(2)
    with threadpool_limits(2):
        globals()[parser.parse_args().stage]()
