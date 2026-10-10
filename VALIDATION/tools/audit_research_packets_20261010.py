"""Read-only independent compact-packet and calls15 artifact audit; no fits."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'VALIDATION/runs'
MONTHS = ['2025-01', '2025-07']

def read(path):
    return json.loads(path.read_text())

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()

def own_arrival():
    out = RUNS / 'v3_20261010_rome_own_arrival_jan_jul'
    rotation = ROOT / 'data/derived/aircraft_rotation_20261010'
    result = {}
    for month in MONTHS:
        source = RUNS / 'v3_20261010_rome_adsblol_jan_jul/inputs' / f'{month}.pkl'
        original = pd.read_pickle(source)
        path = out / 'inputs' / f'{month}.pkl.gz'
        meta = read(path.with_suffix('.json'))
        assert sha(path) == meta['packet_sha256']
        assert sha(source) == meta['original_cache_sha256']
        assert sha(ROOT / 'VALIDATION/rome_own_arrival_jan_jul.py') == meta['runner_sha256']
        assert sha(rotation / 'folds.json') == meta['rotation_manifest_sha256']
        with gzip.open(path, 'rb') as stream:
            packet = pickle.load(stream)
        names = meta['exact_original_columns_verified']
        assert len(names) == 68 and len(packet['categoricals']) == 11
        for key in ['target', 'use', 'ids', 'train_ids']:
            np.testing.assert_array_equal(packet[key], original[key])
        assert set(np.unique(packet['target'])) == {0, 1}
        details = {}
        for part, key, ids in [('train', 'A', packet['train_ids']),
                               ('held', 'B', packet['ids'][packet['use']])]:
            pd.testing.assert_frame_equal(packet[key][names], original[key][names], check_exact=True)
            src = meta['rotation_sources'][part]
            assert sha(ROOT / src['path']) == src['sha256']
            table = pd.read_parquet(ROOT / src['path'], columns=[
                'mvt_id', 'ar_prev_arrival_delay_sec', 'ar_prev_inblock_age_sec']).set_index('mvt_id').loc[ids]
            delay = table.ar_prev_arrival_delay_sec.to_numpy(float)
            expected = np.full(len(delay), np.nan, dtype=np.float32)
            finite = np.isfinite(delay)
            expected[finite] = np.abs(delay[finite]) <= 60
            np.testing.assert_array_equal(packet[key].rome_own_arrival_copy.to_numpy(), expected)
            np.testing.assert_array_equal(packet[key].rome_own_arrival_age_sec.to_numpy(),
                                          table.ar_prev_inblock_age_sec.to_numpy(np.float32))
            assert packet[key].rome_own_arrival_copy.dtype == np.dtype('float32')
            assert packet[key].rome_own_arrival_age_sec.dtype == np.dtype('float32')
            details[part] = dict(rows=len(ids), copy_nonmissing=int(finite.sum()),
                                 positives=int(np.sum(expected == 1)), references=src['reference_months'])
        result[month] = dict(packet_sha256=sha(path), targets=int(packet['target'].sum()), parts=details)
    return result

def calls15():
    out = RUNS / 'v3_20261010_nn_callsign_clocks_jan_jul'
    manifest = read(out / 'callsign_pool15_manifest.json')
    assert sha(ROOT / 'VALIDATION/nn_callsign_pool15.py') == manifest['source_sha256']
    for path, digest in manifest['helpers'].items():
        assert sha(ROOT / path) == digest
    for filename, key in [('code_identity.json', 'original_code_identity_sha256'),
                          ('callsign_pool_continuation.json', 'original_pool3_manifest_sha256')]:
        assert sha(out / filename) == manifest[key]
    code = read(out / 'code_identity.json')
    for path, digest in code['helpers'].items():
        assert sha(ROOT / path) == digest
    control = manifest['historical_control']
    assert sha(ROOT / control['path']) == control['sha256']
    for path, digest in control['predictions'].items():
        assert sha(ROOT / path) == digest
    result = {}
    for month in MONTHS:
        inherited = manifest['inherited'][month]
        pre = out / 'preprocessing' / f'{month}.pkl'
        metadata = out / 'preprocessing' / f'{month}.json'
        assert sha(pre) == inherited['preprocessing_sha256']
        assert sha(metadata) == inherited['input_manifest_sha256']
        inputs = read(metadata)
        assert inputs['preprocessing_sha256'] == sha(pre)
        original_ids = None
        members = []
        for group in [42, 142, 242]:
            record = read(out / 'networks' / f'{month}_s{group}.json')
            assert record['config'] == control['config']
            assert record['inputs'] == inputs
            assert record['stop_epoch'] == manifest['design']['epochs'][month][str(group)]
            assert set(record['networks']) == {'0', '1', '2', '3', '4'}
            for k in range(5):
                entry = record['networks'][str(k)]
                assert entry['seed'] == group + 1000 * k
                prediction = out / 'networks' / f'{month}_s{group}_k{k}.parquet'
                model = out / 'models' / f'{month}_s{group}_k{k}.pt'
                assert sha(prediction) == entry['sha256']
                assert sha(model) == entry['model_sha256']
                frame = pd.read_parquet(prediction)
                assert frame.mvt_id.is_unique
                assert np.isfinite(frame.drop(columns='mvt_id').to_numpy()).all()
                if original_ids is None:
                    original_ids = frame.mvt_id.to_numpy()
                else:
                    np.testing.assert_array_equal(original_ids, frame.mvt_id.to_numpy())
                if group == 42 and k < 3:
                    assert entry == inherited['networks'][str(k)]
                else:
                    assert entry['expansion']['sha256'] == sha(out / 'callsign_pool15_manifest.json')
                members.append(dict(group=group, k=k, prediction_sha256=sha(prediction), model_sha256=sha(model)))
        result[month] = dict(rows=len(original_ids), inherited_members_identical=3,
                             all_models_and_predictions_verified=15, members=members)
    return result

if __name__ == '__main__':
    output = ROOT / 'tmp/research_20261010/error_mechanisms/research_packets_audit.json'
    result = dict(own_arrival=own_arrival(), calls15=calls15(),
        limitations='No repeated raw feature reconstruction or large baseline matrix hashing; prior audits and fit guards cover these.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: {m: {kk: vv for kk, vv in d.items() if kk != 'members'}
                                for m, d in value.items()} for k, value in result.items()
                      if isinstance(value, dict)}, indent=2))
