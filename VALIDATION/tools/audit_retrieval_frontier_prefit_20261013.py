"""Independent retrieval input/firewall and checkpoint audit. No April targets.

`prefit` reconstructs sampling, aliases, features, scaling, permitted targets and
donor counts; `prescore` verifies both persisted arms before held truth is read.
Neither command imports or invokes producer transformations/model inference.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'VALIDATION/runs/v3_20261013_train_label_retrieval'
BANK = ROOT/'data/derived/flight_cohort_context_20261012/train'
TRAIN = [f'2025-{m:02d}' for m in [2, 3, 5, 6, 8, 9, 10, 11, 12]]
ARMS = ['labels_off', 'labels_on']
CAT = ['airport', 'dest', 'aircraft_type', 'runway', 'stand', 'flight_rule',
       'market_segment', 'flight_type', 'wake_cat', 'operator', 'airline']
CLOCKS = dict(sched='SCHED_TIME_UTC_mvt', lobt='LOBT_flt', iobt='IOBT_flt',
              eobt1='EOBT_1_flt', aobt3='AOBT_3_flt')
FEATURES = CAT[:-1]+['hour', 'dow', 'month']+[
    'mvt_minus_'+s+'_sec' for s in CLOCKS]+['physical_'+str(i) for i in range(17)]+[
    'missing_flight_id', 'airline']
NUM = [c for c in FEATURES if c not in CAT]
CHECKED = set()


def read(path): return json.loads(Path(path).read_text())


def art(path):
    path = Path(path).resolve(); h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): h.update(block)
    return dict(path=str(path.relative_to(ROOT)), sha256=h.hexdigest(), bytes=path.stat().st_size)


def verify(record):
    path = (ROOT/record['path']).resolve(); path.relative_to(ROOT)
    key = (str(path), record['sha256'], record['bytes'])
    if key not in CHECKED:
        assert art(path) == record, path
        CHECKED.add(key)
    return path


def save(path, value):
    path = Path(path); assert not path.exists(), path
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); temp.replace(path)


def eq(a, b): np.testing.assert_array_equal(a, b)


def receipt(name):
    path = OUT/name; value = read(path)
    assert value['identity'] == art(OUT/'identity.json'), path
    for key in ['artifacts', 'dependencies']:
        for record in value.get(key, []): verify(record)
    return value


def guard(stage):
    identity = read(OUT/'identity.json')
    assert identity['training_months'] == TRAIN and identity['query_month'] == '2025-04'
    assert identity['features'] == FEATURES and identity['categorical'] == CAT and identity['numerical'] == NUM
    assert identity['no_January_July_files'] and art(__file__) in identity['sources']
    for record in identity['sources']: verify(record)
    settings = identity['settings']
    for k, v in dict(epochs=10, batch=256, seed=42, lr=.001, weight_decay=.0003,
                    arms=ARMS, train_per_airport=4096, bank_per_airport=512, context=16).items():
        assert settings[k] == v
    assert not (OUT/'result.json').exists() and not (OUT/'scored.parquet').exists()
    assert not (OUT/'attempts/score.json').exists(), 'Audit must precede score attempt'
    if stage == 'prefit':
        assert not (OUT/'fit.json').exists() and not (OUT/'attempts/fit.json').exists()
        assert not any((OUT/arm).exists() for arm in ARMS)
    # Current allocation may already be extended for the audit/next stage.
    allocation = read(OUT/'allocation.json')
    assert allocation['authority'] == '/root' and allocation['identity'] == art(OUT/'identity.json')
    return identity


def raw_record(identity, month):
    found = [r for r in identity['sources'] if Path(r['path']).name.startswith('training_'+month+'-01_')]
    assert len(found) == 1 and month in TRAIN+['2025-04']
    return found[0]


def stamps(series):
    t = pd.to_datetime(series, utc=True)
    values = t.astype('datetime64[ms, UTC]').astype('int64').to_numpy()
    return np.where(t.notna(), values, np.nan)


def aliases(frame): return frame.FLIGHT_ID_mvt.astype('string').fillna('').str.strip()


def sample(meta, rows, cap, prefix):
    chosen = []
    for code in np.unique(meta[rows, 2]):
        group = rows[meta[rows, 2] == code]
        ranked = sorted((hashlib.sha256((prefix+':'+str(int(meta[i, 0]))).encode()).digest(),
                         int(meta[i, 0]), int(i)) for i in group)
        chosen.extend(v[2] for v in ranked[:cap])
    return np.array(sorted(chosen, key=lambda i: meta[i, 0]), dtype=np.int64)


def purge(meta, clock, selected, query):
    # Independent keyed joins / time-neighbor checks, not producer index grouping.
    q = meta[query]; ids = set(q[:, 0].tolist()); aircraft = {}
    signatures = set()
    for i in query:
        r = meta[i]
        if r[5] >= 0: aircraft.setdefault(int(r[5]), []).append(int(r[1]))
        if r[3] > 0 and r[4] > 0 and np.isfinite(clock[i]):
            signatures.add((int(r[2]), int(r[1]), int(r[3]), int(r[4]), float(clock[i])))
    aircraft = {k: np.sort(v) for k, v in aircraft.items()}
    removed = []
    for i in selected:
        r = meta[i]; bad = int(r[0]) in ids
        if int(r[5]) in aircraft:
            times = aircraft[int(r[5])]; j = np.searchsorted(times, r[1])
            bad |= (j < len(times) and abs(times[j]-r[1]) <= 120000)
            bad |= (j > 0 and abs(times[j-1]-r[1]) <= 120000)
        bad |= (int(r[2]), int(r[1]), int(r[3]), int(r[4]), float(clock[i])) in signatures
        removed.append(bad)
    removed = np.asarray(removed, bool)
    return selected[~removed], selected[removed]


def make_features(raw, source):
    mapping = dict(airport='ADEP_mvt', dest='ADES_mvt', aircraft_type='AIRCRAFT_TYPE_mvt',
        flight_rule='FLIGHT_RULE_mvt', market_segment='MARKET_SEGMENT_flt',
        flight_type='FLIGHT_TYPE_flt', wake_cat='WK_TBL_CAT_flt', operator='AIRCRAFT_OPERATOR_flt')
    f = pd.DataFrame({k: raw[v] for k, v in mapping.items()})
    for k in ['runway', 'stand']: f[k] = raw.ADEP_mvt+':'+raw[k.upper()+'_mvt']
    t = pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True)
    f['hour'], f['dow'], f['month'] = t.dt.hour, t.dt.dayofweek, t.dt.month
    for name, col in CLOCKS.items(): f['mvt_minus_'+name+'_sec'] = (t-pd.to_datetime(raw[col], utc=True)).dt.total_seconds()
    p = np.asarray(source, np.float32).copy(); assert p.shape[1] == 15 and np.isfinite(p).all()
    for j in range(4): p[:, j] = np.arcsinh(p[:, j]/3600)
    for j in [9, 10]: p[:, j] = np.minimum(10, np.maximum(-10, p[:, j]/2000))
    for j in range(15): f['physical_'+str(j)] = p[:, j]
    gap = f.mvt_minus_sched_sec.to_numpy(float); known = np.isfinite(gap)
    f['physical_15'] = np.clip(np.arcsinh(np.where(known, gap, 0)/7200), -10, 10).astype(np.float32)
    f['physical_16'] = known.astype(np.float32)
    f['missing_flight_id'] = raw.FLIGHT_ID_mvt.isna().astype(float)
    f['airline'] = raw.FLIGHT_mvt.astype('string').str[:3]
    return f[FEATURES]


def donor_counts(q, qc, bank, bc, chunk=256):
    result = np.zeros(len(q), np.int64)
    # Same-airport restriction is applied before forming pair matrices.
    for airport in np.unique(q[:, 2]):
        donors = np.flatnonzero(bank[:, 2] == airport); b = bank[donors]; c = bc[donors]
        positions = np.flatnonzero(q[:, 2] == airport)
        for start in range(0, len(positions), chunk):
            ix = positions[start:start+chunk]; a = q[ix]
            dt = a[:, 1, None]-b[None, :, 1]
            valid = (a[:, 1, None]//86400000 != b[None, :, 1]//86400000)
            valid &= a[:, 0, None] != b[None, :, 0]
            valid &= ~((a[:, 6, None] > 0) & (a[:, 6, None] == b[None, :, 6]))
            valid &= ~((a[:, 5, None] >= 0) & (a[:, 5, None] == b[None, :, 5]) & (abs(dt) <= 120000))
            signature = ((a[:, 3, None] > 0) & (a[:, 4, None] > 0) & np.isfinite(qc[ix, None]) &
                (dt == 0) & (a[:, 3, None] == b[None, :, 3]) &
                (a[:, 4, None] == b[None, :, 4]) & (qc[ix, None] == c[None, :]))
            valid &= ~signature; result[ix] = valid.sum(axis=1)
    return result


def prefit():
    identity = guard('prefit'); prepared = receipt('prepared.json'); flight = receipt('preflight.json')
    assert prepared['no_query_labels_loaded'] and flight['no_query_labels_loaded']
    assert flight['dependencies'] == [art(OUT/'prepared.json')]
    expected_files = {'selection.npz', 'flight_alias_vocabulary.json', 'scaler.json',
                      'bank_train_positions.npy', 'target_scaler.json', 'train_y.npy'}
    expected_files |= {name+'_'+suffix for name in ['train', 'query'] for suffix in
                      ['features.parquet', 'raw_inputs.parquet', 'meta.npy', 'clock.npy', 'cat.npy', 'num.npy']}
    assert {p.name for p in (OUT/'inputs').iterdir()} == expected_files
    assert prepared['artifacts'] == [art(p) for p in sorted((OUT/'inputs').iterdir())]
    assert flight['artifacts'] == [art(OUT/f'{name}_admissible_counts.npy') for name in ['train', 'query']]
    assert flight['capacity_loss_finite'] and flight['no_optimizer_fit']
    manifest = read(BANK/'manifest.json')
    assert manifest['no_labels_loaded'] and manifest['ranking_view'] == 'train'
    meta = np.load(BANK/'meta.npy', mmap_mode='r'); values = np.load(BANK/'values.npy', mmap_mode='r')
    clock = np.load(BANK/'clock.npy', mmap_mode='r')
    assert meta.shape == (manifest['rows'], 6) and np.all(np.diff(meta[:, 0]) > 0)
    months = pd.to_datetime(meta[:, 1], unit='ms', utc=True).strftime('%Y-%m').to_numpy()
    selection = dict(np.load(OUT/'inputs/selection.npz'))
    query = np.flatnonzero(months == '2025-04')
    sampled = sample(meta, np.flatnonzero(np.isin(months, TRAIN)), 4096, 'retrieval42')
    train, removed = purge(meta, clock, sampled, query)
    eq(selection['query'], query); eq(selection['sampled'], sampled); eq(selection['meta_purged'], removed)
    columns = identity['raw_columns']
    assert 'TAXITIME_SEC_mvt' not in columns and 'BLOCK_TIME_UTC_mvt' not in columns
    raw_parts = {'train': [], 'query': []}
    for month in TRAIN+['2025-04']:
        raw = pd.read_parquet(verify(raw_record(identity, month)), columns=columns,
                              filters=[('PHASE_mvt', '==', 'DEP')])
        assert raw.MVT_ID_mvt.is_unique and raw.PHASE_mvt.eq('DEP').all()
        # Authenticate full source population, not just rows surviving selection.
        eq(np.sort(raw.MVT_ID_mvt.to_numpy()), meta[months == month, 0])
        name, rows = ('query', query) if month == '2025-04' else ('train', train)
        raw_parts[name].append(raw.loc[raw.MVT_ID_mvt.isin(meta[rows[months[rows] == month], 0])])
    raw = {}
    for name, rows in [('train', train), ('query', query)]:
        f = pd.concat(raw_parts[name], ignore_index=True).set_index('MVT_ID_mvt', drop=False)
        raw[name] = f.loc[meta[rows, 0]].reset_index(drop=True)
    del raw_parts
    bad = aliases(raw['train']).isin(set(aliases(raw['query']))-{''}).to_numpy()
    eq(selection['flight_purged'], train[bad]); train = train[~bad]
    raw['train'] = raw['train'].loc[~bad].reset_index(drop=True)
    eq(selection['train'], train); eq(selection['purged'], np.sort(np.r_[removed, selection['flight_purged']]))
    bank = sample(meta, train, 512, 'bank42'); eq(selection['bank'], bank)
    positions = np.searchsorted(meta[train, 0], meta[bank, 0])
    eq(np.load(OUT/'inputs/bank_train_positions.npy'), positions); eq(train[positions], bank)
    vocab = sorted((set(aliases(raw['train'])) | set(aliases(raw['query'])))-{''})
    assert read(OUT/'inputs/flight_alias_vocabulary.json')['values'] == vocab
    vocab_map = {v: i+1 for i, v in enumerate(vocab)}
    frames, packet_meta, packet_clock = {}, {}, {}
    for name, rows in [('train', train), ('query', query)]:
        f = raw[name]; stored = pd.read_parquet(OUT/f'inputs/{name}_raw_inputs.parquet')
        assert list(stored) == columns
        pd.testing.assert_frame_equal(f, stored)
        eq(stamps(f.MVT_TIME_UTC_mvt), meta[rows, 1]); eq(stamps(f.AOBT_3_flt), clock[rows])
        for j, key, col in [(2, 'airport', 'ADEP_mvt'), (3, 'stand', 'STAND_mvt'), (4, 'runway', 'RUNWAY_mvt')]:
            mapping = {v: i+1 for i, v in enumerate(manifest['vocabulary'][key])}
            eq(f[col].astype('string').fillna('').map(mapping).fillna(0).to_numpy(np.int64), meta[rows, j])
        alias_code = aliases(f).map(vocab_map).fillna(0).to_numpy(np.int64)
        packet_meta[name] = np.column_stack([meta[rows], alias_code]); packet_clock[name] = clock[rows]
        eq(np.load(OUT/f'inputs/{name}_meta.npy'), packet_meta[name])
        eq(np.load(OUT/f'inputs/{name}_clock.npy'), packet_clock[name])
        frames[name] = make_features(f, values[rows])
        stored = pd.read_parquet(OUT/f'inputs/{name}_features.parquet')
        assert list(stored) == ['mvt_id']+FEATURES; eq(stored.mvt_id, meta[rows, 0])
        pd.testing.assert_frame_equal(stored[FEATURES], frames[name])
    scaler = read(OUT/'inputs/scaler.json'); x = frames['train'][NUM].to_numpy(float)
    x[~np.isfinite(x)] = np.nan
    median = np.nan_to_num(np.nanmedian(x, axis=0), nan=0.)
    quantiles = np.nanquantile(x, [.25, .75], axis=0); scale = quantiles[1]-quantiles[0]
    scale[~np.isfinite(scale) | (scale == 0)] = 1
    eq(scaler['median'], median); eq(scaler['scale'], scale)
    expected_categories = {c: sorted(frames['train'][c].dropna().astype(str).unique()) for c in CAT}
    assert scaler['categories'] == expected_categories
    for name, frame in frames.items():
        cats = np.column_stack([frame[c].astype('string').map({v: j+1 for j, v in enumerate(expected_categories[c])})
                                .fillna(0).to_numpy(np.int64) for c in CAT])
        x = frame[NUM].to_numpy(float); missing = ~np.isfinite(x)
        standardized = np.clip((np.where(missing, median, x)-median)/scale, -10, 10)
        eq(np.load(OUT/f'inputs/{name}_cat.npy'), cats)
        eq(np.load(OUT/f'inputs/{name}_num.npy'), np.column_stack([standardized, missing]).astype(np.float32))
        coverage = {c: frame[c].astype('string').fillna('<missing>').value_counts().to_dict()
                    for c in ['airport', 'aircraft_type', 'flight_rule', 'market_segment', 'flight_type']}
        assert prepared['coverage'][name] == coverage
    labels = []
    for month in TRAIN:
        ids = meta[train[months[train] == month], 0]
        labels.append(pd.read_parquet(verify(raw_record(identity, month)), columns=['MVT_ID_mvt', 'TAXITIME_SEC_mvt'],
            filters=[('PHASE_mvt', '==', 'DEP'), ('MVT_ID_mvt', 'in', ids.tolist())]))
    labels = pd.concat(labels).set_index('MVT_ID_mvt'); assert labels.index.is_unique
    y = labels.loc[meta[train, 0], 'TAXITIME_SEC_mvt'].to_numpy(float)
    assert np.isfinite(y).all(); eq(np.load(OUT/'inputs/train_y.npy'), y)
    expected_target = dict(mean=float(y.mean()), std=float(y.std(ddof=0)), clipping=None)
    assert expected_target['std'] > 0 and read(OUT/'inputs/target_scaler.json') == expected_target
    assert prepared['target_policy'] == expected_target
    expected_rows = dict(train=len(train), query=len(query), bank=len(bank), sampled=len(sampled),
        purged=len(selection['purged']), meta_alias_purged=len(removed), flight_id_purged=int(bad.sum()))
    assert prepared['rows'] == expected_rows
    minimum = {}
    for name in ['train', 'query']:
        counts = donor_counts(packet_meta[name], packet_clock[name], packet_meta['train'][positions], packet_clock['train'][positions])
        eq(np.load(OUT/f'{name}_admissible_counts.npy'), counts)
        minimum[name] = int(counts.min()); assert minimum[name] >= 16
    assert flight['minimum_admissible'] == minimum
    assert flight['checks']['loader']['query_target_poison_and_removal_exact']
    for name in ['cpu']+(['device'] if flight['device'] != 'cpu' else []):
        for key, value in flight['checks'][name].items():
            if key != 'device': assert value is True, (name, key)
    save(OUT/'prefit_audit.json', dict(status='pass', source=art(__file__), identity=art(OUT/'identity.json'),
        prepared=art(OUT/'prepared.json'), preflight=art(OUT/'preflight.json'), rows=expected_rows,
        minimum_admissible=minimum, raw37_exact=True, all_input_selection_scaling_and_aliases_exact=True,
        training_uncapped_targets_exact=True, no_April_target_read=True, no_January_July_raw_file_read=True,
        physical_source_scope='Exact pinned label-free 15-scalar bank reused; no reconstruction of upstream sensors',
        independent_donor_mask='NumPy same-airport blocks, all train and April queries'))
    print(json.dumps(dict(status='pass', stage='prefit', rows=expected_rows)))


def finite_tree(value):
    if isinstance(value, torch.Tensor): assert torch.isfinite(value).all()
    elif isinstance(value, dict):
        for v in value.values(): finite_tree(v)
    elif isinstance(value, (tuple, list)):
        for v in value: finite_tree(v)
    elif isinstance(value, float): assert math.isfinite(value)


def rng_hash(state):
    h = hashlib.sha256(state['torch_cpu'].numpy().tobytes())
    for v in state['torch_cuda']: h.update(v.numpy().tobytes())
    h.update(json.dumps(state['numpy'], sort_keys=True).encode()); return h.hexdigest()


def prescore():
    identity = guard('prescore'); prepared = receipt('prepared.json'); preflight = receipt('preflight.json')
    fit = receipt('fit.json'); audit = read(OUT/'prefit_audit.json')
    assert audit['status'] == 'pass' and audit['identity'] == art(OUT/'identity.json')
    assert audit['prepared'] == art(OUT/'prepared.json') and audit['preflight'] == art(OUT/'preflight.json')
    assert fit['arms'] == ARMS and fit['no_query_labels_loaded']
    assert fit['dependencies'] == [art(OUT/arm/'inference.json') for arm in ARMS]
    query = np.load(OUT/'inputs/query_meta.npy'); train = np.load(OUT/'inputs/train_meta.npy')
    assert np.all(np.diff(query[:, 0]) > 0) and np.all(np.diff(train[:, 0]) > 0)
    scaler = read(OUT/'inputs/scaler.json'); cards = [len(scaler['categories'][c])+1 for c in CAT]
    # Reconstruct initialization only, never learned inference. The frozen model
    # has no input I/O and construction is isolated from caller RNG state.
    model_source = ROOT/'VALIDATION/retrieval_label_model.py'
    assert art(model_source) in identity['sources']
    spec = importlib.util.spec_from_file_location('reviewed_retrieval_model', model_source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        initial = module.Retrieval(cards, 2*len(NUM)).state_dict()
    initial_sha = hashlib.sha256(b''.join(v.numpy().tobytes() for v in initial.values())).hexdigest()
    rng = np.random.default_rng(42); order_hash = hashlib.sha256()
    for _ in range(10): order_hash.update(rng.permutation(len(train)).tobytes())
    updates = 10*math.ceil(len(train)/256); records = []; previous = None
    for arm in ARMS:
        trained = receipt(arm+'/training.json'); inference = receipt(arm+'/inference.json')
        assert trained['arm'] == inference['arm'] == arm
        assert trained['no_query_labels_loaded'] and inference['no_query_labels_loaded']
        assert trained['allocation'] == fit['allocation']; verify(trained['allocation'])
        assert trained['dependencies'] == [art(OUT/n) for n in ['prepared.json', 'preflight.json', 'prefit_audit.json']]
        assert trained['artifacts'] == [art(OUT/arm/'model.pt')]
        assert inference['dependencies'] == [art(OUT/arm/'training.json')]
        assert inference['artifacts'] == [art(OUT/arm/'predictions.parquet')]
        assert trained['runtime'] == preflight['runtime']
        assert trained['updates'] == updates and trained['permutation_sha256'] == order_hash.hexdigest()
        assert [c['epoch'] for c in trained['curve']] == list(range(1, 11))
        assert all(math.isfinite(c['normalized_train_MSE']) and c['normalized_train_MSE'] >= 0 for c in trained['curve'])
        checkpoint = torch.load(OUT/arm/'model.pt', map_location='cpu', weights_only=False)
        assert checkpoint['identity'] == art(OUT/'identity.json') and checkpoint['arm'] == arm
        assert checkpoint['cards'] == cards and checkpoint['numerical_width'] == 2*len(NUM)
        finite_tree(checkpoint)
        state = checkpoint['state']; assert state['label.weight'].shape == (128, 1)
        assert list(state) == list(initial)
        for key in state: assert state[key].shape == initial[key].shape
        assert trained['initial_state_sha256'] == initial_sha
        if arm == 'labels_off': assert torch.count_nonzero(state['label.weight']) == 0
        terminal = checkpoint['rng']; assert terminal['numpy'] == rng.bit_generator.state
        assert rng_hash(terminal) == trained['terminal_rng_sha256']
        optimizer = checkpoint['optimizer']; assert len(optimizer['param_groups']) == 1
        group = optimizer['param_groups'][0]
        assert group['lr'] == .001 and group['weight_decay'] == .0003 and group['betas'] == (.9, .999)
        assert len(optimizer['state']) == len(group['params']) == len(state)
        assert all(int(v['step']) == updates for v in optimizer['state'].values())
        if previous is not None:
            old, old_trained = previous
            assert trained['initial_state_sha256'] == old_trained['initial_state_sha256']
            assert trained['terminal_rng_sha256'] == old_trained['terminal_rng_sha256']
            eq(terminal['torch_cpu'], old['rng']['torch_cpu'])
            assert len(terminal['torch_cuda']) == len(old['rng']['torch_cuda'])
            for a, b in zip(terminal['torch_cuda'], old['rng']['torch_cuda']): eq(a, b)
            assert list(state) == list(old['state'])
            for key in state: assert state[key].shape == old['state'][key].shape
        previous = checkpoint, trained
        prediction = pd.read_parquet(OUT/arm/'predictions.parquet')
        assert list(prediction) == ['mvt_id', 'prediction', 'serialized'] and prediction.mvt_id.is_unique
        eq(prediction.mvt_id, query[:, 0]); assert np.isfinite(prediction[['prediction', 'serialized']]).all().all()
        eq(prediction.serialized, np.round(np.clip(prediction.prediction.to_numpy(), 0, None)))
        assert inference['rows'] == len(query) and inference['exact_reload']
        records.append(dict(arm=arm, training=art(OUT/arm/'training.json'), inference=art(OUT/arm/'inference.json'),
                            model=art(OUT/arm/'model.pt'), prediction=art(OUT/arm/'predictions.parquet')))
    save(OUT/'prescore_audit.json', dict(status='pass', source=art(__file__), identity=art(OUT/'identity.json'),
        fit=art(OUT/'fit.json'), prefit=art(OUT/'prefit_audit.json'), members=records,
        updates_per_arm=updates, all_parameters_and_optimizer_states_finite=True,
        initial_state_reconstructed_exact=True, matched_actual_terminal_RNG=True, full_query_ID_rounding_exact=True,
        no_April_target_read=True, no_model_inference_repeated=True,
        reload_scope='Exact producer all-query reload receipts; independent audit checks saved state and artifacts'))
    print(json.dumps(dict(status='pass', stage='prescore', rows=len(query), updates=updates)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['prefit', 'prescore'])
    args = parser.parse_args(); torch.set_num_threads(1)
    {'prefit': prefit, 'prescore': prescore}[args.stage]()
