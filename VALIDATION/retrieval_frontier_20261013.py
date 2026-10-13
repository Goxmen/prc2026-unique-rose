"""Fixed small learned train-label retrieval falsifier; never a submission.

Preparation reads April inputs only. Both fixed arms save all predictions before
the separate scoring stage is allowed to decode April targets. No Jan/Jul files.
"""
import argparse
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

import retrieval_label_model as model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import common

OUT = ROOT/'VALIDATION/runs/v3_20261013_train_label_retrieval'
PLAN = ROOT/'VALIDATION/reports/retrieval_frontier_20261013/protocol.json'
BANK = ROOT/'data/derived/flight_cohort_context_20261012/train'
TRAIN = [f'2025-{m:02d}' for m in [2, 3, 5, 6, 8, 9, 10, 11, 12]]
QUERY = '2025-04'
ARMS = model.SETTINGS['arms']
CAT = common.CATEGORICAL + ['airline']
FEATURES = common.CATEGORICAL + ['hour', 'dow', 'month'] + [
    'mvt_minus_'+name+'_sec' for name in ['sched', 'lobt', 'iobt', 'eobt1', 'aobt3']
] + ['physical_'+str(i) for i in range(17)] + ['missing_flight_id', 'airline']
NUM = [s for s in FEATURES if s not in CAT]
RAW = ['MVT_ID_mvt', 'PHASE_mvt', 'FLIGHT_ID_mvt', 'FLIGHT_mvt'] + [
    s.strip() for s in common.RAW_COLS.replace('\n', '').split(',')]
PREFIT_AUDITOR = ROOT/'VALIDATION/tools/audit_retrieval_frontier_prefit_20261013.py'
RESULT_AUDITOR = ROOT/'VALIDATION/tools/audit_retrieval_frontier_result_20261013.py'


def read(p): return json.loads(Path(p).read_text())


def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); tmp.replace(p)


def art(p):
    p = Path(p).resolve(); h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): h.update(block)
    return dict(path=str(p.relative_to(ROOT)), sha256=h.hexdigest(), bytes=p.stat().st_size)


def verify(record):
    p = ROOT/record['path']; assert art(p) == record, p
    return p


def raw_path(month):
    paths = list((ROOT/'data').glob('training_'+month+'-01_*.parquet'))
    assert len(paths) == 1
    return paths[0]


def guard():
    identity = read(OUT/'identity.json')
    assert identity['settings'] == model.SETTINGS
    assert identity['training_months'] == TRAIN and identity['query_month'] == QUERY
    assert identity['features'] == FEATURES
    for r in identity['sources']: verify(r)
    return identity


def receipt(path):
    value = read(path)
    assert value['identity'] == art(OUT/'identity.json'), path
    for key in ['artifacts', 'dependencies']:
        for r in value.get(key, []): verify(r)
    return value


def start(stage):
    guard()
    allocation = read(OUT/'allocation.json')
    assert allocation['authority'] == '/root' and allocation['identity'] == art(OUT/'identity.json')
    assert stage in allocation['stages'], (stage, allocation)
    p = OUT/'attempts'/f'{stage}.json'; p.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive, durable attempt marker: no automatic restart or lost-fit retry.
    with p.open('x') as stream:
        json.dump(dict(stage=stage, pid=os.getpid(), time=time.time(),
            identity=art(OUT/'identity.json'), allocation=art(OUT/'allocation.json')), stream)
    return art(OUT/'allocation.json')


def freeze():
    assert not (OUT/'identity.json').exists()
    review = read(OUT/'source_review.json'); assert review['status'] == 'pass'
    sources = [Path(__file__), Path(model.__file__), Path(common.__file__), PLAN,
        PREFIT_AUDITOR, RESULT_AUDITOR, OUT/'source_review.json', BANK/'manifest.json']
    plan = read(PLAN)
    assert plan['features'] == FEATURES and plan['architecture'] == model.SETTINGS
    assert plan['train_months'] == TRAIN and plan['query_month'] == QUERY
    sources += [verify(r) for r in plan['source_metadata']+[plan['historical_assessment']]]
    manifest = read(BANK/'manifest.json')
    assert manifest['no_labels_loaded'] and manifest['ranking_view'] == 'train'
    verify(manifest['source'])
    sources += [ROOT/manifest['source']['path']]
    bank_records = [r for r in manifest['outputs'] if Path(r['path']).name in ['meta.npy', 'values.npy', 'clock.npy']]
    assert len(bank_records) == 3
    sources += [verify(r) for r in bank_records]
    sources += [raw_path(month) for month in TRAIN+[QUERY]]
    # The review explicitly authenticates executable sources before data access.
    for p in [Path(__file__), Path(model.__file__), PREFIT_AUDITOR, RESULT_AUDITOR, PLAN]:
        assert art(p) in review['sources'], p
    write(OUT/'identity.json', dict(settings=model.SETTINGS, training_months=TRAIN,
        query_month=QUERY, features=FEATURES, categorical=CAT, numerical=NUM,
        raw_columns=RAW, sources=[art(p) for p in dict.fromkeys(sources)],
        no_January_July_files=True, no_real_arrays_decoded_before_freeze=True,
        experiment='small mechanism falsifier; not current-v80 validation'))


def hash_select(meta, candidates, maximum, prefix):
    selected = []
    for airport in sorted(set(meta[candidates, 2])):
        rows = candidates[meta[candidates, 2] == airport]
        ordered = sorted(rows, key=lambda i: (hashlib.sha256(
            f'{prefix}:{int(meta[i, 0])}'.encode()).digest(), int(meta[i, 0])))
        selected.extend(ordered[:maximum])
    return np.array(sorted(selected, key=lambda i: int(meta[i, 0])), np.int64)


def alias_purge(meta, clock, selected, queries):
    """Input-only selected-training aliases to April; no resampling or labels."""
    q, s = meta[queries], meta[selected]
    remove = np.isin(s[:, 0], q[:, 0])
    def aircraft_groups(array):
        order = np.argsort(array[:, 5], kind='stable')
        cuts = np.r_[0, 1+np.flatnonzero(np.diff(array[order, 5])), len(order)]
        return {int(array[part[0], 5]): part for part in np.split(order, cuts[1:-1])
            if len(part) and array[part[0], 5] >= 0}
    sg, qg = aircraft_groups(s), aircraft_groups(q)
    for aircraft in sg.keys() & qg.keys():
        sq = sg[aircraft]
        times = np.sort(q[qg[aircraft], 1])
        left = np.searchsorted(times, s[sq, 1]-120000)
        right = np.searchsorted(times, s[sq, 1]+120000, side='right')
        remove[sq] |= right > left
    def signatures(rows):
        return {(int(meta[i, 2]), int(meta[i, 1]), int(meta[i, 3]), int(meta[i, 4]), float(clock[i]))
            for i in rows if meta[i, 3] > 0 and meta[i, 4] > 0 and np.isfinite(clock[i])}
    known = signatures(queries)
    for j, i in enumerate(selected):
        if (int(meta[i, 2]), int(meta[i, 1]), int(meta[i, 3]), int(meta[i, 4]), float(clock[i])) in known:
            remove[j] = True
    return selected[~remove], selected[remove]


def physical(values, gap):
    x = np.asarray(values, np.float32).copy()
    assert x.shape[1] == 15 and np.isfinite(x).all()
    x[:, :4] = np.arcsinh(x[:, :4]/3600)
    x[:, 9:11] = np.clip(x[:, 9:11]/2000, -10, 10)
    present = np.isfinite(gap)
    z = np.clip(np.arcsinh(np.where(present, gap, 0)/7200), -10, 10)
    return np.column_stack([x, z, present]).astype(np.float32)


def features(raw, values):
    f = common.build_features(raw)
    for i, column in enumerate(physical(values, f.mvt_minus_sched_sec.to_numpy(float)).T):
        f['physical_'+str(i)] = column
    f['missing_flight_id'] = raw.FLIGHT_ID_mvt.isna().astype(float)
    f['airline'] = raw.FLIGHT_mvt.astype('string').str[:3]
    return f[FEATURES]


def projected_inputs(path, loader=pd.read_parquet):
    """Actual preparation boundary; target columns never requested or retained."""
    frame = loader(path, columns=RAW, filters=[('PHASE_mvt', '==', 'DEP')])
    # Defensively project even a nonconforming loader that returns extra columns.
    return frame.loc[:, RAW]


def flight_ids(raw):
    return raw.FLIGHT_ID_mvt.astype('string').fillna('').str.strip()


def loader_checks():
    """Synthetic April-label poison/removal at the actual projection boundary."""
    raw = pd.DataFrame({s: [None, None] for s in RAW})
    raw['MVT_ID_mvt'] = [1, 2]; raw['PHASE_mvt'] = 'DEP'
    calls = []
    def loader(poison):
        def load(path, columns, filters):
            assert columns == RAW and 'TAXITIME_SEC_mvt' not in columns
            calls.append((str(path), columns, filters))
            result = raw.copy()
            if poison is not None: result['TAXITIME_SEC_mvt'] = poison
            return result
        return load
    a = projected_inputs('synthetic_april', loader([1., 2.]))
    b = projected_inputs('synthetic_april', loader([1e100, -1e100]))
    c = projected_inputs('synthetic_april', loader(None))
    pd.testing.assert_frame_equal(a, b); pd.testing.assert_frame_equal(a, c)
    return dict(query_target_poison_and_removal_exact=True, projected_columns=RAW, calls=len(calls))


def fit_scaler(frame):
    maps = {s: sorted(frame[s].dropna().astype(str).unique().tolist()) for s in CAT}
    x = frame[NUM].to_numpy(float); x[~np.isfinite(x)] = np.nan
    median = np.nanmedian(x, axis=0)
    # Completely missing train columns have explicit missing flags and neutral scale.
    median = np.nan_to_num(median, nan=0.)
    quartile = np.nanquantile(x, [.25, .75], axis=0)
    scale = quartile[1]-quartile[0]
    scale[~np.isfinite(scale) | (scale == 0)] = 1
    return dict(categories=maps, median=median.tolist(), scale=scale.tolist())


def transform(frame, scaler):
    cat = np.column_stack([frame[s].astype('string').map(
        {value: i+1 for i, value in enumerate(scaler['categories'][s])}).fillna(0).to_numpy(np.int64) for s in CAT])
    x = frame[NUM].to_numpy(float); missing = ~np.isfinite(x)
    x = (np.where(missing, scaler['median'], x)-np.asarray(scaler['median']))/np.asarray(scaler['scale'])
    num = np.column_stack([np.clip(x, -10, 10), missing]).astype(np.float32)
    assert np.isfinite(num).all()
    return cat, num


def prepare():
    allocation = start('prepare'); started = time.monotonic()
    dest = OUT/'inputs'; assert not dest.exists(); dest.mkdir()
    meta = np.load(BANK/'meta.npy', mmap_mode='r')
    values = np.load(BANK/'values.npy', mmap_mode='r')
    clock = np.load(BANK/'clock.npy', mmap_mode='r')
    assert np.all(np.diff(meta[:, 0]) > 0) and meta.shape[1] == 6
    months = pd.to_datetime(meta[:, 1], unit='ms', utc=True).strftime('%Y-%m').to_numpy()
    query = np.flatnonzero(months == QUERY)
    candidates = np.flatnonzero(np.isin(months, TRAIN))
    sampled = hash_select(meta, candidates, model.SETTINGS['train_per_airport'], 'retrieval42')
    train, meta_purged = alias_purge(meta, clock, sampled, query)
    raws = {}
    for name, rows, allowed in [('train', train, TRAIN), ('query', query, [QUERY])]:
        frames = []
        for month in allowed:
            ids = meta[rows[months[rows] == month], 0]
            raw = projected_inputs(raw_path(month))
            frames.append(raw.loc[raw.MVT_ID_mvt.isin(ids)])
        raw = pd.concat(frames, ignore_index=True).set_index('MVT_ID_mvt', drop=False)
        assert raw.index.is_unique
        raw = raw.loc[meta[rows, 0]].reset_index(drop=True)
        np.testing.assert_array_equal(raw.MVT_ID_mvt, meta[rows, 0])
        assert raw.PHASE_mvt.eq('DEP').all()
        stamps = pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True).astype('datetime64[ms, UTC]').astype('int64').to_numpy()
        np.testing.assert_array_equal(stamps, meta[rows, 1])
        raws[name] = raw
    query_flight = set(flight_ids(raws['query']))-{''}
    flight_purge_mask = flight_ids(raws['train']).isin(query_flight).to_numpy()
    flight_purged = train[flight_purge_mask]
    train = train[~flight_purge_mask]
    raws['train'] = raws['train'].loc[~flight_purge_mask].reset_index(drop=True)
    bank = hash_select(meta, train, model.SETTINGS['bank_per_airport'], 'bank42')
    assert len(query) and len(train) and set(meta[query, 2]) == set(meta[train, 2]) == set(meta[bank, 2])
    purged = np.sort(np.r_[meta_purged, flight_purged])
    np.savez(dest/'selection.npz', train=train, query=query, bank=bank, purged=purged,
        meta_purged=meta_purged, flight_purged=flight_purged, sampled=sampled)
    alias_vocab = sorted((set(flight_ids(raws['query'])) | set(flight_ids(raws['train'])))-{''})
    alias_map = {value: i+1 for i, value in enumerate(alias_vocab)}
    write(dest/'flight_alias_vocabulary.json', dict(values=alias_vocab,
        scope='input-only alias codes; never predictive features or fitted categories'))
    packets = {}
    for name, rows in [('train', train), ('query', query)]:
        raw = raws[name]
        frame = features(raw, values[rows]); frame.insert(0, 'mvt_id', meta[rows, 0])
        frame.to_parquet(dest/f'{name}_features.parquet', index=False)
        raw.to_parquet(dest/f'{name}_raw_inputs.parquet', index=False)
        flight_code = flight_ids(raw).map(alias_map).fillna(0).to_numpy(np.int64)
        np.save(dest/f'{name}_meta.npy', np.column_stack([meta[rows], flight_code]))
        np.save(dest/f'{name}_clock.npy', clock[rows])
        packets[name] = frame
    scaler = fit_scaler(packets['train']); write(dest/'scaler.json', scaler)
    for name, frame in packets.items():
        cat, num = transform(frame, scaler)
        np.save(dest/f'{name}_cat.npy', cat); np.save(dest/f'{name}_num.npy', num)
    bank_positions = np.searchsorted(meta[train, 0], meta[bank, 0])
    np.testing.assert_array_equal(meta[train[bank_positions], 0], meta[bank, 0])
    np.save(dest/'bank_train_positions.npy', bank_positions)
    # Target read happens only after all input selections/preprocessing are saved.
    labels = []
    for month in TRAIN:
        ids = meta[train[months[train] == month], 0]
        f = pd.read_parquet(raw_path(month), columns=['MVT_ID_mvt', 'TAXITIME_SEC_mvt'],
            filters=[('PHASE_mvt', '==', 'DEP'), ('MVT_ID_mvt', 'in', ids.tolist())])
        labels.append(f)
    labels = pd.concat(labels, ignore_index=True).set_index('MVT_ID_mvt')
    assert labels.index.is_unique
    y = labels.loc[meta[train, 0], 'TAXITIME_SEC_mvt'].to_numpy(float)
    assert np.isfinite(y).all(), 'Nonfinite selected training target; no resampling permitted'
    target = dict(mean=float(y.mean()), std=float(y.std(ddof=0)), clipping=None)
    assert target['std'] > 0 and math.isfinite(target['std'])
    write(dest/'target_scaler.json', target); np.save(dest/'train_y.npy', y)
    coverage = {}
    for name, frame in packets.items():
        coverage[name] = {s: frame[s].astype('string').fillna('<missing>').value_counts().to_dict()
            for s in ['airport', 'aircraft_type', 'flight_rule', 'market_segment', 'flight_type']}
    write(OUT/'prepared.json', dict(identity=art(OUT/'identity.json'), allocation=allocation,
        artifacts=[art(p) for p in sorted(dest.iterdir())], rows=dict(train=len(train), query=len(query),
        bank=len(bank), sampled=len(sampled), purged=len(purged),
        meta_alias_purged=len(meta_purged), flight_id_purged=len(flight_purged)), coverage=coverage,
        no_query_labels_loaded=True, target_policy=target, seconds=time.monotonic()-started,
        query_population='all April input-bank departures, no target-based filtering'))


def load_inputs(device):
    receipt(OUT/'prepared.json'); dest = OUT/'inputs'
    result = {}
    for name in ['train', 'query']:
        result[name] = {s: torch.as_tensor(np.load(dest/f'{name}_{s}.npy'), device=device)
            for s in ['cat', 'num', 'meta', 'clock']}
    scaler = read(dest/'scaler.json'); target = read(dest/'target_scaler.json')
    result['cards'] = [len(scaler['categories'][s])+1 for s in CAT]
    result['target'] = target
    result['y'] = torch.as_tensor((np.load(dest/'train_y.npy')-target['mean'])/target['std'],
        dtype=torch.float32, device=device)
    result['bank_pos'] = torch.as_tensor(np.load(dest/'bank_train_positions.npy'), device=device)
    result['bank'] = {k: v[result['bank_pos']] for k, v in result['train'].items()}
    assert torch.all(torch.diff(result['bank']['meta'][:, 0]) > 0), 'Canonical donor-ID tie order'
    result['bank_y'] = result['y'][result['bank_pos']]
    return result


def batch(data, name, rows):
    q, b = data[name], data['bank']
    valid = model.admissible(q['meta'][rows], q['clock'][rows], b['meta'], b['clock'])
    return q['cat'][rows], q['num'][rows], b['cat'], b['num'], data['bank_y'], valid


def preflight(device):
    allocation = start('preflight'); data = load_inputs(device)
    reports = dict(cpu=model.synthetic('cpu'), loader=loader_checks())
    if device != 'cpu': reports['device'] = model.synthetic(device)
    minimum = {}; width = model.SETTINGS['batch']
    for name in ['train', 'query']:
        counts = []
        for i in range(0, len(data[name]['cat']), width):
            mask = batch(data, name, slice(i, i+width))[-1]
            counts.append(mask.sum(1).cpu().numpy())
        counts = np.concatenate(counts)
        assert counts.min() >= model.SETTINGS['context']
        np.save(OUT/f'{name}_admissible_counts.npy', counts)
        minimum[name] = int(counts.min())
    torch.manual_seed(42)
    net = model.Retrieval(data['cards'], data['train']['num'].shape[1]).to(device)
    args = batch(data, 'train', slice(0, width))
    if device != 'cpu': torch.cuda.reset_peak_memory_stats()
    loss = (net(*args, True)-data['y'][:width]).square().mean(); loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
    write(OUT/'preflight.json', dict(identity=art(OUT/'identity.json'), allocation=allocation,
        dependencies=[art(OUT/'prepared.json')], artifacts=[art(OUT/f'{s}_admissible_counts.npy')
        for s in ['train', 'query']], checks=reports, minimum_admissible=minimum,
        device=device, capacity_loss_finite=True, no_optimizer_fit=True, no_query_labels_loaded=True,
        runtime=runtime(device),
        peak_device_bytes=torch.cuda.max_memory_allocated() if device != 'cpu' else None))


def runtime(device):
    return dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
        pandas=pd.__version__, device=device, cuda=torch.version.cuda,
        GPU=torch.cuda.get_device_name() if device != 'cpu' else None)


def rng_state(rng):
    return dict(torch_cpu=torch.get_rng_state().cpu(),
        torch_cuda=[v.cpu() for v in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
        numpy=rng.bit_generator.state)


def rng_digest(value):
    h = hashlib.sha256(value['torch_cpu'].numpy().tobytes())
    for v in value['torch_cuda']: h.update(v.numpy().tobytes())
    h.update(json.dumps(value['numpy'], sort_keys=True).encode())
    return h.hexdigest()


@torch.no_grad()
def predict(net, data, arm):
    net.eval(); parts = []
    for i in range(0, len(data['query']['cat']), model.SETTINGS['batch']):
        value = net(*batch(data, 'query', slice(i, i+model.SETTINGS['batch'])), arm == 'labels_on')
        parts.append(value.cpu().numpy())
    value = np.concatenate(parts).astype(float)*data['target']['std']+data['target']['mean']
    assert np.isfinite(value).all()
    return value


def fit(device):
    allocation = start('fit')
    audit = read(OUT/'prefit_audit.json'); assert audit['status'] == 'pass'
    assert audit['identity'] == art(OUT/'identity.json')
    assert audit['prepared'] == art(OUT/'prepared.json') and audit['preflight'] == art(OUT/'preflight.json')
    data = load_inputs(device); preflight_receipt = receipt(OUT/'preflight.json')
    assert preflight_receipt['device'] == device and preflight_receipt['runtime'] == runtime(device)
    completed = []; initial_sha = None; terminal_rng_sha = None
    for arm in ARMS:
        dest = OUT/arm; assert not dest.exists(); dest.mkdir()
        torch.manual_seed(model.SETTINGS['seed'])
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(model.SETTINGS['seed'])
        rng = np.random.default_rng(model.SETTINGS['seed'])
        net = model.Retrieval(data['cards'], data['train']['num'].shape[1]).to(device)
        initial = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in net.state_dict().values())).hexdigest()
        if initial_sha is None: initial_sha = initial
        assert initial == initial_sha
        optimizer = torch.optim.AdamW(net.parameters(), lr=model.SETTINGS['lr'], weight_decay=model.SETTINGS['weight_decay'])
        curve = []; updates = 0; order_hash = hashlib.sha256()
        for epoch in range(model.SETTINGS['epochs']):
            net.train(); order = rng.permutation(len(data['y'])); order_hash.update(order.tobytes())
            losses = []
            for i in range(0, len(order), model.SETTINGS['batch']):
                rows = torch.as_tensor(order[i:i+model.SETTINGS['batch']], device=device)
                optimizer.zero_grad(set_to_none=True)
                output = net(*batch(data, 'train', rows), arm == 'labels_on')
                loss = (output-data['y'][rows]).square().mean()
                assert torch.isfinite(loss)
                loss.backward()
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
                optimizer.step(); updates += 1; losses.append(float(loss.detach()))
            curve.append(dict(epoch=epoch+1, normalized_train_MSE=float(np.mean(losses))))
        assert all(torch.isfinite(p).all() for p in net.parameters())
        terminal = rng_state(rng); terminal_sha = rng_digest(terminal)
        if terminal_rng_sha is None: terminal_rng_sha = terminal_sha
        torch.save(dict(state={k: v.detach().cpu() for k, v in net.state_dict().items()}, cards=data['cards'],
            optimizer=optimizer.state_dict(), rng=terminal,
            numerical_width=data['train']['num'].shape[1], arm=arm, identity=art(OUT/'identity.json')),
            dest/'model.pt')
        assert terminal_sha == terminal_rng_sha, 'Matched dropout and sampling random streams'
        write(dest/'training.json', dict(identity=art(OUT/'identity.json'), arm=arm, allocation=allocation,
            initial_state_sha256=initial, permutation_sha256=order_hash.hexdigest(), updates=updates,
            terminal_rng_sha256=terminal_sha, runtime=runtime(device),
            curve=curve, artifacts=[art(dest/'model.pt')], dependencies=[art(OUT/'prepared.json'),
            art(OUT/'preflight.json'), art(OUT/'prefit_audit.json')], no_query_labels_loaded=True))
        value = predict(net, data, arm)
        saved = torch.load(dest/'model.pt', map_location=device, weights_only=False)
        reloaded = model.Retrieval(saved['cards'], saved['numerical_width']).to(device)
        reloaded.load_state_dict(saved['state'])
        np.testing.assert_array_equal(value, predict(reloaded, data, arm))
        ids = data['query']['meta'][:, 0].cpu().numpy()
        frame = pd.DataFrame(dict(mvt_id=ids, prediction=value, serialized=np.rint(np.maximum(value, 0))))
        frame.to_parquet(dest/'predictions.parquet', index=False)
        write(dest/'inference.json', dict(identity=art(OUT/'identity.json'), arm=arm,
            dependencies=[art(dest/'training.json')], artifacts=[art(dest/'predictions.parquet')],
            rows=len(frame), exact_reload=True, no_query_labels_loaded=True))
        completed.append(art(dest/'inference.json'))
    write(OUT/'fit.json', dict(identity=art(OUT/'identity.json'), allocation=allocation,
        dependencies=completed, artifacts=[], arms=ARMS, no_query_labels_loaded=True))


def score():
    allocation = start('score'); receipt(OUT/'fit.json')
    audit = read(OUT/'prescore_audit.json'); assert audit['status'] == 'pass'
    assert audit['identity'] == art(OUT/'identity.json') and audit['fit'] == art(OUT/'fit.json')
    outputs = []
    for arm in ARMS:
        inference = receipt(OUT/arm/'inference.json'); receipt(OUT/arm/'training.json')
        outputs.append(pd.read_parquet(verify(inference['artifacts'][0])))
    np.testing.assert_array_equal(outputs[0].mvt_id, outputs[1].mvt_id)
    # Sole query-target decoding site. All trained state/predictions already audited.
    raw = pd.read_parquet(raw_path(QUERY), columns=['MVT_ID_mvt', 'PHASE_mvt', 'TAXITIME_SEC_mvt', 'MVT_TIME_UTC_mvt'],
        filters=[('PHASE_mvt', '==', 'DEP')]).set_index('MVT_ID_mvt')
    assert raw.index.is_unique
    f = raw.loc[outputs[0].mvt_id].reset_index()
    y = f.TAXITIME_SEC_mvt.to_numpy(float); finite = np.isfinite(y)
    assert finite.any()
    days = pd.to_datetime(f.MVT_TIME_UTC_mvt, utc=True).dt.strftime('%Y-%m-%d')
    packet = pd.DataFrame(dict(mvt_id=outputs[0].mvt_id, day=days, y=y, finite_target=finite,
        labels_off=outputs[0].serialized, labels_on=outputs[1].serialized))
    packet.to_parquet(OUT/'scored.parquet', index=False)
    scored = packet.loc[finite]
    square = np.column_stack([(scored[s].to_numpy()-scored.y.to_numpy())**2 for s in ARMS])
    unique, inverse = np.unique(scored.day, return_inverse=True)
    n = np.bincount(inverse); sums = np.column_stack([np.bincount(inverse, weights=square[:, i]) for i in range(2)])
    rng = np.random.default_rng(20261013); draws = rng.integers(0, len(unique), (2000, len(unique)))
    boot = np.sqrt(sums[draws].sum(1)/n[draws].sum(1)[:, None])
    delta = boot[:, 1]-boot[:, 0]; np.save(OUT/'bootstrap.npy', delta)
    metrics = np.sqrt(square.mean(0)); interval = np.quantile(delta, [.025, .975])
    per_airport = {}
    airports = pd.read_parquet(OUT/'inputs/query_features.parquet', columns=['mvt_id', 'airport'])
    np.testing.assert_array_equal(airports.mvt_id, packet.mvt_id)
    for airport in sorted(airports.airport.unique()):
        ix = airports.airport.to_numpy()[finite] == airport
        per_airport[airport] = dict(rows=int(ix.sum()), RMSE=np.sqrt(square[ix].mean(0)).tolist())
    write(OUT/'result.json', dict(identity=art(OUT/'identity.json'), allocation=allocation,
        dependencies=[art(OUT/'fit.json'), art(OUT/'prescore_audit.json')],
        artifacts=[art(OUT/'scored.parquet'), art(OUT/'bootstrap.npy')],
        rows=len(packet), scored_rows=int(finite.sum()), nonfinite_target_rows=int((~finite).sum()),
        RMSE=dict(zip(ARMS, metrics.tolist())), delta=float(metrics[1]-metrics[0]), CI95=interval.tolist(),
        gate=bool(metrics[1] < metrics[0] and interval[1] < 0), airports=per_airport,
        scope='April fresh-fit-excluded mechanism screen; no v80 or official score claim',
        failure_scope='Only this fixed37-feature 4096-per-airport/512-donor learned retrieval test'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['freeze', 'prepare', 'preflight', 'fit', 'score'])
    parser.add_argument('--device', default='cuda'); args = parser.parse_args()
    torch.set_num_threads(2)
    {'freeze': freeze, 'prepare': prepare, 'preflight': lambda: preflight(args.device),
        'fit': lambda: fit(args.device), 'score': score}[args.stage]()
