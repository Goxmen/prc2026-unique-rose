"""Audited, label-free airport/UTC-day scalar donor packets for neural training."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import flight_cohort_context as prior
import public_delay_features as public

ROOT = prior.ROOT
OUT = ROOT/'data/derived/daily_cohort_learning_20261012'
DIM = 17
SETTINGS = dict(cohort='airport and UTC takeoff day', width=DIM,
    original_scalar_columns=prior.VALUES,
    extras=['scheduled_takeoff_gap', 'scheduled_takeoff_gap_present'],
    fixed_scaling='clock asinh(seconds/3600); geometry clip(meters/2000,-10,10); gap asinh(seconds/7200) clip[-10,10]; explicit masks unchanged',
    donor_population='all input departures except explicit VFR; unknown retained',
    aliases=prior.SETTINGS['aliases'], no_labels=True, no_predictions=True,
    allowed_scope='same UTC day only, exact non-contiguous original outer-fold months',
    order='sorted (airport code, UTC day), donor movement ID ascending',
    production='separate old/final input banks required; this bank is 2025 only')
art, verify, write = prior.art, prior.verify, prior.write


def transform(values, gap):
    x = np.asarray(values, np.float32).copy()
    assert x.shape[1] == 15 and np.isfinite(x).all()
    x[:, :4] = np.arcsinh(x[:, :4]/3600)
    x[:, 9:11] = np.clip(x[:, 9:11]/2000, -10, 10)
    present = np.isfinite(gap)
    z = np.clip(np.arcsinh(np.where(present, gap, 0)/7200), -10, 10)
    out = np.column_stack([x, z, present]).astype(np.float32)
    assert out.shape[1] == DIM and np.isfinite(out).all()
    return out


def excluded_positions(qmeta, qclock, dmeta, dclock):
    """Query-to-donor aliases, restricted to already identical airport UTC days."""
    same_id = qmeta[:, 0, None] == dmeta[None, :, 0]
    dt = abs(qmeta[:, 1, None] - dmeta[None, :, 1])
    aircraft = ((qmeta[:, 5, None] >= 0) &
        (qmeta[:, 5, None] == dmeta[None, :, 5]) & (dt <= prior.ALIAS_MS))
    known = (qmeta[:, 3] > 0) & (qmeta[:, 4] > 0) & np.isfinite(qclock)
    signature = (known[:, None] & (dt == 0) &
        (qmeta[:, 3, None] == dmeta[None, :, 3]) &
        (qmeta[:, 4, None] == dmeta[None, :, 4]) &
        (qclock[:, None] == dclock[None, :]))
    return same_id | aircraft | signature


def pack(meta, values, clock, gaps, rules):
    """Pack finite input tuples; no API accepts departure labels."""
    assert len(meta) == len(values) == len(clock) == len(gaps) == len(rules)
    assert np.all(np.diff(meta[:, 0]) > 0), 'Bank IDs must have canonical increasing order'
    scalar = transform(values, gaps)
    day = np.floor_divide(meta[:, 1], 86400000)
    keys, group = np.unique(np.column_stack([meta[:, 2], day]), axis=0, return_inverse=True)
    eligible = np.asarray(pd.Series(rules).ne('V'), bool)
    counts = np.bincount(group[eligible], minlength=len(keys))
    width = max(1, int(counts.max(initial=0)))
    padded = np.zeros((len(keys), width, DIM), np.float32)
    mask = np.zeros((len(keys), width), bool)
    padded_ids = np.full((len(keys), width), -1, np.int64)
    exclusions, offsets = [], np.zeros(len(meta)+1, np.int64)
    # A global sorted ID order makes both group donor ordering and query order deterministic.
    lists = [None]*len(meta)
    for g in range(len(keys)):
        queries = np.flatnonzero(group == g)
        donors = queries[eligible[queries]]
        padded[g, :len(donors)] = scalar[donors]
        mask[g, :len(donors)] = True
        padded_ids[g, :len(donors)] = meta[donors, 0]
        aliases = excluded_positions(meta[queries], clock[queries], meta[donors], clock[donors])
        for row, positions in zip(queries, aliases):
            lists[row] = np.flatnonzero(positions).astype(np.int16)
    max_aliases = max(1, max(len(v) for v in lists))
    excluded = np.full((len(meta), max_aliases), -1, np.int16)
    for row, positions in enumerate(lists):
        excluded[row, :len(positions)] = positions
    return dict(ids=np.asarray(meta[:, 0]), group=group.astype(np.int32), keys=keys,
        values=padded, mask=mask, donor_ids=padded_ids, excluded=excluded)


def prepare():
    assert not (OUT/'manifest.json').exists(), 'Preserve completed bank'
    assert not OUT.exists() or not any(OUT.iterdir()), 'Inspect partial bank'
    old = json.loads((prior.OUT/'train/manifest.json').read_text())
    sources = [Path(__file__), Path(prior.__file__), prior.OUT/'train/manifest.json',
        public.OUT/'inputs2025.parquet']
    arrays = {}
    for name in ['meta', 'values', 'clock']:
        record = next(r for r in old['outputs'] if r['path'].endswith('/'+name+'.npy'))
        path = verify(record); sources.append(path)
        arrays[name] = np.load(path, mmap_mode='r')
    f = pd.read_parquet(public.OUT/'inputs2025.parquet', columns=['mvt_id', 'gap', 'fltrule'])
    assert f.mvt_id.is_unique
    f = f.set_index('mvt_id').loc[arrays['meta'][:, 0]]
    assert len(f) == 2085047
    result = pack(arrays['meta'], arrays['values'], arrays['clock'], f.gap.to_numpy(float), f.fltrule.to_numpy())
    OUT.mkdir(parents=True)
    for name, array in result.items():
        np.save(OUT/(name+'.npy'), array)
    write(OUT/'manifest.json', dict(settings=SETTINGS, rows=len(f), source=art(__file__),
        sources=[art(p) for p in sources], outputs=[art(p) for p in sorted(OUT.glob('*.npy'))],
        dimensions={k:list(v.shape) for k,v in result.items()}, no_labels_loaded=True))
    print(dict(rows=len(f), dimensions={k:list(v.shape) for k,v in result.items()}), flush=True)


def guard(full=True):
    value = json.loads((OUT/'manifest.json').read_text())
    assert value['settings'] == SETTINGS and value['source'] == art(__file__)
    assert value['no_labels_loaded'] and value['rows'] == 2085047
    if full:
        for r in value['sources'] + value['outputs']:
            verify(r)
    return value


class Bank:
    def __init__(self, device, folder=OUT):
        self.ids = np.load(folder/'ids.npy', mmap_mode='r')
        self.group = np.load(folder/'group.npy', mmap_mode='r')
        self.keys = np.load(folder/'keys.npy', mmap_mode='r')
        self.excluded = np.load(folder/'excluded.npy', mmap_mode='r')
        self.values = torch.from_numpy(np.load(folder/'values.npy')).to(device)
        self.mask = torch.from_numpy(np.load(folder/'mask.npy')).to(device)
        self.device = device

    def positions(self, ids):
        ids = np.asarray(ids, np.int64)
        positions = np.searchsorted(self.ids, ids)
        assert np.all(positions < len(self.ids))
        np.testing.assert_array_equal(self.ids[positions], ids)
        return positions

    def packet(self, ids):
        rows = self.positions(ids)
        groups, inverse = np.unique(self.group[rows], return_inverse=True)
        ix = torch.as_tensor(groups, dtype=torch.long, device=self.device)
        inverse = torch.as_tensor(inverse, dtype=torch.long, device=self.device)
        excluded = torch.tensor(self.excluded[rows].astype(np.int64), device=self.device)
        return (self.values[ix], self.mask[ix], inverse, excluded)

    def scope(self, ids, allowed):
        pos = self.positions(ids)
        days = self.keys[self.group[pos], 1]
        months = pd.to_datetime(days*86400000, unit='ms', utc=True).strftime('%Y-%m')
        assert set(months) == set(allowed)
        return dict(rows=len(pos), months=sorted(set(months)), same_UTC_day_donors=True)


class Tracks:
    """Original track tensors plus a deterministic scalar-only donor packet."""
    def __init__(self, original, index, device):
        self.original = original
        self.ids = index.mvt_id.to_numpy(np.int64)
        self.bank = Bank(device)
        self.alignments = []

    def align(self, ids):
        rows = self.original.align(ids)
        np.testing.assert_array_equal(self.ids[rows], ids)
        self.alignments.append(rows)
        return rows

    def batch(self, rows):
        return (*self.original.batch(rows), self.bank.packet(self.ids[np.asarray(rows)]))


def checks():
    times = [0, 1000, 120000, 120001, 0, 0]
    meta = np.column_stack([np.arange(6)+10, times, np.ones(6),
        [1, 2, 3, 4, 1, 0], np.ones(6), [7, 7, 7, 7, 8, -1]]).astype(np.int64)
    clocks = np.array([-500., 12., 22., 23., -500., np.nan])
    aliases = excluded_positions(meta[:1], clocks[:1], meta, clocks)[0]
    np.testing.assert_array_equal(aliases, [True, True, True, False, True, False])
    value = np.zeros((6, 15), np.float32)
    value[:, 4:8] = 1
    gaps = np.array([1, 2, np.nan, 4, 5, 6.])
    rules = np.array(['I', 'I', 'I', 'V', None, 'I'], object)
    packed = pack(meta, value, clocks, gaps, rules)
    assert 13 not in packed['donor_ids']
    np.testing.assert_array_equal(packed['donor_ids'][0], [10, 11, 12, 14, 15])
    np.testing.assert_array_equal(packed['excluded'][0], [0, 1, 2, 3])
    assert np.isfinite(packed['values']).all()
    return dict(query_and_shared_aircraft_aliases=True, exact_signature_alias=True,
        inclusive_120_second_boundary=True, unknown_identity_not_alias=True,
        VFR_excluded_unknown_included=True, fixed_finite_scaling=True)


if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage', choices=['prepare','check'])
    a=ap.parse_args()
    if a.stage=='check': print(checks())
    else: prepare()
