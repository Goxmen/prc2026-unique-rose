"""Additive audit-only repair for raw-file versus actual takeoff month.

The original source/identity/preparation/preflight remain unchanged. Exactly one
overstrong audit assertion is replaced; every selected-row check remains intact.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import types

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'VALIDATION/runs/v3_20261013_train_label_retrieval'
FOLDER = OUT/'prefit_population_recovery'
ORIGINAL = ROOT/'VALIDATION/tools/audit_retrieval_frontier_prefit_20261013.py'
ORIGINAL_SHA = '319cbc8e739b4bd85bf2f0e7a21b0e35882e42e4f1a9bb003a6cecce4de4e0a0'
DIAGNOSIS = OUT/'prefit_membership_diagnosis/result.json'
OLD = '        eq(np.sort(raw.MVT_ID_mvt.to_numpy()), meta[months == month, 0])'
NEW = '        verify_file_population(month, raw, meta, months)'
ALLOWED = [f'2025-{m:02d}' for m in [2, 3, 5, 6, 8, 9, 10, 11, 12]]+['2025-04']


def read(p): return json.loads(Path(p).read_text())


def art(p):
    p = Path(p).resolve(); h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): h.update(block)
    return dict(path=str(p.relative_to(ROOT)), sha256=h.hexdigest(), bytes=p.stat().st_size)


def verify(r):
    p = (ROOT/r['path']).resolve(); p.relative_to(ROOT); assert art(p) == r, p
    return p


def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); assert not p.exists(), p
    with p.open('x') as stream: json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def patched_source():
    assert art(ORIGINAL)['sha256'] == ORIGINAL_SHA
    text = ORIGINAL.read_text(); assert text.count(OLD) == 1
    patched = text.replace(OLD, NEW)
    assert patched.count(NEW) == 1
    # A reverse replacement must recover every AST node of the frozen source.
    assert ast.dump(ast.parse(patched.replace(NEW, OLD)), include_attributes=False) == ast.dump(ast.parse(text), include_attributes=False)
    return patched


def freeze():
    assert not (OUT/'prefit_audit.json').exists() and not (OUT/'fit.json').exists()
    assert not (FOLDER/'recovery_identity.json').exists()
    review = read(FOLDER/'source_review.json')
    assert review['status'] == 'pass' and review['source'] == art(__file__)
    assert review['original'] == art(ORIGINAL) and review['diagnosis'] == art(DIAGNOSIS)
    text = patched_source(); archive = FOLDER/'archive'; archive.mkdir()
    records = []
    for p in [ORIGINAL, OUT/'prefit_audit.log', OUT/'identity.json', OUT/'prepared.json',
              OUT/'preflight.json', OUT/'source_review.json', OUT/'allocation.json', DIAGNOSIS]:
        # Unique names keep producer/preflight metadata distinct from repair metadata.
        name = p.name if p != DIAGNOSIS else 'membership_diagnosis.json'
        target = archive/name; assert not target.exists(); shutil.copyfile(p, target)
        assert art(target)['sha256'] == art(p)['sha256']
        records.append(dict(original=art(p), archived=art(target)))
    snap = FOLDER/'executed_auditor_snapshot.py'; assert not snap.exists(); snap.write_text(text)
    write(FOLDER/'recovery_identity.json', dict(source=art(__file__), review=art(FOLDER/'source_review.json'),
        original=art(ORIGINAL), original_identity=art(OUT/'identity.json'), prepared=art(OUT/'prepared.json'),
        preflight=art(OUT/'preflight.json'), allocation=art(OUT/'allocation.json'), diagnosis=art(DIAGNOSIS),
        snapshot=art(snap), archive=records,
        change='One per-file/month population assertion replaced with stricter allowed-union/raw-calendar and exact-April proof; no producer/numeric/source changes'))
    print(json.dumps(dict(status='frozen', recovery=art(FOLDER/'recovery_identity.json'))))


class PopulationProof:
    def __init__(self):
        self.seen = {}; self.expected = None; self.expected_meta = None; self.crossings = {}

    def observe(self, month, raw, meta, months):
        assert month in ALLOWED and month not in self.seen
        ids = raw.MVT_ID_mvt.to_numpy(); integer_ids = ids.astype(np.int64)
        np.testing.assert_array_equal(ids, integer_ids)
        assert len(np.unique(ids)) == len(ids)
        index = np.searchsorted(meta[:, 0], integer_ids)
        assert (index < len(meta)).all(); np.testing.assert_array_equal(meta[index, 0], integer_ids)
        t = pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True)
        actual_month = t.dt.strftime('%Y-%m').to_numpy()
        assert np.isin(actual_month, ALLOWED).all()
        stamp = t.astype('datetime64[ms, UTC]').astype('int64').to_numpy()
        np.testing.assert_array_equal(meta[index, 1], stamp)
        np.testing.assert_array_equal(months[index], actual_month)
        if month == '2025-04':
            np.testing.assert_array_equal(np.sort(integer_ids), meta[months == month, 0])
            assert (actual_month == month).all()
        for j in np.flatnonzero(actual_month != month):
            self.crossings[str(int(integer_ids[j]))] = dict(file_month=month, takeoff=str(t.iloc[j]))
        self.seen[month] = integer_ids
        if self.expected is None:
            self.expected = meta[np.isin(months, ALLOWED), 0]
            self.expected_meta = meta

    def finish(self):
        assert set(self.seen) == set(ALLOWED)
        union = np.concatenate([self.seen[m] for m in ALLOWED])
        assert len(np.unique(union)) == len(union)
        np.testing.assert_array_equal(np.sort(union), self.expected)
        diagnosis = read(DIAGNOSIS)
        assert self.crossings == diagnosis['anomalous_raw_locations']
        assert sorted(map(int, self.crossings)) == diagnosis['all_mismatch_ids'] == [185722540, 196701297]
        selection = dict(np.load(OUT/'inputs/selection.npz'))
        for name in ['train', 'query']:
            selected_ids = self.expected_meta[selection[name], 0]
            assert not np.isin(selected_ids, list(map(int, self.crossings))).any()
        return dict(allowed_union_rows=len(union), exact_April_rows=len(self.seen['2025-04']),
                    all_raw_input_timestamps_match_bank=True, crossings=self.crossings,
                    crossing_IDs_absent_selected_train_and_query=True,
                    no_January_July_file_or_April_target_read=True)


def audit():
    frozen = read(FOLDER/'recovery_identity.json')
    assert frozen['source'] == art(__file__) and frozen['original'] == art(ORIGINAL)
    for key in ['review', 'original', 'original_identity', 'prepared', 'preflight', 'allocation', 'diagnosis', 'snapshot']:
        verify(frozen[key])
    for record in frozen['archive']:
        verify(record['original']); verify(record['archived'])
    text = patched_source(); assert text == verify(frozen['snapshot']).read_text()
    assert not (OUT/'prefit_audit.json').exists() and not (FOLDER/'recovery_result.json').exists()
    attempt = FOLDER/'audit_attempt.json'
    write(attempt, dict(source=art(__file__), recovery=art(FOLDER/'recovery_identity.json')))
    module = types.ModuleType('frozen_retrieval_prefit_audit')
    module.__file__ = str(ORIGINAL)
    exec(compile(text, str(FOLDER/'executed_auditor_snapshot.py'), 'exec'), module.__dict__)
    proof = PopulationProof(); module.verify_file_population = proof.observe
    original_save = module.save
    def audited_save(path, value):
        assert Path(path) == OUT/'prefit_audit.json'
        value['population_scope_recovery'] = proof.finish()
        value['audit_recovery'] = art(FOLDER/'recovery_identity.json')
        value['executed_auditor'] = art(__file__)
        value['executed_snapshot'] = frozen['snapshot']
        original_save(path, value)
    module.save = audited_save
    module.prefit()
    write(FOLDER/'recovery_result.json', dict(status='pass', recovery=art(FOLDER/'recovery_identity.json'),
        source=art(__file__), audit=art(OUT/'prefit_audit.json'), original_identity=art(OUT/'identity.json'),
        unchanged_prepared=art(OUT/'prepared.json'), unchanged_preflight=art(OUT/'preflight.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['freeze', 'audit'])
    args = parser.parse_args(); torch.set_num_threads(1)
    {'freeze': freeze, 'audit': audit}[args.stage]()
