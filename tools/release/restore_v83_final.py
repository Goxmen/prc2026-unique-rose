#!/usr/bin/env python3
"""Restore the owner's previously accepted v83 final, without any model execution.

Reuse the reviewed conditional, single-PUT delivery transport. Only the fixed
expected predecessor and the candidate validator differ. A recorded attempt is
reconciled, never automatically uploaded again. This does not publish GitHub.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/v80_refit'))
spec = importlib.util.spec_from_file_location('v83_restore_transport', ROOT / 'tools/v80_refit/upload.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)

OUT = ROOT / 'VALIDATION/runs/v3_20261014_restore_v83_final'
ORIGINAL = ROOT / 'VALIDATION/runs/v3_20261013_v80_teacher_distillation'
V83_SHA = 'f4f6a15fe5bb0865727cb906b77e17f182d34dc4b516f657c26937ad097af85f'
REFIT_SHA = '3d996ed6c12f32500c46738c1b89133282239903f94d82329c83dc97a7f4ae20'


def validate(out):
    import pyarrow.parquet as pq

    if Path(out).resolve() != OUT:
        raise ValueError('Restoration receipts must use their separate run directory')
    candidate = ORIGINAL / 'production/final/candidate.parquet'
    backup = ROOT / 'VALIDATION/runs/v3_20261013_v80_refit_final/delivery/previous_remote_final.parquet'
    if transport.sha(candidate) != V83_SHA or transport.sha(backup) != V83_SHA:
        raise ValueError('Original v83 and its prior remote backup must match the accepted hash')
    acceptance = transport.read(ORIGINAL / 'final_submission/acceptance.json')
    official_path = transport.verify(acceptance['official_result'])
    official = transport.read(official_path)
    if (acceptance['status'] != 'accepted_final_score_unpublished'
            or official.get('status') != 'Succeeded' or official.get('used_pairs') != 670790
            or official.get('file') != transport.NAME
            or official.get('bucket') != 'prc-2026-unique-rose'
            or official.get('inputs', {}).get('test_set_file') != 'final_submitting.parquet'):
        raise ValueError('Historical acceptance does not confirm the original final route/population')
    template = ROOT / 'data/final_submitting.parquet'
    if transport.sha(template) != transport.TEMPLATE_SHA:
        raise ValueError('Final template changed')
    checks = transport.check_tables(pq.read_table(template), pq.read_table(candidate), pq.read_table(backup))
    return dict(candidate=transport.artifact(candidate), checks=checks,
                historical_acceptance=transport.artifact(ORIGINAL / 'final_submission/acceptance.json'),
                recipe='Previously accepted v83, exact original bytes; no retraining',
                selected_by='Owner explicitly requested returning to v83 on 14 October 2026',
                expected_predecessor_sha256=REFIT_SHA,
                restoration_source=transport.artifact(Path(__file__)),
                transport_source=transport.artifact(ROOT / 'tools/v80_refit/upload.py'))


def deliver(wait_seconds=45):
    old_validator, old_prior = transport.validate, transport.PRIOR_SHA
    transport.validate, transport.PRIOR_SHA = validate, REFIT_SHA
    try:
        return transport.deliver(OUT, wait_seconds=wait_seconds)
    finally:
        transport.validate, transport.PRIOR_SHA = old_validator, old_prior


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'deliver'])
    args = parser.parse_args()
    result = validate(OUT) if args.command == 'check' else deliver()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
