#!/usr/bin/env python3
"""Deliver the explicitly selected v84 final through the reviewed single-PUT transport.

This is a separately scored reconstruction, not a claim to original-v80 weight
identity. The historical approximate-v80 uploader remains disabled. This route
requires independent final checks and exact retained-component replay, the scope
explicitly accepted by the owner. Current owner authorization separately blocks
any final replacement until the owner selects v84 again.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'VALIDATION/runs/v3_20261014_v84_final'
OLD = ROOT / 'VALIDATION/runs/v3_20261014_exact_v80_recovery/prediction_fidelity_recheck'
OLD_SHA = 'ab5749c68560aa02bae68aadf9e13b962a87f0c5227b2f2303303a41cff16763'
PRIOR_SHA = 'f4f6a15fe5bb0865727cb906b77e17f182d34dc4b516f657c26937ad097af85f'
sys.path.insert(0, str(ROOT / 'tools/v80_refit'))
spec = importlib.util.spec_from_file_location('v84_final_transport', ROOT / 'tools/v80_refit/upload.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def selection_checks(owner, official, independent, replay, candidate):
    require(owner.get('selected_version') == 84 and owner.get('allowed_final_upload') is True,
            'Owner selection must authorize v84 final delivery')
    require(owner.get('selected_old_candidate_sha256') == OLD_SHA,
            'Owner selection does not identify the scored v84')
    require(official.get('status') == 'Succeeded' and official.get('score') == 218.9025
            and official.get('used_pairs') == 344841
            and official.get('file') == 'unique-rose_v84.parquet',
            'Official result does not confirm the selected regular v84')
    require(independent.get('status') == 'pass' and independent.get('candidate') == candidate,
            'Independent audit must bind this exact final candidate')
    require(independent.get('old_leaderboard_candidate', {}).get('sha256') == OLD_SHA,
            'Final audit is not linked to the selected v84 recipe')
    require(replay.get('status') == 'pass' and replay.get('sha256') == candidate['sha256']
            and replay.get('rows') == 670790,
            'Saved-artifact replay must reproduce the exact final bytes')


def validate(out):
    import pyarrow.parquet as pq

    require(Path(out).resolve() == OUT, 'Use the separate v84 final run directory')
    current = transport.read(OUT / 'current_owner_authorization.json')
    require(current.get('allowed_final_upload') is True
            and current.get('selected_final_version') == 84,
            'Owner currently preserves v83; a new explicit selection is required before v84 delivery')
    require(current.get('accepted_reproduction_scope') == 'retained_component_assembly',
            'Current owner authorization must acknowledge the actual replay scope')
    owner = transport.read(OUT / 'owner_selection.json')
    ready = transport.read(OUT / 'ready_for_delivery.json')
    require(ready.get('status') == 'ready', 'Final readiness is not established')
    paths = {name: transport.verify(ready[name]) for name in
             ['candidate', 'independent_audit', 'replay_validation']}
    require(paths['candidate'] == OUT / 'final/candidate.parquet', 'Unexpected final candidate path')
    candidate = transport.artifact(paths['candidate'])
    require(transport.sha(OLD / 'old/candidate.parquet') == OLD_SHA, 'Selected old v84 changed')
    official = transport.read(OLD / 'leaderboard_delivery/official_result.json')
    independent = transport.read(paths['independent_audit'])
    replay = transport.read(paths['replay_validation'])
    selection_checks(owner, official, independent, replay, candidate)
    for name in ['assembly_identity', 'producer_audit', 'old_leaderboard_candidate']:
        transport.verify(independent[name])
    template = ROOT / 'data/final_submitting.parquet'
    require(transport.sha(template) == transport.TEMPLATE_SHA, 'Final template changed')
    actual = pq.read_table(paths['candidate'])
    checks = transport.check_tables(pq.read_table(template), actual, actual)
    require(paths['candidate'].stat().st_size < 50_000_000, 'Unexpected final artifact size')
    return dict(candidate=candidate, checks=checks, selected_version=84,
                recipe='Owner-selected v84 reconstruction; fixed original ensemble coefficients',
                selected_by=owner['owner_instruction'],
                original_v80_weight_identity_claimed=False,
                reproduction_scope='Exact retained-component assembly replay',
                complete_underlying_model_weight_recovery_claimed=False,
                regular_leaderboard_score=218.9025,
                regular_leaderboard_rows=344841,
                final_score_inferred=False,
                expected_predecessor_sha256=PRIOR_SHA,
                owner_selection=transport.artifact(OUT / 'owner_selection.json'),
                current_owner_authorization=transport.artifact(OUT / 'current_owner_authorization.json'),
                readiness=transport.artifact(OUT / 'ready_for_delivery.json'),
                source=transport.artifact(Path(__file__)),
                transport_source=transport.artifact(ROOT / 'tools/v80_refit/upload.py'))


def deliver(wait_seconds=45):
    original_validator, original_prior = transport.validate, transport.PRIOR_SHA
    transport.validate, transport.PRIOR_SHA = validate, PRIOR_SHA
    try:
        return transport.deliver(OUT, wait_seconds=wait_seconds)
    finally:
        transport.validate, transport.PRIOR_SHA = original_validator, original_prior


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'deliver'])
    args = parser.parse_args()
    result = validate(OUT) if args.command == 'check' else deliver()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
