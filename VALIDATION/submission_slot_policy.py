"""Enforce a user-reserved final daily slot before any external upload."""
import json
from pathlib import Path

from records import sha256

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / 'VALIDATION/submission_slot_policy.json'


def check_slot(manifest, used, today):
    if not POLICY.exists():
        return None
    policy = json.loads(POLICY.read_text())
    if today.isoformat() != policy['utc_date']:
        return None
    receipt = {'policy_sha256': sha256(POLICY), 'reserved_slots': policy['reserved_slots']}
    if len(used) < policy['daily_limit'] - policy['reserved_slots']:
        return receipt
    selected = manifest.get('reserved_slot_selection')
    assert selected, 'Last daily slot is reserved for the strongest validated candidate'
    path = (ROOT / selected['path']).resolve()
    assert path.is_relative_to(ROOT / 'VALIDATION/runs')
    assert sha256(path) == selected['sha256'], 'Final-selection receipt changed'
    decision = json.loads(path.read_text())
    assert decision['status'] == 'selected_best_available'
    assert decision['utc_date'] == today.isoformat()
    assert decision['file'] == manifest['file'] and decision['sha256'] == manifest['sha256']
    assert decision['compared_candidates'] and decision['selection_reason']
    assert decision['remaining_promising_experiments_resolved'] is True
    for item in decision['evidence']:
        assert sha256(ROOT / item['path']) == item['sha256'], item['path']
    assert decision['evidence'], 'Final selection needs validation evidence'
    receipt['selection'] = selected
    return receipt
