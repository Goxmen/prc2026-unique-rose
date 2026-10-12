"""Register completed, audited legacy ports; no model operations or uploads."""
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'VALIDATION/runs'


def read(p):
    return json.loads(p.read_text())


def write(p, value):
    temporary = p.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(p)


def main():
    receipt = RUNS / 'v3_20261012_final_legacy_audit/audit.json'
    audit = read(receipt)
    assert audit['status'] == 'pass'
    versions = sorted(a['version'] for a in audit['predictions'])
    assert versions == [17, 18, 19, 20, 21, 38, 45, 46, 54, 57, 58, 64, 66, 68, 70]
    stamp = datetime.now(timezone.utc).isoformat()
    path = ROOT / 'VALIDATION/reports/final_route_inventory_20261012.json'
    value = read(path)
    value['at'] = stamp
    value['completed']['legacy_adsb'] = dict(versions=[45, 46], old_integer_vectors_exact=True,
        final_model_reloads_exact=True, no_refit=True,
        receipt='VALIDATION/runs/v3_20261012_final_legacy_adsb_replay/final.json')
    value['completed']['independent_legacy_audit'] = dict(receipt=str(receipt.relative_to(ROOT)),
        versions=versions, excluded=[12, 13, 15], old_and_final_values_checked=15 * (344841 + 670790))
    value['in_progress'] = ['Fixed deeper-TabM CV screen and automatic scoring',
        'Missing historical affine-basis recovery; backup availability requested']
    write(path, value)
    path = ROOT / 'VALIDATION/research_registry.json'
    value = read(path)
    value['updated_at'] = stamp
    value['current']['desktop'] = 'All15 retained legacy final components audited; missing original affine basis still pending'
    value['next'] = ['Review frozen deeper-TabM comparison after automatic completion',
        'Recover missing historical model backups before claiming an exact final ensemble',
        'Continue distinct supported research; all October12 submission slots used']
    write(path, value)
    path = ROOT / 'LOCAL_PROGRESS.md'
    text = path.read_text().replace('Thirteen archived final-set components recovered',
                                    'Fifteen archived final-set components recovered and audited')
    path.write_text(text)
    path = ROOT / 'VALIDATION/reports/TARGET205_RESEARCH.md'
    title = '## Fifteen archived final components audited'
    if title not in path.read_text():
        with path.open('a') as f:
            f.write('\n\n' + title + '\n\n')
            f.write(f'Completed at {stamp}. Retained versions: ' + ', '.join('v' + str(v) for v in versions) + '.\n')
            f.write('Independent SQL checks cover15,234,465 old/final values, original-file equality, ID membership, integer serialization and four-month populations. '
                    'V12/v13/v15 control substitutions remain excluded. The15 archived October8 base-network arrays also reconstruct their final mean exactly; '
                    'their recorded old-input RMS difference of2.39918e-6seconds is independently reproduced and explicitly retained as refit provenance. '
                    'No new fit, complete affine port, final candidate or upload follows. Receipt: '
                    '`../runs/v3_20261012_final_legacy_audit/audit.json`.\n')
    print(json.dumps(dict(status='registered', versions=versions, final_affine_port_complete=False)))


if __name__ == '__main__':
    main()
