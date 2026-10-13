"""Index audited historical final vectors; never bind an incomplete basis.

Only receipt metadata, hashes and parquet schemas are read. Existing model and
all-row audits remain authoritative; final_public_corrections.bind performs the
one complete population/value check when every component is available.
"""
import hashlib
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'VALIDATION'))
from final_public_corrections import audit_binds

RUNS = ROOT / 'VALIDATION/runs'
OUT = RUNS / 'v3_20261012_final_basis_catalog'


def read(path):
    return json.loads(path.read_text())


def art(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.file_digest(path.open('rb'), 'sha256').hexdigest())


def main():
    components = {}
    required = set(read(RUNS / 'v3_20261012_final_public_corrections/identity.json')['old_basis'])

    def add(version, old, final, audit_path, provenance, result_path=None):
        key = str(version)
        assert key in required and key not in components
        audit = read(audit_path)
        assert audit['status'] == 'pass'
        result_record = art(result_path) if result_path else None
        result = read(result_path) if result_path else None
        assert audit_binds(audit, final, result_record, result), (version, 'unbound audit')
        names = []
        for record in [old, final]:
            path = ROOT / record['path']
            assert art(path)['sha256'] == record['sha256']
            names.append(pq.read_schema(path).names)
        assert all('mvt_id' in n and 'submitted' in n for n in names)
        components[key] = dict(old=old, final=final, audit=art(audit_path), provenance=provenance,
                               id_column='mvt_id', value_column='submitted')
        if result_record:
            components[key]['result'] = result_record

    legacy = RUNS / 'v3_20261012_final_legacy_audit/audit.json'
    audit = read(legacy)
    accepted = {r['version']: r for r in audit['predictions']}
    for family in ['v38', 'overnight', 'modern', 'adsb']:
        folder = RUNS / f'v3_20261012_final_legacy_{family}_replay'
        old, final = read(folder / 'old.json'), read(folder / 'final.json')
        olds = {r['version']: r for r in old['results']} if 'results' in old else {38: old}
        finals = final['results'] if 'results' in final else [dict(final, version=38)]
        for result in finals:
            v = result['version']
            assert olds[v]['all_original_predictions_exact']
            assert accepted[v]['final_prediction'] == result['prediction']
            add(v, olds[v]['prediction'], result['prediction'], legacy,
                'Archived port with exact old integers. Original15-member base-network descendants retain October8 refit provenance; no original-weight byte identity claim.')
    means = RUNS / 'v3_20261012_final_legacy_airport_mean_replay/result.json'
    result = read(means)
    assert result['views']['old']['old_integer_reference_exact']
    add(1, result['views']['old']['prediction'], result['views']['final']['prediction'], means,
        'Original airport means independently reconstructed from exact integer training sums; no model fit.')
    families = [RUNS / 'v3_20261012_final_missing_basic_reconstruct' / name for name in ['baseline', 'recovery']]
    families.append(RUNS / 'v3_20261012_final_missing_stand_reconstruct')
    for folder in families:
        result = read(folder / 'result.json')
        for version, old in result['views']['old'].items():
            assert old['exact_original_integer_replay']
            add(version, old['prediction'], result['views']['final'][version]['prediction'],
                folder / 'independent_audit.json',
                'Fixed archived-recipe reconstruction. All original integers exact; model text identity and historical thread caveats remain in producer/audit receipts.',
                folder / 'result.json')
    folder = RUNS / 'v3_20261012_final_missing_flightnumber_reconstruct'
    result = read(folder / 'result.json')
    assert result['views']['old']['exact_original_integer_replay']
    add(56, result['views']['old']['prediction'], result['views']['final']['prediction'],
        folder / 'independent_audit.json',
        'Original fixed flight-number classifier recipe reconstructed; old integers exact, original model hash absent. Unseen-number fallback retained.', folder / 'result.json')
    folder = RUNS / 'v3_20261012_final_weather_inference'
    for version in [60, 61]:
        old, final = [read(folder / f'{view}_v{version}.json') for view in ['old', 'final']]
        assert old['all_original_predictions_exact']
        add(version, old['prediction'], final['prediction'], folder / 'independent_audit.json',
            'Saved weather weights, exact original component floats and integers; v60 retains audited October8 base-network refit provenance.')
    for version in [4, 5, 6, 7, 8, 9]:
        folder = RUNS / 'v3_20261012_final_missing_early_reconstruct' / f'v{version}'
        if not (folder / 'independent_audit.json').exists():
            continue
        result = read(folder / 'result.json')
        assert result['views']['old']['exact_original_integer_replay']
        add(version, result['views']['old']['prediction'], result['views']['final']['prediction'],
            folder / 'independent_audit.json',
            f"Fixed early LightGBM reconstruction. Original source exact={result['historical_model_source_exact']}; all old integers exact. Original weight hash absent; source/runtime caveats retained.",
            folder / 'result.json')
    # This is explicitly partial while recovery remains unfinished. Missing
    # versions cause the downstream binder to reject it without substitution.
    value = dict(schema='final_public_corrections_basis_v1', source=art(Path(__file__)),
        components=dict(sorted(components.items(), key=lambda item: int(item[0]))),
        missing=sorted(map(int, required-set(components))), complete=set(components) == required,
        scope='Metadata/hash/schema catalog only; prior all-row audits retained; no inference or new replay',
        no_binding=True, no_substitution=True, no_upload=True)
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f'partial_{len(components)}.json'
    if target.exists():
        assert read(target) == value, 'Preserve different prior catalog'
    else:
        target.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(dict(ready=len(components), missing=value['missing'], complete=value['complete'],
                         artifact=art(target))))


if __name__ == '__main__':
    main()
