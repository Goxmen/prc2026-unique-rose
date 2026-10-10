"""Bind the completed production reload audit to a prepared physical candidate."""
import argparse
import json
from pathlib import Path
import nn_track_physical_submit as production
from records import sha256, write_json


def main(manifest_path):
    path = Path(manifest_path).resolve()
    root = production.ROOT
    assert path.is_relative_to(root / 'VALIDATION/runs')
    manifest = json.loads(path.read_text())
    assert manifest['state'] == 'prepared' and 'production_audit' not in manifest
    assert manifest['change'] == 'physical-scale sparse track encoder'
    members = manifest['members']
    assert members in ([0], [0, 1, 2])
    audit_path = production.OUT / ('audit_' + '_'.join(map(str, members)) + '.json')
    audit = json.loads(audit_path.read_text())
    assert audit['status'] == 'pass' and audit['members'] == members
    assert audit['manifest'] == production.art(production.MANIFEST)
    production.verify(audit['source'])
    expected = {(k, view) for k in members for view in ['old', 'final']}
    assert {(c['member'], c['view']) for c in audit['checks']} == expected
    assert len(audit['checks']) == len(expected)
    airports = {'EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LEMD', 'LFPG', 'LIRF', 'LSZH', 'LTFM'}
    for check in audit['checks']:
        assert check['record'] == production.art(production.paths(check['member'])[2])
        assert set(check['airports']) == airports and check['rows'] >= 512
        assert 0 <= check['max_difference_sec'] < .01
    for name, digest in manifest['sources'].items():
        assert sha256(root / name) == digest, name
    assert sha256(root / manifest['file']) == manifest['sha256']
    manifest['production_audit'] = production.art(audit_path)
    for source in [audit_path, production.verify(audit['source']), Path(__file__)]:
        manifest['sources'][str(source.relative_to(root))] = sha256(source)
    write_json(path, manifest)
    print(f'Bound {len(expected)} completed reload checks to v{manifest["version"]}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest')
    main(parser.parse_args().manifest)
