"""Verify the bounded laptop return packet and restore original point bytes.

Run only in an allocated desktop CPU slot. Existing different committed arrays
are never overwritten. This restores audited bytes and performs no modelling.
"""
from pathlib import Path
import gzip
import hashlib
import json
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/derived/sparse_track_submit_20261010'


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def run():
    started = time.monotonic()
    packet_path = OUT / 'return_packet.json'
    packet = json.loads(packet_path.read_text())
    for item in packet['files']:
        path = ROOT / item['path']
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('Returned packet file mismatch: '+str(path))
    audit_path = ROOT / packet['audit']['path']
    if sha(audit_path) != packet['audit']['sha256']:
        raise ValueError('Returned independent audit changed')
    audit = json.loads(audit_path.read_text())
    if not audit['passed']:
        raise ValueError('Returned cache has not passed its independent audit')
    if ROOT / audit['prepared_manifest']['path'] != OUT / 'manifest.json':
        raise ValueError('Independent audit points to another prepared cache')
    for field in ['auditor', 'prepared_manifest']:
        item = audit[field]
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Independent audit source/preparation changed: '+field)
    result = {}
    for view in ['old', 'final']:
        target = OUT / view
        manifest_path = target / 'full.json'
        manifest = json.loads(manifest_path.read_text())
        description = packet['views'][view]
        expected = manifest['outputs']['full_points.bin']
        if (sha(target / 'full_index.parquet') != manifest['outputs']['full_index.parquet']
                or sha(target / 'full_design.json') != manifest['design_sha256']
                or manifest['settings']['ranking_view'] != view):
            raise ValueError('Returned index/design namespace changed')
        if (description['original_sha256'] != expected
                or description['original_bytes'] != manifest['raw_array_bytes']
                or audit['views'][view]['manifest']['sha256'] != sha(manifest_path)):
            raise ValueError('Original point identity differs between packet/manifest/audit')
        dest = target / 'full_points.bin'
        reused = dest.exists()
        if reused:
            if dest.stat().st_size != description['original_bytes'] or sha(dest) != expected:
                raise ValueError('Refusing to replace a different committed point array')
        else:
            compressed = ROOT / description['compressed']['path']
            if (sha(compressed) != description['compressed']['sha256']
                    or compressed.stat().st_size != description['compressed']['bytes']):
                raise ValueError('Compressed point source changed')
            temporary = dest.with_suffix('.bin.tmp')
            digest = hashlib.sha256()
            with gzip.open(compressed, 'rb') as source, temporary.open('wb') as output:
                for block in iter(lambda: source.read(8*1024*1024), b''):
                    digest.update(block)
                    output.write(block)
            if digest.hexdigest() != expected or temporary.stat().st_size != description['original_bytes']:
                raise ValueError('Restored point bytes do not match audited original')
            temporary.replace(dest)
        result[view] = dict(sha256=expected, bytes=dest.stat().st_size,
                            rows=manifest['rows'], stored_points=manifest['stored_points'], reused=reused)
    report = dict(passed=True, source_sha256=sha(Path(__file__)),
                  packet_sha256=sha(packet_path), views=result,
                  packet_files=len(packet['files']), seconds=time.monotonic()-started)
    (OUT / 'transport_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    run()
