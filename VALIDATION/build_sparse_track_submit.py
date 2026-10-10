"""Prepare and stream separate old/final production sparse-track caches.

Only allowlisted departure inputs and existing audited identities are joined.
The frozen full-year sequence builder owns all point selection and streaming.
Portable builds consume compact query packets plus the recorded raw-day union;
raw ranking files, identity tables and the 2025 point cache stay on the desktop.
No target, fitting, candidate selection or upload is performed here.
"""
import argparse
from datetime import date, timedelta
import gc
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa

import build_sparse_track_full as full

seq = full.seq
ROOT = full.ROOT
OUT = ROOT / 'data/derived/sparse_track_submit_20261010'
IDENTITIES = ROOT / 'data/derived/callsign_submit_20261010'
TRAINING_MANIFEST = seq.OUT / 'full.json'
RANKING = {'old': ROOT / 'data/ranking.parquet',
           'final': ROOT / 'data/final_ranking.parquet'}
MONTHS = {'old': ['2026-01', '2026-07'],
          'final': ['2026-01', '2026-02', '2026-06', '2026-07']}
ROWS = {'old': 344841, 'final': 670790}
QUERY_COLUMNS = ['mvt_id', 'airport', 'stand', 'runway', 'to_ms', 'clock_ms',
                 'hex', 'day', 'stand_lat', 'stand_lon', 'runway_lat',
                 'runway_lon', 'origin_lat', 'origin_lon']
SETTINGS = dict(original_sequence=seq.SETTINGS,
    views=MONTHS, expected_rows=ROWS, labels_used=False,
    clock='genuine raw AOBT_3_flt only; missing stays missing',
    identity='existing per-view audited identity; exact ID/phase/airport/takeoff join',
    geometry='unchanged frozen training maps and add_geometry',
    separate_views='same movement ID never reuses points across old/final clocks',
    representation='frozen build_sparse_track_full build_chunk and assemble',
    portable='prejoined query parquet; source hashes retained; absent raw ranking/identity not re-read',
    boundary_days='verify recorded presence and absence before and after each build')
sha = full.sha
read = full.read
write_json = full.write_json
record = full.path_record


def relative(path):
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_absolute() else str(path)


def verify(item):
    path = ROOT / item['path']
    if not path.is_file() or sha(path) != item['sha256']:
        raise ValueError(f'Changed or absent source: {item["path"]}')
    if 'bytes' in item and path.stat().st_size != item['bytes']:
        raise ValueError(f'Changed source size: {item["path"]}')
    return path


def helpers():
    return {relative(Path(m.__file__)): sha(Path(m.__file__))
            for m in [full, seq, seq.rotation, seq.sf, seq.sg]}


def training_contract():
    """Small manifests/maps only; never scan or copy the 2GB training points."""
    training = read(TRAINING_MANIFEST)
    if training['source_sha256'] != sha(Path(full.__file__)):
        raise ValueError('Frozen training builder changed')
    expected = {relative(Path(m.__file__)): sha(Path(m.__file__))
                for m in [seq, seq.rotation, seq.sf, seq.sg]}
    if training['helpers'] != expected:
        raise ValueError('Frozen training sequence/geometry helpers changed')
    if training['settings']['original_sequence'] != seq.SETTINGS:
        raise ValueError('Production settings differ from frozen training representation')
    if (training['stand_map_sha256'] != sha(seq.sf.COORDS)
            or training['runway_map_sha256'] != sha(seq.sf.ENDS)):
        raise ValueError('Static maps differ from frozen training maps')
    if sha(seq.OUT / 'full_design.json') != training['design_sha256']:
        raise ValueError('Frozen training design manifest changed')
    return dict(manifest=record(TRAINING_MANIFEST),
                design=record(seq.OUT / 'full_design.json'),
                outputs=training['outputs'], rows=training['rows'],
                identity_sha256=training['identity_sha256'])


def join_identity(query, identities):
    """No label fields are accepted; compare event keys before retaining hex."""
    if query.mvt_id.duplicated().any() or identities.mvt_id.duplicated().any():
        raise ValueError('Duplicate movement identity or query')
    dep = identities.loc[identities.phase.eq('DEP'),
                         ['mvt_id', 'airport', 'to_ms', 'hex']]
    joined = query.merge(dep, on='mvt_id', how='outer', validate='one_to_one',
                         suffixes=('', '_identity'), indicator=True)
    if not joined['_merge'].eq('both').all():
        raise ValueError('Departure population differs from audited identity population')
    if (not joined.airport.eq(joined.airport_identity).all()
            or not joined.to_ms.eq(joined.to_ms_identity).all()):
        raise ValueError('Wrong ranking namespace or changed event time/airport')
    return joined.drop(columns=['airport_identity', 'to_ms_identity', '_merge'])


def validate_query(query, view):
    if list(query.columns) != QUERY_COLUMNS:
        raise ValueError('Query packet columns changed or contain unapproved fields')
    if len(query) != ROWS[view] or query.mvt_id.isna().any() or query.mvt_id.duplicated().any():
        raise ValueError('Query packet departure population changed')
    if not query.mvt_id.is_monotonic_increasing:
        raise ValueError('Query packet is not deterministically ID-sorted')
    days = pd.to_datetime(query.to_ms, unit='ms', utc=True).dt.strftime('%Y-%m-%d')
    if not query.day.eq(days).all() or sorted(set(days.str[:7])) != MONTHS[view]:
        raise ValueError('Query clock/month namespace mismatch')
    if query[['airport', 'to_ms', 'origin_lat', 'origin_lon']].isna().any().any():
        raise ValueError('Required query metadata absent')
    for col in ['to_ms', 'clock_ms', 'hex']:
        values = query[col].dropna().to_numpy(np.float64)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f'Invalid integer query field {col}')
    if not query.hex.dropna().between(0, 2**24-1).all():
        raise ValueError('Aircraft identity outside 24-bit range')


def prepare_query(con, view):
    ip = IDENTITIES / view / 'identity.parquet'
    im_path = ip.with_suffix('.json')
    im = read(im_path)
    design = record(IDENTITIES / 'design.json')
    if (im['ranking_view'] != view or im['ranking_months'] != MONTHS[view]
            or im['departure_rows'] != ROWS[view]
            or im['design_sha256'] != design['sha256']):
        raise ValueError('Audited production identity view/design changed')
    if relative(im['artifact']['path']) != relative(ip):
        raise ValueError('Identity artifact points at another namespace')
    verify(im['artifact'])
    ranking = record(RANKING[view])
    source_hashes = {relative(r['path']): r['sha256'] for r in im['sources']}
    if source_hashes.get(ranking['path']) != ranking['sha256']:
        raise ValueError('Raw ranking differs from existing audited identities')
    q = con.execute('''SELECT MVT_ID_mvt::BIGINT mvt_id, ADEP_mvt airport,
        STAND_mvt stand, RUNWAY_mvt runway,
        epoch_ms(MVT_TIME_UTC_mvt)::BIGINT to_ms,
        epoch_ms(AOBT_3_flt)::BIGINT clock_ms
        FROM read_parquet(?) WHERE PHASE_mvt='DEP' ''', [str(RANKING[view])]).df()
    identities = con.execute('''SELECT mvt_id,phase,airport,
        epoch_ms(t)::BIGINT to_ms,hex FROM read_parquet(?)''', [str(ip)]).df()
    q = join_identity(q, identities)
    q['day'] = pd.to_datetime(q.to_ms, unit='ms', utc=True).dt.strftime('%Y-%m-%d')
    q = seq.add_geometry(q).sort_values('mvt_id').reset_index(drop=True)[QUERY_COLUMNS]
    validate_query(q, view)
    return q, dict(raw_ranking=ranking, identity=record(ip),
                   identity_manifest=record(im_path), identity_design=design,
                   identity_provenance=im)


def required_days(days):
    return sorted({str(date.fromisoformat(d)+timedelta(days=k))
                   for d in days for k in [-1, 0, 1]})


def position_inventory(days):
    paths = [seq.HEATMAP / 'positions' / f'{d}.parquet' for d in required_days(days)]
    return ([record(p) for p in paths if p.exists()],
            [p.stem for p in paths if not p.exists()])


def overlap(old, final):
    both = old[['mvt_id', 'airport', 'to_ms', 'clock_ms', 'hex']].merge(
        final[['mvt_id', 'airport', 'to_ms', 'clock_ms', 'hex']],
        on='mvt_id', suffixes=('_old', '_final'), how='left', validate='one_to_one')
    if both.to_ms_final.isna().any() or not both.airport_old.eq(both.airport_final).all():
        raise ValueError('Old departures are not a same-airport subset of final')
    jitter = both.to_ms_final-both.to_ms_old
    if not jitter.abs().le(12000).all():
        raise ValueError('Unexpected old/final takeoff jitter beyond frozen source description')
    clock_equal = both.clock_ms_old.eq(both.clock_ms_final) | (both.clock_ms_old.isna() & both.clock_ms_final.isna())
    hex_equal = both.hex_old.eq(both.hex_final) | (both.hex_old.isna() & both.hex_final.isna())
    return dict(rows=len(both), takeoff_jitter_ms_min=int(jitter.min()),
                takeoff_jitter_ms_max=int(jitter.max()), shifted_rows=int(jitter.ne(0).sum()),
                original_clock_differences=int((~clock_equal).sum()),
                identity_differences=int((~hex_equal).sum()),
                arrays_reused_between_views=False)


def prepare(out=OUT):
    pa.set_cpu_count(2)
    if (out / 'manifest.json').exists():
        result = validate(out, portable=False)
        print('Existing production packet manifest fully revalidated', flush=True)
        return result
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    training = training_contract()
    queries, views = {}, {}
    con = seq.rotation.connection()
    try:
        for view in RANKING:
            q, provenance = prepare_query(con, view)
            path = out / view / 'queries.parquet'
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix('.parquet.tmp')
            q.to_parquet(temp, index=False)
            full.publish_identical(temp, path)
            queries[view] = q
            views[view] = dict(**provenance, query=record(path), rows=len(q),
                               days=sorted(set(q.day)), identified=int(q.hex.notna().sum()),
                               original_clock_known=int(q.clock_ms.notna().sum()))
    finally:
        con.close()
    positions, absent = position_inventory([d for v in views.values() for d in v['days']])
    result = dict(settings=SETTINGS, source_sha256=sha(Path(__file__)), helpers=helpers(),
                  training_contract=training, views=views,
                  maps=[record(seq.sf.COORDS), record(seq.sf.ENDS)],
                  positions=positions, missing_position_days=absent,
                  overlap=overlap(queries['old'], queries['final']),
                  versions=dict(numpy=np.__version__, pandas=pd.__version__, pyarrow=pa.__version__),
                  prepare_seconds=time.monotonic()-started)
    write_json(out / 'manifest.json', result)
    files = portable_files(out, result)
    (out / 'portable_files.txt').write_text(''.join(p+'\n' for p in files))
    print(json.dumps(dict(query_bytes=sum(v['query']['bytes'] for v in views.values()),
                          position_files=len(positions), position_bytes=sum(p['bytes'] for p in positions),
                          missing_position_days=absent, overlap=result['overlap'],
                          prepare_seconds=result['prepare_seconds']), indent=2), flush=True)
    return result


def portable_files(out, manifest):
    names = {relative(Path(__file__)), relative(out / 'manifest.json'),
             *manifest['helpers'].keys()}
    for item in [manifest['training_contract']['manifest'], manifest['training_contract']['design'],
                 *manifest['maps'], *manifest['positions']]:
        names.add(item['path'])
    for value in manifest['views'].values():
        names.add(value['query']['path'])
    return sorted(names)


def validate_positions(manifest):
    present = {p['path'] for p in manifest['positions']}
    expected_days = required_days([d for v in manifest['views'].values() for d in v['days']])
    absent = set(manifest['missing_position_days'])
    if len(present) != len(manifest['positions']):
        raise ValueError('Duplicate position source record')
    expected_present = {relative(seq.HEATMAP / 'positions' / f'{d}.parquet')
                        for d in expected_days if d not in absent}
    if present != expected_present or not absent.issubset(set(expected_days)):
        raise ValueError('Raw source-day inventory does not partition required days')
    for item in manifest['positions']:
        verify(item)
    for d in absent:
        if (seq.HEATMAP / 'positions' / f'{d}.parquet').exists():
            raise ValueError(f'Previously missing source day appeared: {d}; preserve frozen availability')


def validate(out=OUT, portable=False):
    manifest = read(out / 'manifest.json')
    if (manifest['source_sha256'] != sha(Path(__file__))
            or manifest['settings'] != SETTINGS or manifest['helpers'] != helpers()):
        raise ValueError('Production builder/helper/settings identity changed')
    if manifest['training_contract'] != training_contract():
        raise ValueError('Frozen 2025 training representation changed')
    current_versions = dict(numpy=np.__version__, pandas=pd.__version__, pyarrow=pa.__version__)
    if manifest['versions'] != current_versions:
        raise ValueError('Portable serialization/array versions differ')
    for item in manifest['maps']:
        verify(item)
    if set(manifest['views']) != set(RANKING):
        raise ValueError('Prepared ranking view set changed')
    for view, value in manifest['views'].items():
        query = pd.read_parquet(verify(value['query']))
        validate_query(query, view)
        if (value['rows'] != len(query) or value['days'] != sorted(set(query.day))
                or value['identified'] != int(query.hex.notna().sum())
                or value['original_clock_known'] != int(query.clock_ms.notna().sum())):
            raise ValueError('Prepared query metadata differs from packet')
        if not portable:
            for field in ['raw_ranking', 'identity', 'identity_manifest', 'identity_design']:
                verify(value[field])
    validate_positions(manifest)
    return manifest


def view_design(out, view, manifest):
    value = manifest['views'][view]
    days = set(required_days(value['days']))
    return dict(settings=dict(original_sequence=seq.SETTINGS, ranking_view=view,
                              ranking_days=value['days'], labels_used=False),
                source_sha256=manifest['source_sha256'], helpers=manifest['helpers'],
                query=value['query'], identity_sha256=value['identity']['sha256'],
                identity_manifest_sha256=value['identity_manifest']['sha256'],
                raw_ranking=value['raw_ranking'], training_contract=manifest['training_contract'],
                prepared_manifest=record(out / 'manifest.json'),
                stand_map_sha256=manifest['maps'][0]['sha256'],
                runway_map_sha256=manifest['maps'][1]['sha256'],
                positions={r['path']: r['sha256'] for r in manifest['positions']
                           if Path(r['path']).stem in days},
                missing_position_days=[d for d in manifest['missing_position_days'] if d in days],
                versions=manifest['versions'])


def build(view, out=OUT, portable=False):
    started = time.monotonic()
    pa.set_cpu_count(2)
    manifest = validate(out, portable=portable)
    target = out / view
    identity = view_design(out, view, manifest)
    design_path = target / 'full_design.json'
    if design_path.exists():
        if read(design_path) != identity:
            raise ValueError('Production cache namespace/source changed; do not mix chunks')
    else:
        if (target / 'full_chunks').exists() or any((target / n).exists()
                for n in ['full_index.parquet', 'full_points.bin', 'full.json']):
            raise ValueError('Unrecorded production sequence artifacts require inspection')
        write_json(design_path, identity)
    if (target / 'full.json').exists():
        result = read(target / 'full.json')
        if any(result[k] != value for k, value in identity.items()):
            raise ValueError('Completed production cache identity changed')
        for filename, digest in result['outputs'].items():
            if sha(target / filename) != digest:
                raise ValueError('Completed production cache bytes changed')
        print(f'{view}: completed production cache hash checked', flush=True)
        return result
    query = pd.read_parquet(verify(manifest['views'][view]['query']))
    con = seq.rotation.connection()
    try:
        records = {}
        for i, (day, own) in enumerate(query.groupby('day', sort=True)):
            records[day] = full.build_chunk(con, target, day, own)
            if (i+1) % 15 == 0:
                print(f'{view}: {i+1}/{query.day.nunique()} days in {time.monotonic()-started:.1f}s', flush=True)
        ids = query.mvt_id.to_numpy().copy()
        del query
        gc.collect()
        # Guard against source replacement or newly added boundary files during the pass.
        validate_positions(manifest)
        result = full.assemble(target, ids, records, identity, started)
        write_json(target / 'execution.json', dict(portable=portable,
            scope=('query packets and raw positions revalidated; original ranking/identity hashes '
                   'retained from desktop preparation' if portable else 'all original sources revalidated'),
            prepared_manifest_sha256=sha(out / 'manifest.json'), output_manifest_sha256=sha(target / 'full.json')))
        print(json.dumps(dict(view=view, **{k: result[k] for k in
                         ['rows', 'stored_points', 'raw_array_bytes', 'elapsed_sec']}), indent=2), flush=True)
        return result
    finally:
        con.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'check', 'build'])
    parser.add_argument('--view', choices=['old', 'final', 'both'], default='both')
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--portable', action='store_true')
    args = parser.parse_args()
    if args.stage == 'prepare':
        if args.portable:
            parser.error('prepare requires desktop original sources')
        prepare(args.out)
    elif args.stage == 'check':
        validate(args.out, portable=args.portable)
        print('Production packet/source checks passed', flush=True)
    else:
        for view in (['old', 'final'] if args.view == 'both' else [args.view]):
            build(view, args.out, portable=args.portable)
