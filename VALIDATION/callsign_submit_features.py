"""Full-year callsign inputs, with independent original/final ranking clocks.

Only orchestration is new. The audited aircraft matcher, interval arithmetic,
and arrival comparison are imported unchanged. Existing 2025 identity/shared
clock caches are read-only. Training arrival references are all2025; ranking
references are all2025 plus the query's own supplied ranking month. Old/final
movement IDs overlap but never share a feature or identity table.

This module builds features only. It has no network download, model fit or
submission/upload entry point. Heavy build stages require caller scheduling.
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

import aircraft_rotation as rotation
import callsign_clocks as clocks

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'data/derived/callsign_submit_20261010'
CV = ROOT / 'VALIDATION/runs/v3_20261010_nn_callsign_clocks_jan_jul'
CV_SCORE = CV / 'score_g42_142_242_k0_1_2_3_4.json'
TRAIN_MONTHS = [f'2025-{m:02d}' for m in range(1, 13)]
RANKING = {'old': ROOT / 'data/ranking.parquet',
           'final': ROOT / 'data/final_ranking.parquet'}
RANK_MONTHS = {'old': ['2026-01', '2026-07'],
               'final': ['2026-01', '2026-02', '2026-06', '2026-07']}
EXPECTED_ROWS = {'old': 344841, 'final': 670790}
FEATURE_NAMES = list(clocks.FEATURE_NAMES)
SETTINGS = dict(feature_names=FEATURE_NAMES, callsign=clocks.SETTINGS,
    identity=rotation.SETTINGS, training_references=TRAIN_MONTHS,
    ranking_references='all2025 plus own ranking month only',
    views='old and final isolated even for equal movement IDs',
    training_cache='reuse audited2025 identity and shared clocks without mutation',
    cross_year_identity_conflict='reject ranking conflicts; fail if cached2025 identity would change',
    label_use='supplied arrival taxi-in only; no departure label or block read')

sha = rotation.sha256
write = rotation.write_json


def read(path):
    return json.loads(Path(path).read_text())


def artifact(path):
    return dict(path=str(Path(path).relative_to(ROOT)), sha256=sha(path))


def verify_artifact(record):
    path = ROOT / record['path']
    if sha(path) != record['sha256']:
        raise ValueError(f'Changed artifact: {path}')
    return path


def evidence():
    score = read(CV_SCORE)
    pool = read(CV / 'callsign_pool15_manifest.json')
    if (score['pool_size'] != 15 or score['groups'] != [42, 142, 242]
            or score['nets'] != list(range(5))
            or abs(score['scores']['v58']['cv'] - 295.7414415741871) > 1e-7
            or score['comparisons']['callsign_clocks_vs_v58']['decision'] != 'improvement'
            or score['pool_expansion'] != pool):
        raise ValueError('Full15 callsign CV evidence changed or is incomplete')
    if score['source_sha256'] != sha(ROOT / 'VALIDATION/nn_callsign_clocks_jan_jul.py'):
        raise ValueError('Validated callsign runner changed')
    if pool['source_sha256'] != sha(ROOT / 'VALIDATION/nn_callsign_pool15.py'):
        raise ValueError('Validated pool expansion changed')
    for path, digest in pool['helpers'].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f'Validated helper changed: {path}')
    return dict(full15_score=artifact(CV_SCORE), full15_pool=artifact(CV / 'callsign_pool15_manifest.json'))


def guard(out=DEFAULT):
    """Freeze source/evidence identities before writing any feature artifact."""
    out.mkdir(parents=True, exist_ok=True)
    identity = read(rotation.DEFAULT / 'identity.json')
    original = read(clocks.DEFAULT / 'manifest.json')
    if identity['source_sha256'] != sha(Path(rotation.__file__)):
        raise ValueError('2025 aircraft identity builder changed')
    if original['source_sha256'] != sha(Path(clocks.__file__)) or original['settings'] != clocks.SETTINGS:
        raise ValueError('2025 callsign builder changed')
    if identity['identity_sha256'] != original['identity_sha256']:
        raise ValueError('2025 identity and callsign caches disagree')
    current = dict(settings=SETTINGS, source_sha256=sha(Path(__file__)), evidence=evidence(),
        helpers={str(Path(m.__file__).relative_to(ROOT)): sha(Path(m.__file__)) for m in [rotation, clocks]},
        original_identity_manifest=artifact(rotation.DEFAULT / 'identity.json'),
        original_callsign_manifest=artifact(clocks.DEFAULT / 'manifest.json'),
        original_identity_sha256=identity['identity_sha256'],
        original_shared_sha256=original['shared_sha256'])
    path = out / 'design.json'
    if path.exists():
        if read(path) != current:
            raise ValueError('Production feature design/source/evidence changed; use a separate directory')
    else:
        if list(out.glob('**/*.parquet')):
            raise ValueError('Unrecorded production feature files require inspection')
        write(path, current)
    return current


def original_events():
    path = rotation.DEFAULT / 'identity.parquet'
    manifest = read(rotation.DEFAULT / 'identity.json')
    if sha(path) != manifest['identity_sha256']:
        raise ValueError('Original2025 aircraft identities changed')
    for source in manifest['movement_sources']:
        if sha(Path(source['path'])) != source['sha256']:
            raise ValueError('A source2025 movement file changed')
    events = pd.read_parquet(path)
    if set(events.month) != set(TRAIN_MONTHS) or events.mvt_id.duplicated().any():
        raise ValueError('2025 identity population changed')
    return events


def existing(out, name):
    record_path = out / f'{name}.json'
    data_path = out / f'{name}.parquet'
    if record_path.exists():
        record = read(record_path)
        if record['design_sha256'] != sha(out / 'design.json'):
            raise ValueError('Feature record design changed')
        verify_artifact(record['artifact'])
        for source in record.get('sources', []):
            verify_artifact(source)
        return record
    if data_path.exists():
        raise ValueError(f'Unrecorded file: {data_path}; inspect interrupted stage')
    return None


def save_table(out, name, table, **details):
    path = out / f'{name}.parquet'
    if path.exists() or path.with_suffix('.json').exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.parquet.tmp')
    table.to_parquet(temp, index=False)
    temp.replace(path)
    record = dict(design_sha256=sha(out / 'design.json'), artifact=artifact(path), rows=len(table), **details)
    write(path.with_suffix('.json'), record)
    return record


def arrival_features(events, ids, reference_months):
    """Use the exact audited ASOF/72h/same-airport arrival semantics."""
    con = rotation.connection()
    try:
        return rotation.feature_table(events, ids, reference_months, con=con)
    finally:
        con.close()


def feature_summary(frame):
    if list(frame.columns) != ['mvt_id'] + FEATURE_NAMES or frame.mvt_id.duplicated().any():
        raise ValueError('Invalid production feature identity/order')
    if any(frame[c].dtype != np.dtype('float32') for c in FEATURE_NAMES):
        raise ValueError('All six features must retain float32 dtype')
    return {c: int(frame[c].notna().sum()) for c in FEATURE_NAMES}


def build_train(out=DEFAULT):
    guard(out)
    if existing(out, 'train'):
        return
    started = time.monotonic()
    events = original_events()
    shared_path = clocks.DEFAULT / 'shared.parquet'
    if sha(shared_path) != read(out / 'design.json')['original_shared_sha256']:
        raise ValueError('Audited2025 shared callsign clocks changed')
    shared = pd.read_parquet(shared_path).set_index('mvt_id')
    ids = events.loc[events.phase.eq('DEP'), 'mvt_id'].to_numpy()
    arrivals = arrival_features(events, ids, TRAIN_MONTHS)
    result = clocks.fold_features(shared, arrivals).reset_index()
    save_table(out, 'train', result, reference_months=TRAIN_MONTHS,
        sources=[artifact(rotation.DEFAULT / 'identity.parquet'), artifact(shared_path)],
        nonmissing=feature_summary(result), elapsed_sec=time.monotonic() - started)


def clean_rank_identities(con, identified, train):
    """The old/final view is isolated; cross-year duplicates cannot mutate2025."""
    con.register('identified', identified)
    con.register('train_identity', train[['mvt_id', 'hex', 'phase', 't']])
    con.execute('''CREATE OR REPLACE TEMP TABLE identity_raw AS
        SELECT m.*, i.hex, i.near_ms, i.position_n FROM movements m
        LEFT JOIN identified i USING(mvt_id)''')
    con.execute('''CREATE OR REPLACE TEMP TABLE known AS
        SELECT mvt_id,hex,phase,t,FALSE training FROM identity_raw WHERE hex IS NOT NULL
        UNION ALL SELECT mvt_id,hex,phase,t,TRUE FROM train_identity WHERE hex IS NOT NULL''')
    con.execute('''CREATE OR REPLACE TEMP TABLE conflicts AS SELECT mvt_id,training FROM (
        SELECT *,epoch(t)-lag(epoch(t)) OVER w prev_gap,
            lead(epoch(t)) OVER w-epoch(t) next_gap FROM known
        WINDOW w AS (PARTITION BY hex,phase ORDER BY t,mvt_id))
        WHERE prev_gap<=? OR next_gap<=?''', [rotation.SETTINGS['identity_conflict_sec']] * 2)
    if con.execute('SELECT count(*) FROM conflicts WHERE training').fetchone()[0]:
        raise ValueError('Ranking cross-year identity conflict would change the frozen2025 cache')
    result = con.execute('''SELECT r.* EXCLUDE(hex,near_ms,position_n),
        CASE WHEN c.mvt_id IS NULL THEN r.hex END hex,
        CASE WHEN c.mvt_id IS NULL THEN r.near_ms END near_ms,
        CASE WHEN c.mvt_id IS NULL THEN r.position_n END position_n,
        strftime(r.t,'%Y-%m') AS month
        FROM identity_raw r LEFT JOIN conflicts c USING(mvt_id)''').df()
    return result, con.execute('SELECT count(*) FROM conflicts').fetchone()[0]


def build_identity(view, out=DEFAULT):
    guard(out)
    name = f'{view}/identity'
    if existing(out, name):
        return
    started = time.monotonic()
    train = original_events()
    con = rotation.connection()
    try:
        # read_movements reads departure observable inputs, supplied arrival taxi-in only.
        rotation.read_movements(con, [RANKING[view]])
        months = [v[0] for v in con.execute("SELECT DISTINCT strftime(t,'%Y-%m') FROM movements ORDER BY 1").fetchall()]
        if months != RANK_MONTHS[view]:
            raise ValueError('Ranking month population changed')
        dep_n = con.execute("SELECT count(*) FROM movements WHERE phase='DEP'").fetchone()[0]
        if dep_n != EXPECTED_ROWS[view]:
            raise ValueError('Ranking departure count changed')
        days = [v[0] for v in con.execute("SELECT DISTINCT strftime(t,'%Y-%m-%d') FROM movements ORDER BY 1").fetchall()]
        matches = []
        for i, day in enumerate(days):
            matches.append(rotation.match_day(con, day))
            if (i + 1) % 20 == 0:
                print(f'{view} identities {i+1}/{len(days)} days', flush=True)
        identified = pd.concat(matches, ignore_index=True)
        if identified.mvt_id.duplicated().any():
            raise ValueError('Duplicate ranking matches')
        result, conflicts = clean_rank_identities(con, identified, train)
        if np.intersect1d(result.mvt_id, train.mvt_id).size or result.mvt_id.duplicated().any():
            raise ValueError('Training/ranking movement identities overlap')
    finally:
        con.close()
    dates = sorted({str(date.fromisoformat(d) + timedelta(days=delta)) for d in days for delta in [-1, 0, 1]})
    source_paths = [rotation.HEATMAP / kind / f'{d}.parquet'
                    for kind in ['callsigns', 'positions'] for d in dates
                    if (rotation.HEATMAP / kind / f'{d}.parquet').exists()]
    save_table(out, name, result, ranking_view=view, ranking_months=months, departure_rows=dep_n,
        conflicts=conflicts, identified=int(result.hex.notna().sum()),
        sources=[artifact(RANKING[view]), artifact(rotation.DEFAULT / 'identity.parquet')]
                + [artifact(p) for p in source_paths],
        missing_identity_days=[d for d in days if any(not (rotation.HEATMAP / kind / f'{d}.parquet').exists()
                                                    for kind in ['callsigns', 'positions'])],
        elapsed_sec=time.monotonic() - started)


def build_shared(view, out=DEFAULT):
    guard(out)
    name = f'{view}/shared'
    if existing(out, name):
        return
    identity = existing(out, f'{view}/identity')
    if identity is None:
        raise ValueError('Build ranking identities first')
    started = time.monotonic()
    dep = pd.read_parquet(ROOT / identity['artifact']['path'])
    dep = dep.loc[dep.phase.eq('DEP')].copy()
    dep['day'] = dep.t.dt.strftime('%Y-%m-%d')
    dep['to_ms'] = dep.t.astype('datetime64[ms, UTC]').astype('int64')
    history = dep.loc[dep.hex.notna()].sort_values(['hex', 'cs', 'day', 'to_ms'])
    dep['previous_departure_ms'] = history.groupby(['hex', 'cs', 'day'], sort=False).to_ms.shift().reindex(dep.index)
    frames, sources = [], set()
    for day, query in dep.groupby('day', sort=True):
        tables = []
        for delta in [-2, -1, 0, 1]:
            source_day = str(date.fromisoformat(day) + timedelta(days=delta))
            path = rotation.HEATMAP / 'callsigns' / f'{source_day}.parquet'
            if path.exists():
                tables.append(pd.read_parquet(path).assign(source_day=source_day))
                sources.add(path)
        calls = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(
            columns=['hex', 'callsign', 'records', 'first_ms', 'last_ms', 'squawk', 'source_day'])
        frames.append(clocks.day_clocks(query[['mvt_id', 'hex', 'cs', 'to_ms', 'previous_departure_ms']], calls, day))
    result = pd.concat(frames, ignore_index=True)
    if result.mvt_id.duplicated().any() or len(result) != EXPECTED_ROWS[view]:
        raise ValueError('Ranking shared clock rows changed')
    save_table(out, name, result, ranking_view=view,
        sources=[identity['artifact']] + [artifact(p) for p in sorted(sources)],
        elapsed_sec=time.monotonic() - started)


def build_ranking(view, out=DEFAULT):
    guard(out)
    if existing(out, view):
        return
    build_identity(view, out)
    build_shared(view, out)
    started = time.monotonic()
    identity = existing(out, f'{view}/identity')
    shared_record = existing(out, f'{view}/shared')
    train = original_events()
    ranking = pd.read_parquet(ROOT / identity['artifact']['path'])
    events = pd.concat([train, ranking], ignore_index=True)
    shared = pd.read_parquet(ROOT / shared_record['artifact']['path']).set_index('mvt_id')
    result, refs = ranking_features(events, ranking, shared, RANK_MONTHS[view])
    if len(result) != EXPECTED_ROWS[view]:
        raise ValueError('Ranking feature count changed')
    save_table(out, view, result, ranking_view=view, per_query_month_reference_months=refs,
        sources=[identity['artifact'], shared_record['artifact'], artifact(rotation.DEFAULT / 'identity.parquet')],
        nonmissing=feature_summary(result), elapsed_sec=time.monotonic() - started)
    del events, train, ranking, shared
    gc.collect()


def ranking_features(events, ranking, shared, months):
    """Pure scoped orchestration shared by the production build and tiny tests."""
    frames, refs = [], {}
    for month in months:
        ids = ranking.loc[ranking.phase.eq('DEP') & ranking.month.eq(month), 'mvt_id'].to_numpy()
        refs[month] = TRAIN_MONTHS + [month]
        arrivals = arrival_features(events, ids, refs[month])
        frames.append(clocks.fold_features(shared, arrivals).reset_index())
    return pd.concat(frames, ignore_index=True), refs


def load_features(view, out=DEFAULT):
    record = existing(out, view)
    if record is None:
        raise FileNotFoundError(out / f'{view}.parquet')
    table = pd.read_parquet(ROOT / record['artifact']['path'])
    feature_summary(table)
    return table.set_index('mvt_id'), record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['train', 'old', 'final'])
    parser.add_argument('--out', type=Path, default=DEFAULT)
    args = parser.parse_args()
    pa.set_cpu_count(2)
    if args.stage == 'train':
        build_train(args.out)
    else:
        build_ranking(args.stage, args.out)
