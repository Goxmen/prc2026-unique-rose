"""Find test-like 2025 conditions without reading departure outcomes or predictions."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline')]
import disruption_features as features
import opdi_disruption_run as run
import stress_analogues as matching
import weather
from common import connect, TRAIN_GLOB, RANKING
from records import data_identity, sha256, write_json

CACHE = ROOT / 'tmp/opdi_disruption_stress'
OUT = ROOT / 'VALIDATION/reports/opdi_disruption_stress'
DESIGN = ROOT / 'VALIDATION/docs/OPDI_DISRUPTION_STRESS_CHECK.md'


def read(path):
    return json.loads(Path(path).read_text())


def inputs():
    import glob
    identity = dict(data=data_identity([*glob.glob(TRAIN_GLOB), RANKING, weather.METAR]),
        sources={str(path.relative_to(ROOT)): sha256(path) for path in
                 [Path(__file__), Path(features.__file__), Path(matching.__file__), DESIGN]})
    manifest_path = CACHE / 'inputs.json'
    cache = CACHE / 'observables.parquet'
    if manifest_path.exists():
        manifest = read(manifest_path)
        if manifest['identity'] != identity or sha256(cache) != manifest['sha256']:
            raise ValueError('Changed stress input cache')
        return pd.read_parquet(cache), manifest
    con = connect()
    con.execute('SET threads=2')
    deps, arrs = [], []
    for label, path in [('training', TRAIN_GLOB), ('ranking', str(RANKING))]:
        # No departure label or departure block-time column is selected.
        deps.append(con.execute(f'''SELECT MVT_ID_mvt mvt_id, ADEP_mvt, FLIGHT_mvt,
            MVT_TIME_UTC_mvt, SCHED_TIME_UTC_mvt, FLIGHT_ID_mvt IS NULL raw_flight_id_missing,
            '{label}' population FROM read_parquet(?) WHERE PHASE_mvt='DEP' ''', [path]).df())
        arrs.append(con.execute('''SELECT MVT_ID_mvt mvt_id, ADES_mvt, MVT_TIME_UTC_mvt,
            TAXITIME_SEC_mvt taxi_sec FROM read_parquet(?) WHERE PHASE_mvt='ARR' ''', [path]).df())
    con.close()
    dep, arr = pd.concat(deps, ignore_index=True), pd.concat(arrs, ignore_index=True)
    if dep.mvt_id.duplicated().any() or len(dep) != 2429888:
        raise ValueError('Unexpected stress input population')
    grid = features.snow_grid(pd.read_parquet(weather.METAR))
    values = features.add_features(pd.DataFrame(index=dep.index), dep, arr, grid)
    uncapped = matching.arrival_tail_checks(dep, arr)
    result = dep.rename(columns={'ADEP_mvt': 'airport', 'MVT_TIME_UTC_mvt': 'mvt_time', 'FLIGHT_mvt': 'flight'})
    result['month'] = pd.to_datetime(result.mvt_time, utc=True).dt.strftime('%Y-%m')
    result['gap_sec'] = (pd.to_datetime(result.mvt_time, utc=True)
                         - pd.to_datetime(result.SCHED_TIME_UTC_mvt, utc=True)).dt.total_seconds()
    result['gap_band'] = matching.gap_band(result.gap_sec)
    result = pd.concat([result.drop(columns='SCHED_TIME_UTC_mvt'), values, uncapped], axis=1)
    run.overnight_run.save_frame(cache, result)
    manifest = dict(identity=identity, path=str(cache.relative_to(ROOT)), rows=len(result), sha256=sha256(cache))
    write_json(manifest_path, manifest)
    return result, manifest


def summaries(source, target):
    rows = []
    for population, frame in [('2025', source), ('ranking', target)]:
        for name in matching.COHORTS:
            part = frame.loc[matching.cohort(frame, name)]
            for airport, group in part.groupby('airport'):
                item = dict(population=population, cohort=name, airport=airport, rows=len(group),
                    dates=group.mvt_time.dt.normalize().nunique(),
                    missing_id_rows=int(group.raw_flight_id_missing.sum()),
                    complete_measurements=int(np.isfinite(group[matching.MATCH_COLUMNS]).all(axis=1).sum()))
                for col in matching.SEVERITY:
                    item[col + '_min'] = float(group[col].min())
                    item[col + '_median'] = float(group[col].median())
                    item[col + '_max'] = float(group[col].max())
                rows.append(item)
    return pd.DataFrame(rows)


def save_matches(source, target, scope):
    results, selected = {}, []
    for name in matching.COHORTS:
        pairs, status = matching.match_targets(source, target, name)
        run.overnight_run.save_frame(CACHE / f'{scope}_{name}_pairs.parquet', pairs)
        run.overnight_run.save_frame(CACHE / f'{scope}_{name}_status.parquet', status)
        counts = status.groupby(['airport', 'reason']).size().rename('rows').reset_index()
        counts['cohort'], counts['scope'] = name, scope
        counts.to_csv(OUT / f'{scope}_{name}_coverage.csv', index=False)
        ids = pairs.source_id.unique()
        sample = source.loc[source.mvt_id.isin(ids)]
        results[name] = dict(target_rows=len(status), matched_target_rows=int(status.reason.eq('matched').sum()),
            matched_target_fraction=float(status.reason.eq('matched').mean()) if len(status) else None,
            historical_rows=len(ids), historical_dates=int(sample.mvt_time.dt.normalize().nunique()),
            reasons=status.reason.value_counts().to_dict(), historical_months=sample.month.value_counts().to_dict(),
            pairs_path=str((CACHE / f'{scope}_{name}_pairs.parquet').relative_to(ROOT)),
            pairs_sha256=sha256(CACHE / f'{scope}_{name}_pairs.parquet'),
            status_path=str((CACHE / f'{scope}_{name}_status.parquet').relative_to(ROOT)),
            status_sha256=sha256(CACHE / f'{scope}_{name}_status.parquet'))
        selected.extend(ids.tolist())
    return results, selected


def main():
    for variant in run.HISTORICAL:
        if read(run.DEFAULT / variant / 'result.json')['status'] != 'complete':
            raise ValueError('Run this full-data check after both fits finish')
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    frame, manifest = inputs()
    for variant in run.HISTORICAL:
        if manifest['identity']['data'] != read(run.DEFAULT / variant / 'result.json')['data_identity']:
            raise ValueError('Stress inputs differ from model-training data identities')
    source = frame.loc[frame.population.eq('training') & frame.mvt_time.dt.year.eq(2025)]
    target = frame.loc[frame.population.eq('ranking')]
    summaries(source, target).to_csv(OUT / 'cohort_severity.csv', index=False)
    results, selected = save_matches(source, target, 'all_2025')
    supplemental_months = sorted(set(source.loc[source.mvt_id.isin(selected), 'month'])
                                 & {'2025-02', '2025-03', '2025-04', '2025-05'})
    note = dict(status='prepared', design_sha256=sha256(DESIGN), source_sha256=sha256(Path(__file__)),
        matcher_sha256=sha256(Path(matching.__file__)), inputs_manifest_sha256=sha256(CACHE / 'inputs.json'),
        all_2025=results, supplemental_months=supplemental_months,
        selection_uses_departure_outcomes=False, selection_uses_predictions=False,
        unscored_month='2025-06', ranking_departure_labels_available=False,
        training_file_rows_outside_2025=int((frame.population.eq('training') & ~frame.mvt_time.dt.year.eq(2025)).sum()))
    write_json(OUT / 'selection.json', note)
    print(json.dumps(note, indent=2), flush=True)


if __name__ == '__main__':
    main()
