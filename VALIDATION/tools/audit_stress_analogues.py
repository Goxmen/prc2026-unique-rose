"""Check accepted matches against literal limits and uncapped arrivals in SQL."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]
import prepare_stress_analogues as prepare
import stress_analogues as matching
import opdi_disruption_run as run
from common import connect, TRAIN_GLOB, RANKING
from records import sha256, write_json


def read(path):
    return json.loads(Path(path).read_text())


def mask(frame, name):
    snow = frame.dis_snow_24h_fraction > .25
    congestion = ((frame.dis_arr_60m_count >= 5) & (frame.dis_arr_60m_mean_sec >= 1800)) | (
        (frame.dis_arr_180m_count >= 10) & (frame.dis_arr_180m_mean_sec >= 1800))
    return snow if name == 'prolonged_snow' else congestion if name == 'heavy_arrival_congestion' else snow & congestion


def main():
    inputs, _ = prepare.inputs()
    source = inputs.loc[inputs.population.eq('training') & inputs.mvt_time.dt.year.eq(2025)]
    target = inputs.loc[inputs.population.eq('ranking')].set_index('mvt_id')
    heldout = pd.read_parquet(prepare.CACHE / 'heldout_observables.parquet')
    frozen = read(prepare.OUT / 'heldout_selection.json')
    if sha256(prepare.CACHE / 'heldout_observables.parquet') != frozen['heldout_observables_sha256']:
        raise ValueError('Changed held-out matching inputs')
    jobs = {j['id']: j for j in run.overnight_run.schedule() if j['scheme'] == 'v3'}
    for job, part in heldout.groupby('model_id'):
        if part.month.isin(jobs[job]['train']).any():
            raise ValueError('In-sample month entered held-out matching')
    checked = {}
    for scope, population in [('all_2025', source), ('heldout', heldout)]:
        indexed = population.set_index('mvt_id')
        for name in matching.COHORTS:
            pairs = pd.read_parquet(prepare.CACHE / f'{scope}_{name}_pairs.parquet')
            statuses = pd.read_parquet(prepare.CACHE / f'{scope}_{name}_status.parquet')
            if not set(pairs.target_id.unique()) == set(statuses.loc[statuses.reason.eq('matched'), 'mvt_id']):
                raise ValueError('Coverage does not agree with accepted edges')
            if pairs.groupby('target_id').size().gt(10).any():
                raise ValueError('Too many neighbours per target')
            a, b = indexed.loc[pairs.source_id].reset_index(drop=True), target.loc[pairs.target_id].reset_index(drop=True)
            if not len(pairs):
                checked[f'{scope}/{name}'] = 0
                continue
            for column in ('airport', 'raw_flight_id_missing', 'gap_band'):
                np.testing.assert_array_equal(a[column], b[column])
            if not mask(a, name).all() or not mask(b, name).all():
                raise ValueError('Accepted row is outside the stated stress group')
            if not np.isfinite(a[matching.MATCH_COLUMNS].to_numpy()).all() or not np.isfinite(b[matching.MATCH_COLUMNS].to_numpy()).all():
                raise ValueError('Accepted match has missing measurements')
            for column, limit in [('dis_snow_6h_fraction', 1/6), ('dis_snow_24h_fraction', 1/24)]:
                if (np.abs(a[column].to_numpy() - b[column].to_numpy()) > limit + 1e-6).any():
                    raise ValueError('Snow-duration limit violated')
            for minutes in (60, 180):
                counts = (a[f'dis_arr_{minutes}m_count'] + 1) / (b[f'dis_arr_{minutes}m_count'] + 1)
                if ((counts < 1/1.5 - 1e-6) | (counts > 1.5 + 1e-6)).any():
                    raise ValueError('Traffic-count limit violated')
                for column in (f'dis_arr_{minutes}m_mean_sec', f'stress_arr_{minutes}m_uncapped_mean_sec'):
                    ratio = np.maximum(a[column], 60) / np.maximum(b[column], 60)
                    if ((ratio < .8 - 1e-6) | (ratio > 1.25 + 1e-6)).any():
                        raise ValueError('Arrival-mean limit violated')
                for suffix, limit in [('over30_share', .1), ('over60_share', .05), ('over120_share', .025)]:
                    prefix = 'stress' if suffix == 'over120_share' else 'dis'
                    column = f'{prefix}_arr_{minutes}m_{suffix}'
                    if (np.abs(a[column] - b[column]) > limit + 1e-6).any():
                        raise ValueError('Arrival-tail limit violated')
            pool = population.loc[mask(population, name)]
            pool = pool.loc[np.isfinite(pool[matching.MATCH_COLUMNS].to_numpy()).all(axis=1)]
            maxima = pool.groupby('airport')[matching.SEVERITY].max()
            ceiling = maxima.loc[b.airport].to_numpy()
            if (b[matching.SEVERITY].to_numpy() > ceiling + 1e-7).any():
                raise ValueError('An extrapolated target was falsely accepted')
            checked[f'{scope}/{name}'] = len(pairs)
    # Independently recompute uncapped arrival means and >2h shares on severe
    # and randomly chosen source/target rows; no matching aggregation is reused.
    stress = inputs.loc[mask(inputs, 'prolonged_snow') | mask(inputs, 'heavy_arrival_congestion')]
    sample = pd.concat([part.sample(n=min(40, len(part)), random_state=14)
                        for _, part in stress.groupby('population')], ignore_index=True)
    sample['audit_id'] = np.arange(len(sample))
    con = connect()
    con.execute('SET threads=2')
    con.register('queries', sample[['audit_id', 'airport', 'mvt_time']])
    con.execute(f'''CREATE TEMP TABLE donors AS
        SELECT ADES_mvt airport, TAXITIME_SEC_mvt::DOUBLE taxi,
               MVT_TIME_UTC_mvt::TIMESTAMP + TAXITIME_SEC_mvt * INTERVAL '1 second' completed
        FROM read_parquet('{TRAIN_GLOB}') WHERE PHASE_mvt='ARR'
        AND isfinite(TAXITIME_SEC_mvt::DOUBLE) AND TAXITIME_SEC_mvt >= 0
        UNION ALL
        SELECT ADES_mvt, TAXITIME_SEC_mvt::DOUBLE,
               MVT_TIME_UTC_mvt::TIMESTAMP + TAXITIME_SEC_mvt * INTERVAL '1 second'
        FROM read_parquet('{RANKING}') WHERE PHASE_mvt='ARR'
        AND isfinite(TAXITIME_SEC_mvt::DOUBLE) AND TAXITIME_SEC_mvt >= 0''')
    errors = {}
    for minutes in (60, 180):
        actual = con.execute(f'''SELECT q.audit_id, avg(d.taxi) mean_sec,
            avg((d.taxi > 7200)::DOUBLE) tail_share FROM queries q LEFT JOIN donors d
            ON d.airport=q.airport AND d.completed < q.mvt_time::TIMESTAMP
            AND d.completed >= q.mvt_time::TIMESTAMP - INTERVAL '{minutes} minutes'
            GROUP BY q.audit_id ORDER BY q.audit_id''').df()
        for suffix, column in [('uncapped_mean_sec', 'mean_sec'), ('over120_share', 'tail_share')]:
            name = f'stress_arr_{minutes}m_{suffix}'
            np.testing.assert_allclose(sample[name], actual[column], atol=1e-7, rtol=0, equal_nan=True)
            errors[name] = float(np.nanmax(np.abs(sample[name].to_numpy() - actual[column].to_numpy())))
    con.close()
    result = dict(status='passed', accepted_edges_checked=checked, heldout_months_disjoint_from_training=True,
        sql_sampled_rows=len(sample), sql_values_checked=len(sample)*4, maximum_sql_error=errors,
        source_sha256=sha256(Path(__file__)), results_sha256=sha256(prepare.OUT / 'results.json'))
    write_json(prepare.OUT / 'independent_audit.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
