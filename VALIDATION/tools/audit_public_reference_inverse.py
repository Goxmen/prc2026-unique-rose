"""Independent raw-calendar/count and coefficient audit for the reference pilot."""
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'VALIDATION'))
sys.path.insert(0, str(ROOT/'VALIDATION/tools'))
import probe_public_reference_inverse as run
from records import write_json


def main():
    out = run.OUT
    assert not (out/'audit.json').exists(), 'Preserve completed audit'
    result = run.source.read(out/'result.json')
    for record in result['artifacts']:
        run.source.verify(record)
    obs = pd.read_parquet(out/'observations.parquet')
    estimates = pd.read_parquet(out/'reference_estimates.parquet')
    zones = 'case ADEP_mvt '+' '.join(f"when '{a}' then '{z}'" for a, z in run.source.ZONES.items())+' end'
    report = []
    for view in ['2025', '2026']:
        paths = ([ROOT/'data'/f'training_2025-{m:02}-01_2025-{m+1:02}-01.parquet' for m in [1, 7]] if view == '2025'
                 else [ROOT/'data/ranking.parquet'])
        with duckdb.connect(config={'threads': 1}) as con:
            con.read_parquet([str(p) for p in paths]).create_view('raw')
            raw = con.execute(f'''select MVT_ID_mvt mvt_id,ADEP_mvt airport,RUNWAY_mvt runway,STAND_mvt stand,
                strftime(cast(MVT_TIME_UTC_mvt as timestamptz) AT TIME ZONE ({zones}),'%Y-%m-%d') as day
                from raw where PHASE_mvt='DEP' and STAND_mvt is not null and RUNWAY_mvt is not null
                and strftime(cast(MVT_TIME_UTC_mvt as timestamptz) AT TIME ZONE 'UTC','%m') in ('01','07')''').df()
            raw = raw[raw.day.str[5:7].isin(['01', '07'])].reset_index(drop=True)
            con.register('metadata', raw)
            counts = con.execute('''select airport,runway,day,stand,count(*) n from metadata group by all''').df()
        expected = run.metadata(view)
        columns = ['mvt_id', 'airport', 'day', 'runway', 'stand']
        a, b = [f[columns].sort_values('mvt_id').reset_index(drop=True) for f in [raw, expected]]
        pd.testing.assert_frame_equal(a, b, check_dtype=False)
        nchecked, max_held_error, max_full_error = 0, 0., 0.
        for (airport, runway), c in obs[obs.view.eq(view)].groupby(['airport', 'runway']):
            c = c.sort_values('day').reset_index(drop=True)
            q = counts[counts.airport.eq(airport) & counts.runway.eq(runway)]
            matrix = q.pivot(index='day', columns='stand', values='n').fillna(0).reindex(c.day).fillna(0)
            matrix = matrix.loc[:, matrix.sum() > 0]
            np.testing.assert_array_equal(matrix.sum(axis=1), c.metadata_n)
            np.testing.assert_array_equal(c.held, c.day.str[-2:].astype(int)%4 == 0)
            assert c.day.str[-2:].ne('01').all()
            X = matrix.to_numpy()/c.metadata_n.to_numpy()[:, None]
            prediction, baseline, _, _ = run.solve(X, c.reference.to_numpy(), c.month.to_numpy(), ~c.held.to_numpy())
            np.testing.assert_allclose(prediction, c.prediction, atol=1e-8, rtol=0)
            np.testing.assert_allclose(baseline, c.baseline, atol=1e-10, rtol=0)
            max_held_error = max(max_held_error, float(np.max(abs(prediction-c.prediction))))
            for month, ix in c.groupby('month').indices.items():
                ref = estimates[estimates.view.eq(view) & estimates.airport.eq(airport)
                                & estimates.runway.eq(runway) & estimates.month.eq(month)].set_index('stand')
                full = X[ix]@ref.loc[matrix.columns, 'reference'].to_numpy()
                np.testing.assert_allclose(full, c.full_fit.iloc[ix], atol=1e-8, rtol=0)
                max_full_error = max(max_full_error, float(np.max(abs(full-c.full_fit.iloc[ix]))))
            nchecked += len(c)
        report.append(dict(view=view, raw_metadata_rows=len(raw), cells=nchecked,
            independent_sql_raw_calendar_stand_counts_exact=True,
            held_prediction_replay_max_error=max_held_error, full_coefficient_replay_max_error=max_full_error))
    with duckdb.connect(config={'threads': 1}) as con:
        con.read_parquet(str(out/'observations.parquet')).create_view('o')
        con.read_parquet(str(run.source.SOURCE/'daily_runway.parquet')).create_view('s')
        n, errors = con.execute('''select count(*),sum(case when o.n!=s.n or abs(o.reference-60*s.ref_min/s.n)>1e-8
            then 1 else 0 end) from o join s using(airport,day,runway)''').fetchone()
        assert n == len(obs) and errors == 0
        scored = con.execute('''select view,count(*) cells,sqrt(avg(pow(baseline-reference,2))) baseline_rmse,
            sqrt(avg(pow(prediction-reference,2))) candidate_rmse from o where held group by view''').df()
        for row in scored.to_dict('records'):
            expected = result['public_holdout'][row['view']]
            assert row['cells'] == expected['cells']
            for key in ['baseline_rmse', 'candidate_rmse']:
                np.testing.assert_allclose(row[key], expected[key], atol=1e-10, rtol=0)
    value = dict(status='pass', source=run.source.art(__file__), result=run.source.art(out/'result.json'),
        checks=report, independent_sql_public_targets_scores=True,
        only_public_reference_targets=True, not_competition_validation=True)
    write_json(out/'audit.json', value)
    print(value, flush=True)


if __name__ == '__main__':
    with threadpool_limits(1):
        main()
