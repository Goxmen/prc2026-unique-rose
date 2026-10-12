"""Independent split, covariance, SQL and uncertainty audit of the fixed screen."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import multivariate_t
import public_runway_covariance_cv as screen

ROOT, OUT = screen.ROOT, screen.OUT
pilot = screen.pilot
read, art, write, verify = screen.read, screen.art, screen.write, screen.verify


def covariance_check(con, cells, beta, common, local, counts):
    target = cells[['omitted_count', 'omitted_sum']].to_numpy(float)
    X = np.column_stack([np.ones(len(cells)), cells.n / 100.])
    z = (target - X @ np.asarray(beta)) / np.sqrt(cells.n.to_numpy())[:, None]
    med = np.median(z, axis=0)
    scale = np.maximum(1.4826 * np.median(abs(z - med), axis=0), [1e-6, 1e-3])
    selected = np.max(abs(z - med) / scale, axis=1) <= 3
    frame = pd.DataFrame(dict(runway=cells.runway.to_numpy()[selected], a=z[selected, 0], b=z[selected, 1]))
    con.register('residuals', frame)
    def matrix(where='', parameters=None):
        a, b, c = con.execute('select covar_samp(a,a),covar_samp(a,b),covar_samp(b,b) from residuals ' + where,
                             parameters or []).fetchone()
        return np.array([[a, b], [b, c]])
    global_covariance = matrix()
    np.testing.assert_allclose(global_covariance, common, atol=1e-9, rtol=1e-11)
    for runway, cov in local.items():
        n = con.execute('select count(*) from residuals where runway=?', [runway]).fetchone()[0]
        assert n == counts[runway]
        specific = matrix('where runway=?', [runway]) if n >= 3 else global_covariance
        alpha = n / (n + 100.)
        np.testing.assert_allclose(alpha * specific + (1 - alpha) * global_covariance, cov, atol=1e-9, rtol=1e-11)
        assert np.linalg.eigvalsh(cov).min() > 0
    con.unregister('residuals')


def main():
    assert not (OUT / 'audit.json').exists()
    result = read(OUT / 'result.json')
    identity = screen.guard()
    assert result['identity'] == art(OUT / 'identity.json')
    source_result = read(pilot.OUT / 'result.json')
    cells = pd.read_parquet(pilot.parent.OUT / 'calibration.parquet')
    source_scores = pd.read_parquet(pilot.OUT / 'scored_source_cells.parquet')
    params = read(OUT / 'parameters.json')
    fits = read(pilot.OUT / 'fits.json')
    total = 0
    with duckdb.connect(config={'threads': 1, 'memory_limit': '512MB'}) as con:
        for item in fits:
            fold, month = item['fold'], item['held_month']
            subset = cells[cells.fold.eq(fold)].copy()
            subset['month'] = pd.to_datetime(subset.local_day).dt.strftime('%Y-%m')
            train = subset[subset.month.isin(item['train_months'])].copy()
            dates = pd.DatetimeIndex(train.local_day).tz_localize('Europe/Rome')
            start = dates.tz_convert('UTC').strftime('%Y-%m')
            # Independent local-date endpoint construction, including DST.
            end = (pd.DatetimeIndex(train.local_day) + pd.Timedelta(days=1)).tz_localize('Europe/Rome') - pd.Timedelta(nanoseconds=1)
            train = train[start.isin(item['train_months']) & end.tz_convert('UTC').strftime('%Y-%m').isin(item['train_months'])]
            held = subset[subset.month.eq(month)].copy()
            assert len(train) == item['train_cells'] and len(held) == item['held_cells']
            assert month not in item['train_months'] and set(item['train_months']).issubset(params[fold]['taxi']['allowed'])
            covariance_check(con, train, item['coefficients'], item['covariance'], item['runway_covariances'], item['counts'])
            expected = source_scores[source_scores.fold.eq(fold) & source_scores.month.eq(month)].set_index(['day', 'runway'])
            assert len(expected) == len(held)
            for row in held.itertuples():
                residual = np.array([row.omitted_count, row.omitted_sum]) - np.array([1., row.n / 100.]) @ item['coefficients']
                for label, cov in [('control', item['covariance']), ('candidate', item['runway_covariances'].get(row.runway, item['covariance']))]:
                    shape = .6 * (row.n * np.asarray(cov) + np.diag([1., 3600.]))
                    sd = np.sqrt(np.diag(shape))
                    nll = -multivariate_t.logpdf(residual / sd, shape=shape / np.outer(sd, sd), df=5) + np.log(sd).sum() - np.log(2 * np.pi)
                    np.testing.assert_allclose(nll, expected.loc[(str(row.local_day), row.runway), label], atol=1e-9, rtol=0)
                total += 1
        assert total == len(source_scores) == 810
        for fold, p in params.items():
            c = cells[cells.fold.eq(fold)]
            covariance_check(con, c, p['taxi']['coefficients'], p['taxi']['covariance_per_row'],
                             p['taxi']['runway_covariances'], p['taxi']['runway_covariance_cells'])
        for name, path in [('f', OUT / 'scored.parquet'), ('c', screen.CURRENT / 'scored.parquet'),
            ('n', screen.old.OUT / 'scored.parquet'), ('p', OUT / 'predictions.parquet'),
            ('raw', pilot.parent.daily.base.OUT / 'rows2025.parquet')]:
            con.read_parquet(str(path)).create_view(name)
        assert con.execute('select count(*),count(distinct mvt_id) from f').fetchone() == (344419, 344419)
        assert con.execute('''select count(*) from f full join c using(mvt_id) full join n using(mvt_id)
            join raw using(mvt_id) left join p using(mvt_id) where f.mvt_id is null or c.mvt_id is null or n.mvt_id is null
            or f.y is distinct from raw.y or f.month is distinct from raw.month or f.day is distinct from raw.day
            or f.control is distinct from c.three_tabm_runway or f.native_control is distinct from n.candidate
            or f.posterior_delta is distinct from coalesce(p.posterior_delta,0)
            or f.candidate is distinct from round_even(greatest(0,f.control+f.native_candidate-f.native_control),0)''').fetchone()[0] == 0
        rng = np.random.default_rng(0)
        draws = np.zeros(2000)
        for month in sorted(result['scores']):
            actual = con.execute('select sqrt(avg(pow(control-y,2))),sqrt(avg(pow(candidate-y,2))) from f where month=?', [month]).fetchone()
            np.testing.assert_allclose(actual, [result['scores'][month][k] for k in ['control', 'candidate']], atol=1e-8, rtol=0)
            days = np.asarray(con.execute('''select count(*),sum(pow(control-y,2)),sum(pow(candidate-y,2))
                from (select *,row_number() over() pos from f) where month=? group by day order by min(pos)''', [month]).fetchall())
            sample = days[rng.integers(0, len(days), (2000, len(days)))].sum(axis=1)
            draws += (np.sqrt(sample[:, 2] / sample[:, 0]) - np.sqrt(sample[:, 1] / sample[:, 0])) / 2
        ci = np.percentile(draws, [2.5, 97.5])
        np.testing.assert_allclose(ci, result['paired']['ci95'], atol=1e-8, rtol=0)
        gate = bool(ci[1] < 0 and all(s['candidate'] < s['control'] for s in result['scores'].values()))
        assert gate == result['gate']
    value = dict(status='pass', identity=art(OUT / 'identity.json'), result=art(OUT / 'result.json'),
        source=art(Path(__file__)), source_cells=total, challenge_rows=344419, inner_month_and_DST_boundaries_exact=True,
        independent_SQL_covariances_and_density_checks=True, exact_v80_SQL_truth_transport_scores=True,
        independent_CV_bootstrap=True, gate=gate, source_pilot_gate=source_result['gate'],
        scope='Source cells and CV development data are reused; no independent confirmation claim.')
    write(OUT / 'audit.json', value)
    print(dict(status='pass', source_cells=total, cv=result['cv'], paired=result['paired'], gate=gate))


if __name__ == '__main__':
    main()
