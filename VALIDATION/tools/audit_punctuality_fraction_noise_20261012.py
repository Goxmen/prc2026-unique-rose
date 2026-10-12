"""Independent permitted-label SQL/source audit of the fractional OTP pilot."""
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'VALIDATION/runs/v3_20261012_punctuality_fraction_noise'
OUT = ROOT / 'VALIDATION/reports/punctuality_counts_20261012/fractional_noise_audit.json'
MONTHS = ['2025-03', '2025-09', '2025-10', '2025-11']


def art(path):
    path = Path(path)
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    assert not OUT.exists()
    result = json.loads((RUN / 'result.json').read_text())
    identity = json.loads((RUN / 'identity.json').read_text())
    assert result['identity'] == art(RUN / 'identity.json')
    assert identity['design']['label_months'] == MONTHS
    for rec in [identity['source'], *identity['files'], *result['artifacts']]:
        assert art(ROOT / rec['path']) == rec
    with duckdb.connect(config={'threads': 1, 'memory_limit': '256MB'}) as con:
        con.execute("SET TimeZone='UTC'")
        labels = ROOT / 'VALIDATION/runs/v3_20261011_daily_offblock_alignment/rows2025.parquet'
        days = con.execute('''SELECT month,day,count(*) n,
            avg(CASE WHEN gap-y<=900 THEN 1.0 ELSE 0.0 END) own_otp
            FROM read_parquet(?) WHERE airport='LIRF'
            AND month IN (SELECT unnest(?)) AND fltrule IS DISTINCT FROM 'V'
            GROUP BY month,day ORDER BY day''', [str(labels), MONTHS]).df()
        arrival = con.execute('''SELECT strftime(t,'%Y-%m-%d') AS "day",
            count(*) arrivals,count(inblock-arrival_schedule) valid_arrivals,
            avg(CASE WHEN inblock IS NULL OR arrival_schedule IS NULL THEN NULL
                WHEN epoch(inblock)-epoch(arrival_schedule)<=900 THEN 1.0 ELSE 0.0 END) own_arrival_otp
            FROM read_parquet(?) WHERE airport='LIRF' AND phase='ARR'
            AND strftime(t,'%Y-%m') IN (SELECT unnest(?)) GROUP BY day ORDER BY day''',
            [str(ROOT / 'data/derived/aircraft_rotation_20261010/identity.parquet'), MONTHS]).df()
    saved = pd.read_parquet(RUN / 'days.parquet').sort_values('day').reset_index(drop=True)
    assert list(days.day) == list(saved.day) and int(days.n.sum()) == result['label_rows'] == 52681
    for name in ['n', 'own_otp']:
        np.testing.assert_array_equal(days[name], saved[name])
    public = json.loads((ROOT / 'data/external/eurocontrol_performance_20261011/pun_apt_data.json').read_text())
    pos = {name: i for i, name in enumerate(public[0])}
    values = {r[pos['Date']]: (r[pos['Departure Punctuality %']], r[pos['Arrival Punctuality %']])
              for r in public[2:] if r[pos['Airport']] == 'Rome Fiumicino'}
    for name, index in [('public_otp', 0), ('public_arrival_otp', 1)]:
        np.testing.assert_array_equal(saved[name], [float(values[day][index]) for day in saved.day])
    error = saved.public_otp.to_numpy() - days.own_otp.to_numpy()
    corrected = np.empty(len(days))
    folds = {r['query_month']: r for r in result['calibration_folds']}
    for month in MONTHS:
        select = days.month.eq(month).to_numpy()
        refs = error[~select]
        bias = float(np.median(refs))
        corrected[select] = error[select] - bias
        e = corrected[select]
        expected = dict(bias=bias, reference_mad=1.4826*np.median(abs(refs-np.median(refs))),
                        query_mad=1.4826*np.median(abs(e-np.median(e))), query_rmse=np.sqrt(np.mean(e*e)),
                        query_mae=np.mean(abs(e)), query_mean=np.mean(e))
        assert folds[month]['reference_months'] == [m for m in MONTHS if m != month]
        for key, value in expected.items():
            np.testing.assert_allclose(value, folds[month][key], atol=1e-14, rtol=0)
    np.testing.assert_array_equal(corrected, saved.lomo_error)
    np.testing.assert_allclose(np.sqrt(np.mean(corrected**2)), result['training_fraction_noise']['lomo_rmse'], atol=1e-14, rtol=0)
    witness = pd.read_parquet(RUN / 'arrival_witness.parquet').sort_values('day').reset_index(drop=True)
    assert list(arrival.day) == list(witness.day)
    for name in ['arrivals', 'valid_arrivals', 'own_arrival_otp']:
        np.testing.assert_array_equal(arrival[name], witness[name])
    np.testing.assert_array_equal(witness.arrival_fraction_error, saved.public_arrival_otp-arrival.own_arrival_otp)
    for record in result['arrival_witness']['statistics']:
        ix = np.ones(len(saved), bool) if record['month'] == 'ALL' else saved.month.eq(record['month']).to_numpy()
        x = witness.arrival_fraction_error.to_numpy()[ix]
        for target, values in [('error', error), ('lomo_error', corrected)]:
            np.testing.assert_allclose(np.corrcoef(x, values[ix])[0, 1], record['statistics'][target]['correlation'], atol=1e-13, rtol=0)
    prior = pd.read_parquet(RUN / 'prior_uncertainty.parquet')
    np.testing.assert_allclose(prior.independent_variance, prior.variance_sum/prior.n**2, atol=1e-15, rtol=0)
    np.testing.assert_allclose(prior.common_variance, (15*prior.loading_sum/prior.n)**2, atol=1e-15, rtol=0)
    np.testing.assert_allclose(prior.prior_sd, np.sqrt(prior.independent_variance+prior.common_variance), atol=1e-15, rtol=0)
    for r in result['noise_vs_prior']:
        median = float(prior.loc[prior.month.eq(r['prior_reference_month']), 'prior_sd'].median())
        np.testing.assert_allclose(r['rmse_to_median_prior_sd'], folds[r['noise_query_month']]['query_rmse']/median, atol=1e-13, rtol=0)
    receipt = dict(status='pass', source=art(__file__), result=art(RUN/'result.json'), days=len(days),
                   departure_rows=52681, arrival_rows=int(arrival.arrivals.sum()),
                   all_permitted_departure_and_arrival_daily_values_rebuilt_in_sql=True,
                   every_public_fraction_and_lomo_calibration_exact=True,
                   arrival_correlations_and_prior_variance_arithmetic_checked=True,
                   scope='Source noise and aggregation audit. Individual retained-support prior probabilities and membership-overlap diagnostics retain their separate reviewed implementation; not independently reimplemented here.',
                   no_held_departure_labels=True, no_fit=True, no_cv=True)
    OUT.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(dict(status='pass', days=len(days), rows=52681)))


if __name__ == '__main__':
    main()
