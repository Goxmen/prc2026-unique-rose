"""Training-source feasibility of partially pooled runway-specific error covariance.

Existing calibration already scales variance by count and departure count
mismatch. This screen changes only taxi source covariance, holding the original
Huber bias fit fixed. It reads audited training cells, never challenge CV errors.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import multivariate_t
from threadpoolctl import threadpool_limits

import taxi_count_sum_jan_jul as parent
from records import sha256, write_json

ROOT = parent.ROOT
OUT = ROOT / 'VALIDATION/runs/v3_20261012_public_runway_covariance_pilot'
DESIGN = dict(change='Runway-specific taxi measurement covariance, partially pooled with original global matrix',
    retained='Original Huber bias, sqrt(count) normalization, three-MAD covariance filter, t5 likelihood, diagonal floor',
    shrinkage='Selected runway training cell count divided by itself plus100; unseen/under3cell runway uses global',
    validation='Leave one calendar month out inside each allowed outer training window; drop new boundary days',
    score='Mean per-month held source negative t5 log likelihood; all held source cells including untrimmed tails',
    gate='Both outer windows improve and each paired whole-day95% interval upper endpoint below0',
    bootstrap=dict(reps=2000, seed=0), covariance_pseudocells=100,
    continuation='Only a passing fixed source screen permits challenge Jan/Jul evaluation',
    no_hyperparameter_search=True, no_challenge_cv_predictions=True, no_taxi_model_fit=True,
    no_production=True, no_upload=True, failure='Close; do not tune shrinkage or trim thresholds',
    caveat='Source calibration validation is not taxi prediction validation; outer training windows may overlap')


def art(p):
    return dict(path=str(p.relative_to(ROOT)), sha256=sha256(p))


def fit(cells):
    X = np.column_stack([np.ones(len(cells)), cells.n / 100.])
    target = cells[['omitted_count', 'omitted_sum']].to_numpy(float)
    scale = np.maximum(1.4826 * np.median(abs(target - np.median(target, axis=0)), axis=0), [1., 60.])
    beta = []
    for k in range(2):
        r = least_squares(lambda b: (X @ b - target[:, k]) / scale[k],
                          [np.median(target[:, k]), 0.], loss='huber', f_scale=1.)
        assert r.success
        beta.append(r.x)
    beta = np.asarray(beta).T
    z = (target - X @ beta) / np.sqrt(cells.n.to_numpy())[:, None]
    center = np.median(z, axis=0)
    mad = np.maximum(1.4826 * np.median(abs(z - center), axis=0), [1e-6, 1e-3])
    selected = np.max(abs(z - center) / mad, axis=1) <= 3
    assert selected.sum() >= 100
    common = np.cov(z[selected].T)
    assert np.linalg.eigvalsh(common).min() > 0
    by_runway, counts = {}, {}
    for runway in cells.runway.unique():
        keep = selected & cells.runway.eq(runway).to_numpy()
        n = int(keep.sum())
        alpha = n / (n + DESIGN['covariance_pseudocells'])
        local = np.cov(z[keep].T) if n >= 3 else common
        cov = alpha * local + (1 - alpha) * common
        assert np.linalg.eigvalsh(cov).min() > 0
        by_runway[runway], counts[runway] = cov, n
    return beta, common, by_runway, counts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'identity.json').exists(), 'Preserve completed fixed pilot'
    source = parent.OUT / 'calibration.parquet'
    params_path = parent.OUT / 'parameters.json'
    audit_path = parent.OUT / 'audit.json'
    audit = json.loads(audit_path.read_text())
    assert audit['status'] == 'pass'
    identity = dict(design=DESIGN, sources=[art(p) for p in [Path(__file__), Path(parent.__file__),
        Path(parent.daily.__file__), source, params_path, audit_path]], before_source_comparison=True)
    write_json(OUT / 'identity.json', identity)
    cells = pd.read_parquet(source)
    parameters = json.loads(params_path.read_text())
    records, fits, controls = [], [], []
    for fold, all_cells in cells.groupby('fold', sort=True):
        allowed = parameters[fold]['allowed']
        all_cells = all_cells.copy()
        all_cells['cell_month'] = pd.to_datetime(all_cells.local_day).dt.strftime('%Y-%m')
        assert set(all_cells.cell_month).issubset(allowed)
        beta, covariance, _, _ = fit(all_cells)
        np.testing.assert_array_equal(beta, parameters[fold]['coefficients'])
        np.testing.assert_array_equal(covariance, parameters[fold]['covariance_per_row'])
        controls.append(dict(fold=fold, original_bias_and_covariance_exact=True, cells=len(all_cells)))
        for month in sorted(all_cells.cell_month.unique()):
            train_months = [m for m in allowed if m != month]
            train = all_cells[all_cells.cell_month.isin(train_months)].copy()
            train = train[parent.daily.complete_days(train, train_months)]
            held = all_cells[all_cells.cell_month.eq(month)].copy()
            assert not set(train.local_day) & set(held.local_day)
            beta, common, local, counts = fit(train)
            fits.append(dict(fold=fold, held_month=month, train_months=train_months, train_cells=len(train),
                held_cells=len(held), counts=counts, coefficients=beta.tolist(), covariance=common.tolist(),
                runway_covariances={k: v.tolist() for k, v in local.items()}))
            for r in held.itertuples():
                residual = np.array([r.omitted_count, r.omitted_sum]) - np.array([1., r.n / 100.]) @ beta
                scores = []
                for matrix in [common, local.get(r.runway, common)]:
                    cov = r.n * matrix + np.diag([1., 60. ** 2])
                    shape = .6 * cov
                    sign, det = np.linalg.slogdet(shape)
                    assert sign > 0
                    mahal = residual @ np.linalg.solve(shape, residual)
                    nll = .5 * det + 3.5 * np.log1p(max(mahal, 0) / 5)
                    # Standardized SciPy calculation independently checks density algebra.
                    sd = np.sqrt(np.diag(shape))
                    z = residual / sd
                    corr = shape / np.outer(sd, sd)
                    ref = -multivariate_t.logpdf(z, shape=corr, df=5) + np.log(sd).sum()
                    # In two dimensions the t normalizing constant is log(2*pi).
                    np.testing.assert_allclose(nll + np.log(2 * np.pi), ref, atol=1e-9, rtol=0)
                    scores.append(float(nll))
                records.append(dict(fold=fold, month=month, day=str(r.local_day), runway=r.runway,
                    n=int(r.n), control=scores[0], candidate=scores[1], change=scores[1] - scores[0]))
    f = pd.DataFrame(records)
    f.to_parquet(OUT / 'scored_source_cells.parquet', index=False)
    write_json(OUT / 'fits.json', fits)
    summary = {}
    for fold, group in f.groupby('fold', sort=True):
        rng = np.random.default_rng(DESIGN['bootstrap']['seed'])
        bootstrap = np.zeros(DESIGN['bootstrap']['reps'])
        months = sorted(group.month.unique())
        scores = group.groupby('month')[['control', 'candidate']].mean()
        for month in months:
            days = group[group.month.eq(month)].groupby('day').change.agg(['sum', 'count']).to_numpy()
            sampled = days[rng.integers(0, len(days), (len(bootstrap), len(days)))].sum(axis=1)
            bootstrap += sampled[:, 0] / sampled[:, 1] / len(months)
        change = float((scores.candidate - scores.control).mean())
        ci = np.percentile(bootstrap, [2.5, 97.5]).tolist()
        summary[fold] = dict(control=float(scores.control.mean()), candidate=float(scores.candidate.mean()),
            change=change, ci95=ci, months=scores.to_dict('index'), cells=len(group), pass_gate=change < 0 and ci[1] < 0)
    result = dict(identity=art(OUT / 'identity.json'), controls=controls, folds=summary,
        gate=all(s['pass_gate'] for s in summary.values()), source_density_scipy_exact=True,
        predictions=art(OUT / 'scored_source_cells.parquet'), fits=art(OUT / 'fits.json'),
        no_challenge_cv_or_production=True)
    write_json(OUT / 'result.json', result)
    print(json.dumps(dict(gate=result['gate'], folds={k: {n: v[n] for n in ['change', 'ci95', 'cells']} for k, v in summary.items()})))


if __name__ == '__main__':
    with threadpool_limits(1):
        main()
