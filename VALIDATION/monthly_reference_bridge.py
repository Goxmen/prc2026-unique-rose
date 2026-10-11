"""Fixed monthly calibration using an approximate reference-flight population.

Public taxi-out means exclude flights lacking a stand/runway reference. Rank
cells by their frequency in allowed input months and match the public covered
fraction, splitting frequency ties fractionally. This is an approximation,
not recovery of the official reference mask. Keep the original half-strength
shift, invalid-share gate, cap and training-only composition regression.
"""
import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

import monthly_taxi_reconcile as parent
import public_constraints_candidate as constraints
from records import write_json

base = parent.base
ROOT = base.ROOT
OUT = ROOT / 'VALIDATION/runs/v3_20261011_monthly_reference_bridge'
V74 = constraints.OUT / 'submission_v74/cv.parquet'
V75 = ROOT / 'VALIDATION/runs/v3_20261011_rome_public_delay_cat/scored.parquet'
KEYS = ['airport', 'stand', 'runway']
DESIGN = dict(parent='v74 monthly calibration; primary transfer on prepared v75',
    change='calibrate the predicted reference-population mean instead of all eligible departures',
    frequency='stand/runway counts in allowed training plus held input months; no targets',
    reference_mass='TOTAL_REF_NB_FL / VALID_FL of the local eligible population',
    ordering='most frequent cells first; equal frequency receives equal fractional weight',
    composition=parent.DESIGN, no_threshold_or_weight_search=True, no_new_taxi_fit=True,
    limitation='Reference availability only approximated; deicing and some aircraft exclusions remain unknown',
    evaluation='reused Jan/Jul development; source audit motivated this experiment')


def read(path):
    return json.loads(Path(path).read_text())


def weights(frequency, fraction):
    """Select the requested mass without arbitrary ordering inside ties."""
    frequency = np.asarray(frequency, float)
    assert frequency.ndim == 1 and len(frequency) and np.isfinite(frequency).all()
    assert np.all(frequency >= 0) and 0 < fraction <= 1
    levels, counts = np.unique(frequency, return_counts=True)
    levels, counts = levels[::-1], counts[::-1]
    before = np.r_[0, np.cumsum(counts)[:-1]]
    per_level = np.clip((len(frequency)*fraction-before)/counts, 0, 1)
    result = pd.Series(frequency).map(dict(zip(levels, per_level))).to_numpy(float)
    np.testing.assert_allclose(result.sum(), len(frequency)*fraction, rtol=0, atol=1e-8)
    return result


def checks():
    f = np.array([10, 10, 3, 1, 1], float)
    np.testing.assert_array_equal(weights(f, .5), [1, 1, .5, 0, 0])
    np.testing.assert_allclose(weights(f, .7), [1, 1, 1, .25, .25])
    np.testing.assert_array_equal(weights(f, 1), np.ones(5))
    np.testing.assert_array_equal(weights(np.ones(4), .5), np.full(4, .5))
    order = [4, 2, 0, 3, 1]
    np.testing.assert_array_equal(weights(f[order], .7), weights(f, .7)[order])
    return dict(hand_mass=True, fractional_ties=True, uniform_ties=True,
        full_coverage_identity=True, permutation_invariant=True)


def frequencies(rows):
    # Whitelist prevents labels and prediction errors from entering cell counts.
    r = rows[KEYS]
    return r.groupby(KEYS, observed=True, dropna=False).size().rename('frequency').reset_index()


def cells(rows, values, source, frequency):
    f = rows[['mvt_id', 'airport', 'local_month', 'eligible', 'stand', 'runway']].copy()
    f['value'] = np.asarray(values, float)
    f = f[f.eligible & f.value.between(0, 7200)].copy()
    f = f.merge(frequency, on=KEYS, how='left', validate='many_to_one')
    assert f.frequency.notna().all()
    indexed = source.set_index(['airport', 'local_month'])
    records = []
    for (airport, month), g in f.groupby(['airport', 'local_month'], sort=True):
        if (airport, month) not in indexed.index:
            continue
        s = indexed.loc[(airport, month)]
        ratio = float(s.TOTAL_REF_NB_FL/s.VALID_FL)
        assert 0 < ratio <= 1
        w = weights(g.frequency.to_numpy(), ratio)
        mean = float(w@g.value.to_numpy()/w.sum())
        records.append(dict(airport=airport, local_month=month, reference_mean=mean,
            n=len(g), effective_mass=float(w.sum()), source_fraction=ratio,
            **s.to_dict()))
    return pd.DataFrame(records)


def fit(rows, source, frequency):
    c = cells(rows, rows.y.to_numpy(float), source, frequency)
    c = c[(c.n >= 100) & (c.invalid < .05)].copy()
    scale = np.r_[np.ones(10), .05, .1]
    A = parent.matrix(c)/scale
    y = (c.reference_mean-c.mean_sec).to_numpy(float)
    fun = lambda b: np.r_[(A@b-y)/20., np.sqrt(.001)*b/20.]
    result = least_squares(fun, np.zeros(12), loss='huber', f_scale=1.)
    assert result.success
    return dict(coefficients=(result.x/scale).tolist(), cells=c.to_dict('records'))


def apply(rows, prediction, source, frequency, parameters):
    c = cells(rows, prediction, source, frequency)
    c['composition'] = parent.matrix(c)@np.asarray(parameters['coefficients'])
    c['raw_shift'] = c.mean_sec+c.composition-c.reference_mean
    c['shift'] = np.where((c.invalid < .05) & (c.n >= 100),
        .5*c.raw_shift.clip(-120, 120), 0.)
    c['shift'] = c['shift'].fillna(0.)
    shifts = rows[['airport', 'local_month']].merge(c[['airport', 'local_month', 'shift']],
        on=['airport', 'local_month'], how='left', validate='many_to_one')['shift'].fillna(0).to_numpy()
    eligible = rows.eligible.to_numpy() & (prediction <= 7200)
    return np.maximum(0, prediction+np.where(eligible, shifts, 0)), c


def guard():
    paths = sorted((ROOT / 'data').glob('training_2025-*.parquet'))
    inputs = [Path(__file__), Path(parent.__file__), Path(constraints.__file__),
        Path(constraints.exclusion.__file__), parent.OUT / 'rows2025.parquet',
        parent.OUT / 'result.json', parent.OUT / 'predictions.parquet',
        parent.PARENT / 'cv.parquet', base.OUT / 'rows2025.parquet',
        constraints.exclusion.OUT / 'matches2025.parquet', V74, V75,
        base.SOURCE / 'taxi_out_additional_time_2025.csv', *paths]
    value = dict(design=DESIGN, sources=[base.artifact(p) for p in inputs], checks=checks())
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'identity.json'
    if path.exists():
        assert read(path) == value
    else:
        assert not list(OUT.glob('*.parquet'))
        write_json(path, value)
    return paths


def screen():
    paths = guard()
    assert not (OUT / 'result.json').exists(), 'Preserve completed screen'
    rows = pd.read_parquet(parent.OUT / 'rows2025.parquet')
    with duckdb.connect(config={'threads': 1}) as con:
        extra = con.execute('''select MVT_ID_mvt mvt_id, STAND_mvt stand, RUNWAY_mvt runway
            from read_parquet(?) where PHASE_mvt='DEP' ''', [list(map(str, paths))]).fetchdf()
    rows = rows.merge(extra, on='mvt_id', validate='one_to_one')
    source = parent.public(2025)
    cv = pd.read_parquet(V74)
    v75 = pd.read_parquet(V75).set_index('mvt_id').loc[cv.mvt_id].reset_index()
    incoming = pd.read_parquet(parent.PARENT / 'cv.parquet').set_index('mvt_id').loc[cv.mvt_id, 'pred'].to_numpy(float)
    np.testing.assert_array_equal(incoming, cv.parent)
    np.testing.assert_array_equal(v75.v74, cv.pred)
    np.testing.assert_array_equal(v75.y, cv.y)
    mrows = rows.set_index('mvt_id').loc[cv.mvt_id].reset_index()
    brows = pd.read_parquet(base.OUT / 'rows2025.parquet').set_index('mvt_id').loc[cv.mvt_id].reset_index()
    matches = pd.read_parquet(constraints.exclusion.OUT / 'matches2025.parquet')
    new = incoming.copy()
    parameters, logs = {}, []
    for month, allowed in base.FOLDS.items():
        ix = np.flatnonzero(cv.month.eq(month))
        train = rows[rows.utc_month.isin(allowed)].copy()
        context = rows[rows.utc_month.isin(allowed+[month])]
        freq = frequencies(context)
        # Labels cannot influence frequency weights.
        pd.testing.assert_frame_equal(freq, frequencies(context.assign(y=1e12)))
        original = parent.fit(train, source)
        replay, _ = parent.apply(mrows.iloc[ix].reset_index(drop=True), incoming[ix], source, original)
        np.testing.assert_array_equal(replay, cv.monthly.iloc[ix])
        replay = constraints.matched_bound(brows.iloc[ix], replay)
        permitted = matches[matches.donor_month.isin(allowed+[month])]
        replay, _ = constraints.exclusion.apply(cv.mvt_id.iloc[ix].to_numpy(), replay, permitted)
        np.testing.assert_array_equal(replay, cv.pred.iloc[ix])
        parameters[month] = fit(train, source, freq)
        new[ix], log = apply(mrows.iloc[ix].reset_index(drop=True), incoming[ix], source, freq, parameters[month])
        logs.append(log.assign(fold=month))
        new[ix] = constraints.matched_bound(brows.iloc[ix], new[ix])
        new[ix], _ = constraints.exclusion.apply(cv.mvt_id.iloc[ix].to_numpy(), new[ix], permitted)
    transferred = constraints.matched_bound(brows, v75.cat_only_transfer.to_numpy()+new-cv.pred.to_numpy())
    for month, allowed in base.FOLDS.items():
        ix = np.flatnonzero(cv.month.eq(month))
        permitted = matches[matches.donor_month.isin(allowed+[month])]
        transferred[ix], _ = constraints.exclusion.apply(cv.mvt_id.iloc[ix].to_numpy(), transferred[ix], permitted)
    f = cv[['mvt_id', 'month', 'day', 'airport', 'y']].copy()
    f['control'] = cv.pred
    f['pred'] = new
    f['v75'] = v75.cat_only_transfer
    f['transferred'] = transferred
    assert len(f) == 344419 and f.mvt_id.is_unique
    columns = ['control', 'pred', 'v75', 'transferred']
    assert np.isfinite(f[columns]).all().all() and f[columns].ge(0).all().all()
    scores = {m: {c: float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in columns} for m, g in f.groupby('month')}
    import tuning_seasonal as ts
    paired = {f'{b}_vs_{a}': ts.paired_bootstrap(f.day, f.month, (f[a]-f.y)**2, (f[b]-f.y)**2)
        for a, b in [('control', 'pred'), ('v75', 'transferred')]}
    f.to_parquet(OUT / 'scored.parquet', index=False)
    pd.concat(logs).to_parquet(OUT / 'cells.parquet', index=False)
    value = dict(design=DESIGN, scores=scores, cv={c: float(np.mean([s[c] for s in scores.values()])) for c in columns},
        paired=paired, parameters=parameters, original_v74_replay_exact=True,
        identity=base.artifact(OUT / 'identity.json'), predictions=base.artifact(OUT / 'scored.parquet'),
        interpretation='Approximate reference coverage, no exact source-identity claim; no production or upload selected')
    write_json(OUT / 'result.json', value)
    print(dict(scores=scores, cv=value['cv'], paired=paired), flush=True)


def audit():
    guard()
    assert not (OUT / 'audit.json').exists(), 'Preserve completed audit'
    result = read(OUT / 'result.json')
    assert result['identity'] == base.artifact(OUT / 'identity.json')
    assert result['predictions'] == base.artifact(OUT / 'scored.parquet')
    with duckdb.connect(config={'threads': 1}) as con:
        for name, path in [('candidate', OUT / 'scored.parquet'), ('truth', base.OUT / 'rows2025.parquet'),
                           ('v74', V74), ('v75', V75)]:
            con.read_parquet(str(path)).create_view(name)
        n, distinct, wrong = con.execute('''select count(*),count(distinct c.mvt_id),
            sum(case when c.y is distinct from t.y or c.day is distinct from t.day or c.month is distinct from t.month
            or c.control is distinct from a.pred or c.v75 is distinct from b.cat_only_transfer then 1 else 0 end)
            from candidate c join truth t using(mvt_id) join v74 a using(mvt_id) join v75 b using(mvt_id)''').fetchone()
        assert n == distinct == 344419 and wrong == 0
        for name in result['cv']:
            metrics = con.execute(f'select month,sqrt(avg(pow({name}-y,2))) value from candidate group by month').fetchall()
            for month, value in metrics:
                np.testing.assert_allclose(value, result['scores'][month][name], atol=1e-8, rtol=0)
            np.testing.assert_allclose(np.mean([v for m, v in metrics]), result['cv'][name], atol=1e-8, rtol=0)
    cells_frame = pd.read_parquet(OUT / 'cells.parquet')
    np.testing.assert_allclose(cells_frame.effective_mass, cells_frame.n*cells_frame.source_fraction, atol=1e-8, rtol=0)
    value = dict(status='pass', rows=n, truth_calendar_controls_exact=True, independent_sql_scores=True,
        mass_exact=True, result=base.artifact(OUT / 'result.json'), source=base.artifact(__file__))
    write_json(OUT / 'audit.json', value)
    print(value, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['guard', 'screen', 'audit'])
    args = parser.parse_args()
    with threadpool_limits(1):
        globals()[args.stage]()
