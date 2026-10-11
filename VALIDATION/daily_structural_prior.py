"""Use retained Rome conditional predictions in the public daily posterior.

The old posterior infers a copy share from an airport-wide ordinary location.
This screen instead derives the two conditional endpoints from the existing
v70 correction function. The incoming serialized mean and all likelihood,
grid, noise, spread and non-Rome predictions remain unchanged. Public-input
classifiers are deliberately absent from the prior.
"""
import argparse
import gc
import types
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import daily_delay_posterior as old
import rome_public_delay_cat as retained
from records import write_json

base = old.base
ROOT = base.ROOT
OUT = ROOT / 'VALIDATION/runs/v3_20261011_daily_structural_prior'
DESIGN = dict(parent='serialized v70', matched_control='retained full daily posterior on serialized v70',
    change='Rome long-flight conditional support from saved ordinary/copy endpoints',
    fixed_copy='equal endpoints at schedule gap use a point mass at the unchanged incoming prediction',
    long_threshold=3600, copy_tolerance=60, denominator_floor=1e-8,
    likelihood=old.DESIGN, population='same all-departure population as matched retained posterior',
    public_in_prior=False, no_new_fit_or_hyperparameter_search=True,
    non_rome='exact retained daily posterior',
    held_labels='final reused development scoring only')


def support(d, p, nm, nm_gap, airport, prm, ordinary, copy):
    """Exact-mean support; fall back when endpoints do not identify a copy mode."""
    eligible = (airport == 'LIRF' and d > 3600 and np.isfinite(ordinary+copy)
        and abs(copy-d) <= 60 and ordinary <= p+1e-8 and p <= copy+1e-8)
    if not eligible:
        return old.support(d, p, nm, nm_gap, airport, prm), 'original'
    if abs(copy-ordinary) < 1e-8:
        return (np.array([p]), np.array([1.]),
            np.array([int(np.rint(max(0, d-p)/old.DESIGN['grid_sec']))])), 'fixed_copy'
    probability = np.clip((p-ordinary)/(copy-ordinary), 0, 1)
    held = np.maximum(0, ordinary+np.asarray(prm['spread']))
    broad = np.linspace(30, max(d, 30), 9)
    values = np.r_[held, copy, broad]
    weights = np.r_[np.full(len(held), (1-probability)*.95/len(held)),
        probability*.95, np.full(len(broad), .05/len(broad))]
    if nm and np.isfinite(nm_gap):
        lo, hi = max(0, nm_gap-3606), nm_gap+3606
        if lo <= p <= hi:
            keep = (values >= lo) & (values <= hi)
            if keep.any():
                values, weights = values[keep], weights[keep]
    values, weights = np.r_[values, p], np.r_[weights, 1e-8]
    weights = old.tilt(values, weights, p)
    np.testing.assert_allclose(values@weights, p, atol=1e-5, rtol=1e-8)
    contributions = np.rint(np.maximum(0, d-values)/old.DESIGN['grid_sec']).astype(int)
    return (values, weights, contributions), 'conditional_copy'


def condition(d, p, nm, nm_gap, airport, prm, total, dn, ordinary, copy):
    rows = np.flatnonzero(d > 3600)
    prepared = [support(d[i], p[i], nm[i], nm_gap[i], airport, prm, ordinary[i], copy[i]) for i in rows]
    iterator = iter([item[0] for item in prepared])
    # Reuse the unchanged, independently enumerated message algorithm.
    def supplied_support(*args):
        return next(iterator)
    fn = types.FunctionType(old.condition_day.__code__,
        {**old.condition_day.__globals__, 'support': supplied_support},
        'structural_condition_day', old.condition_day.__defaults__, old.condition_day.__closure__)
    prediction, info = fn(d, p, nm, nm_gap, airport, prm, total, dn)
    assert next(iterator, None) is None
    modes = [item[1] for item in prepared]
    for mode in ['original', 'fixed_copy', 'conditional_copy']:
        info[mode] = modes.count(mode)
    return prediction, info


def checks():
    prm = dict(spread=[-300., -100., 0., 100., 300.], tau=1200., sigma=120., kappa=600., bias=0.)
    maximum = 0.
    for q in [.1, .5, .9]:
        target = (1-q)*800+q*7000
        (values, weights, contributions), mode = support(7000, target, False, np.nan, 'LIRF', prm, 800, 7000)
        assert mode == 'conditional_copy' and (weights >= 0).all()
        np.testing.assert_allclose(weights.sum(), 1, atol=1e-12)
        maximum = max(maximum, abs(values@weights-target))
    fixed, mode = support(18000, 18000, False, np.nan, 'LIRF', prm, 18000, 18000)
    assert mode == 'fixed_copy'
    np.testing.assert_array_equal(fixed[0], [18000])
    for args in [(7000, 1500, False, np.nan, 'EDDF', prm, 800, 7000),
                 (7000, 1500, False, np.nan, 'LIRF', prm, 800, 4000)]:
        actual, mode = support(*args)
        assert mode == 'original'
        for a, b in zip(actual, old.support(*args[:6])):
            np.testing.assert_array_equal(a, b)
    # A point-mass copy stays unchanged even with strongly conflicting totals.
    for total in [0., 36000.]:
        result, info = condition(np.array([18000.]), np.array([18000.]), np.array([False]),
            np.array([np.nan]), 'LIRF', prm, total, 0., np.array([18000.]), np.array([18000.]))
        np.testing.assert_array_equal(result, [18000.])
        assert info['fixed_copy'] == 1
    return dict(exact_prior_mean_max_error=float(maximum), fallback_exact=True,
        fixed_copy_unchanged=True, message_bytecode_unchanged=True)


def guard():
    OUT.mkdir(parents=True, exist_ok=True)
    value = dict(design=DESIGN, source=base.artifact(__file__),
        helpers=[base.artifact(m.__file__) for m in [old, base, retained, retained.r20.physical]],
        old_result=base.artifact(old.OUT / 'v70/result.json'),
        old_predictions=base.artifact(old.OUT / 'v70/predictions.parquet'),
        parameters=base.artifact(old.OUT / 'screen.json'),
        prior_predictions=base.artifact(retained.r20.physical.OUT / 'pool3/cv/predictions.parquet'),
        prior_identity=base.artifact(retained.r20.physical.production.cv.MANIFEST),
        raw_rows=base.artifact(base.OUT / 'rows2025.parquet'),
        public=base.artifact(base.SOURCE / 'all_pre_departure_delays_2025.csv'), checks=checks())
    path = OUT / 'identity.json'
    if path.exists():
        assert retained.parent.shared.ts.read(path) == value
    else:
        write_json(path, value)


def prepare():
    path = OUT / 'endpoints.parquet'
    record = path.with_suffix('.json')
    if record.exists():
        rec = retained.parent.shared.ts.read(record)
        assert rec['identity'] == base.artifact(OUT / 'identity.json') and rec['prediction'] == base.artifact(path)
        return
    assert not path.exists(), 'Unrecorded endpoints'
    ph = retained.r20.physical
    replay = ph.replay
    sc, rome, _, trees = replay.baseline.baseline()
    identity = retained.parent.shared.ts.read(ph.production.cv.MANIFEST)
    saved = pd.read_parquet(old.OUT / 'v70/predictions.parquet').set_index('mvt_id')
    frames = []
    for month in base.FOLDS:
        d = sc.d[month]
        ids = d['ids']
        calls = replay.values(replay.combo.sequence.CALLS_PRED, ids, 'candidate_nn')
        physical = np.mean([replay.values(ph.verify(ph.pool.completed(month, k, identity)['prediction']), ids)
            for k in [0, 1, 2]], axis=0)
        main = replay.values(replay.combo.parent.control.candidate.OUT / 'predictions' / f'{month}_s42.parquet', ids, 'pred')
        ordinary = .71*.5*(calls+physical)+.29*np.mean([main, trees['airport'][42][month], trees['catboost'][42][month]], axis=0)
        probability = .5*retained.parent.probabilities(month, ids)+.5*rome[month]
        pred = sc.final(month, ordinary, probability)
        np.testing.assert_array_equal(np.rint(pred), saved.loc[ids, 'parent'])
        lo = sc.final(month, ordinary, np.zeros(len(ids)))
        hi = sc.final(month, ordinary, np.ones(len(ids)))
        frames.append(pd.DataFrame(dict(mvt_id=ids, ordinary=lo, copy=hi, p=probability,
            prior_float=pred, prior=np.rint(pred))))
    result = pd.concat(frames, ignore_index=True)
    assert result.mvt_id.is_unique and len(result) == 344419
    result.to_parquet(path, index=False)
    write_json(record, dict(identity=base.artifact(OUT / 'identity.json'), prediction=base.artifact(path),
        all_prior_serialized_predictions_exact=True, rows=len(result)))
    del sc
    gc.collect()


def screen():
    assert not (OUT / 'result.json').exists(), 'Completed screen retained'
    prepare()
    endpoint = pd.read_parquet(OUT / 'endpoints.parquet').set_index('mvt_id')
    f = pd.read_parquet(old.OUT / 'v70/predictions.parquet')
    raw = pd.read_parquet(base.OUT / 'rows2025.parquet').set_index('mvt_id').loc[f.mvt_id]
    np.testing.assert_array_equal(raw.y, f.y)
    prm = retained.parent.shared.ts.read(old.OUT / 'screen.json')['parameters']
    public = base.public(2025).set_index(['airport', 'day'])
    candidate = f.pred.to_numpy(float).copy()
    logs = []
    for (month, day), group in f[f.airport.eq('LIRF')].groupby(['month', 'day']):
        positions = group.index.to_numpy()
        rows = raw.iloc[positions]
        e = endpoint.loc[group.mvt_id]
        source = public.loc[('LIRF', day)]
        candidate[positions], info = condition(rows.gap.to_numpy(float), group.parent.to_numpy(float),
            rows.has_id.to_numpy(bool), rows.nm_gap.to_numpy(float), 'LIRF', prm[month]['LIRF'],
            float(source.total), float(source.n_public-len(group)), e.ordinary.to_numpy(float), e['copy'].to_numpy(float))
        logs.append(dict(month=month, day=day, rows=len(group), **info))
    f = f.rename(columns={'pred': 'old_posterior'})
    f['pred'] = candidate
    assert np.array_equal(f.loc[f.airport.ne('LIRF'), 'pred'], f.loc[f.airport.ne('LIRF'), 'old_posterior'])
    assert np.isfinite(f.pred).all() and f.pred.ge(0).all()
    f.to_parquet(OUT / 'scored.parquet', index=False)
    pd.DataFrame(logs).to_parquet(OUT / 'days.parquet', index=False)
    cols = ['parent', 'old_posterior', 'pred']
    scores = {m: {c: float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in cols} for m, g in f.groupby('month')}
    paired = {f'pred_vs_{c}': retained.parent.shared.ts.paired_bootstrap(f.day, f.month,
        (f[c]-f.y)**2, (f.pred-f.y)**2) for c in ['parent', 'old_posterior']}
    result = dict(design=DESIGN, scores=scores, cv={c: float(np.mean([v[c] for v in scores.values()])) for c in cols},
        paired=paired, changed_rows=int((f.pred != f.old_posterior).sum()),
        mode_counts={c: int(pd.DataFrame(logs)[c].sum()) for c in ['original', 'fixed_copy', 'conditional_copy']},
        identity=base.artifact(OUT / 'identity.json'), only_rome_changes=True,
        interpretation='Isolated prior-shape comparison on original v70; no v74/v75 transfer or production selected')
    write_json(OUT / 'result.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['screen'])
    args = parser.parse_args()
    with threadpool_limits(2):
        guard()
        globals()[args.stage]()
