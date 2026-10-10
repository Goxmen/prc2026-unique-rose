"""A different classifier family for v58's Rome recording mechanism.

Original R20 rows and inputs only. A single fixed CatBoost recipe chooses its
iteration count by log loss on the last allowed training month, then refits all
allowed training rows. Score both the new probabilities and their fixed equal
mean with R20 through the unchanged v58 correction/rule stack. No uploads.
"""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import rome_adsblol_jan_jul as shared
import catboost_jan_jul as cbj

ROOT = shared.ROOT
OUT = shared.ts.RUNS / 'v3_20261010_rome_catboost_jan_jul'
DESIGN = dict(parent='v58', part='Rome recording classifier', new_features=[],
    seed=42, depth=6, learning_rate=.05, l2_leaf_reg=10., max_iterations=4000,
    stopping='last allowed training month log loss; patience 150',
    train_membership='original R20 after_drop', network_share=.71,
    arms=['catboost', 'half_catboost_half_R20'],
    heldout_months=shared.ts.SPLITS, exploratory_reused_development_months=True)


def fit():
    import duckdb
    from catboost import CatBoostClassifier, Pool
    con = duckdb.connect(config={'threads': 2})
    con.execute("SET TimeZone='UTC'")
    raw = con.execute("SELECT MVT_ID_mvt AS mvt_id, MVT_TIME_UTC_mvt FROM read_parquet(?) "
                      "WHERE PHASE_mvt='DEP'", [str(ROOT / 'data/training_2025-*.parquet')]).df()
    con.close()
    dates = pd.to_datetime(raw.set_index('mvt_id').MVT_TIME_UTC_mvt, utc=True).dt.strftime('%Y-%m')
    for month in shared.ts.SPLITS:
        source = shared.OUT / 'inputs' / f'{month}.pkl'
        dest = OUT / 'predictions' / f'{month}_s42.parquet'
        identity = dict(design=DESIGN, source_sha256=shared.sha256(Path(__file__)),
                        input_sha256=shared.sha256(source),
                        helper_sha256={str(Path(module.__file__).relative_to(ROOT)):
                            shared.sha256(Path(module.__file__))
                            for module in [shared, cbj, shared.baseline, shared.hpo, shared.signals]})
        if dest.exists():
            old = shared.ts.read(dest.with_suffix('.json'))
            if any(old.get(k) != v for k, v in identity.items()):
                raise ValueError('Existing prediction has different provenance')
            if shared.sha256(dest) != old['prediction_sha256']:
                raise ValueError('Existing prediction hash changed')
            continue
        d = pd.read_pickle(source)
        names = [c for c in d['A'] if c not in shared.NEW]
        cats = shared.signals._categorical(names)
        X, V = cbj.frame(d['A'][names], cats), cbj.frame(d['B'][names], cats)
        months = dates.loc[d['train_ids']].to_numpy()
        assert set(months) == set(shared.ts.FOLDS[month])
        assert month not in months
        assert not np.intersect1d(d['train_ids'], d['ids']).size
        hold = months == max(shared.ts.FOLDS[month])
        assert hold.any() and (~hold).any()
        params = dict(loss_function='Logloss', eval_metric='Logloss', depth=DESIGN['depth'],
            learning_rate=DESIGN['learning_rate'], l2_leaf_reg=DESIGN['l2_leaf_reg'],
            random_seed=42, task_type='GPU', devices='0', thread_count=2,
            gpu_ram_part=.65, allow_writing_files=False, verbose=500)
        started = time.monotonic()
        shared.log(f'{month}: inner CatBoost on {int((~hold).sum()):,} rows; stopping month {max(months)}')
        inner = CatBoostClassifier(iterations=DESIGN['max_iterations'], **params)
        inner.fit(Pool(X.loc[~hold], d['target'][~hold], cat_features=cats),
            eval_set=Pool(X.loc[hold], d['target'][hold], cat_features=cats),
            early_stopping_rounds=150, use_best_model=True)
        rounds = int(inner.get_best_iteration()) + 1
        assert 1 <= rounds <= DESIGN['max_iterations']
        curve = inner.get_evals_result()
        del inner
        gc.collect()
        shared.log(f'{month}: refit {len(X):,} rows for {rounds} iterations')
        model = CatBoostClassifier(iterations=rounds, **params)
        model.fit(Pool(X, d['target'], cat_features=cats))
        p = np.zeros(len(d['ids']))
        p[d['use']] = model.predict_proba(Pool(V, cat_features=cats))[:, 1]
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        model_path = OUT / 'models' / f'{month}_s42.cbm'
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(model_path))
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_suffix('.tmp')
        pd.DataFrame(dict(mvt_id=d['ids'], p=p)).to_parquet(temp, index=False)
        temp.replace(dest)
        shared.write_json(dest.with_suffix('.json'), dict(**identity, params=params,
            columns=names, categoricals=cats, stop_month=max(months), rounds=rounds,
            inner_curve=curve, training_rows=len(X), positives=int(d['target'].sum()),
            model_sha256=shared.sha256(model_path), prediction_sha256=shared.sha256(dest),
            minutes=(time.monotonic()-started)/60))
        del model, X, V, d
        gc.collect()


def score():
    sc, original, nn, trees = shared.baseline.baseline()
    errors = {name: {} for name in ['control'] + DESIGN['arms']}
    rows = []
    for month in shared.ts.SPLITS:
        f = pd.read_parquet(OUT / 'predictions' / f'{month}_s42.parquet')
        np.testing.assert_array_equal(f.mvt_id, sc.d[month]['ids'])
        probabilities = dict(control=original[month], catboost=f.p.to_numpy(),
                             half_catboost_half_R20=(f.p.to_numpy()+original[month])/2)
        for arm, p in probabilities.items():
            errors[arm][month] = np.mean([sc.se(month, seed, nn=nn[month], p=p,
                **{shared.hpo.MEMBER_OF[part]: trees[part][seed][month] for part in trees})
                for seed in shared.hpo.SEEDS], axis=0)
        rows.append(pd.DataFrame(dict(mvt_id=f.mvt_id, month=month, day=sc.d[month]['day'],
            **{arm: errors[arm][month] for arm in errors})))
    frame = pd.concat(rows, ignore_index=True)
    scores = {name: sc.cv(value) for name, value in errors.items()}
    assert abs(scores['control']['cv']-295.7414415741871) < 1e-7
    comparisons = {name: shared.ts.paired_bootstrap(frame.day.to_numpy(), frame.month.to_numpy(),
        frame.control.to_numpy(), frame[name].to_numpy()) for name in DESIGN['arms']}
    result = dict(design=DESIGN, scores=scores, comparisons=comparisons)
    frame.to_parquet(OUT / 'scored_errors.parquet', index=False)
    shared.write_json(OUT / 'result.json', result)
    shared.log(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['fit', 'score'])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    design_path = OUT / 'design.json'
    if design_path.exists() and shared.ts.read(design_path) != DESIGN:
        raise ValueError('Existing design changed')
    shared.write_json(design_path, DESIGN)
    with threadpool_limits(2):
        if args.stage == 'fit': fit()
        else: score()


if __name__ == '__main__':
    main()
