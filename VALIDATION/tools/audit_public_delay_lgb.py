"""Reload both public-input main trees and score their fixed v75 combination."""
import gc
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'VALIDATION'))
import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import public_delay_lgb_jan_jul as run
import rome_public_delay_cat as cat
from records import write_json


def main():
    table = pd.read_parquet(run.features.OUT / 'features2025.parquet').set_index('mvt_id')
    checks = []
    for month in run.parent.ts.SPLITS:
        d = pd.read_pickle(run.parent.input_cache.input_path(month))
        names, ids = d['baseline'], d['ids'].copy()
        V = d['V'].loc[:, names].copy()
        del d
        gc.collect()
        clocks = run.parent.clocks.load_features(run.parent.feature_path(month, 'held'))
        V = run.parent.clocks.add_features(V, ids, clocks)
        V = pd.concat([V, table.loc[ids, run.features.FEATURES].set_axis(V.index)], axis=1)
        path = run.OUT / 'predictions' / f'{month}_s42.parquet'
        rec = run.parent.ts.read(path.with_suffix('.json'))
        mp = ROOT / rec['model']['path']
        assert rec['model'] == run.features.base.artifact(mp)
        assert rec['prediction'] == run.features.base.artifact(path)
        model = lgb.Booster(model_file=str(mp))
        assert model.feature_name() == list(V) == rec['columns']
        actual = model.predict(V, num_threads=2).astype('float32')
        stored = pd.read_parquet(path)
        np.testing.assert_array_equal(stored.mvt_id, ids)
        np.testing.assert_array_equal(actual, stored.pred)
        checks.append(dict(month=month, rows=len(ids), all_predictions_exact=True))
        del V, clocks, model, actual
        gc.collect()
    with duckdb.connect(config={'threads': 1}) as con:
        metrics = con.execute('''select month, sqrt(avg(pow(pred-y,2))) pred,
            sqrt(avg(pow(control-y,2))) control, sqrt(avg(pow(transferred-y,2))) transferred,
            sqrt(avg(pow(v74-y,2))) v74 from read_parquet(?) group by month order by month''',
            [str(run.OUT / 'scored.parquet')]).fetchdf()
    stored_scores = run.parent.ts.read(run.OUT / 'result.json')['scores']
    for row in metrics.to_dict('records'):
        for column in ['pred', 'control', 'transferred', 'v74']:
            assert abs(row[column]-stored_scores[row['month']][column]) < 1e-8

    ph = run.selected.physical
    replay = ph.replay
    sc, rome, _, trees = replay.baseline.baseline()
    old = pd.read_parquet(cat.OUT / 'scored.parquet').set_index('mvt_id')
    raw = pd.read_parquet(run.features.base.OUT / 'rows2025.parquet').set_index('mvt_id')
    matches = pd.read_parquet(run.selected.constraints.exclusion.OUT / 'matches2025.parquet')
    pool_id = run.parent.ts.read(ph.production.cv.MANIFEST)
    frames = []
    for month in run.parent.ts.SPLITS:
        d = sc.d[month]
        ids = d['ids']
        f = old.loc[ids]
        calls = replay.values(replay.combo.sequence.CALLS_PRED, ids, 'candidate_nn')
        physical = np.mean([replay.values(ph.verify(ph.pool.completed(month, k, pool_id)['prediction']), ids)
                            for k in [0, 1, 2]], axis=0)
        old_main = replay.values(replay.combo.parent.control.candidate.OUT / 'predictions' / f'{month}_s42.parquet', ids, 'pred')
        new_main = replay.values(run.OUT / 'predictions' / f'{month}_s42.parquet', ids, 'pred')
        p = .5*f.newcat.to_numpy()+.5*rome[month]
        def final(main):
            ordinary = .71*.5*(calls+physical)+.29*np.mean([main, trees['airport'][42][month], trees['catboost'][42][month]], axis=0)
            return sc.final(month, ordinary, p)
        np.testing.assert_array_equal(final(old_main), f.cat_only)
        candidate = final(new_main)
        q = run.selected.constraints.matched_bound(raw.loc[ids].reset_index(), f.v74.to_numpy()+candidate-f.control.to_numpy())
        permitted = matches[matches.donor_month.isin(run.features.base.FOLDS[month]+[month])]
        q, _ = run.selected.constraints.exclusion.apply(ids, np.maximum(q, 0), permitted)
        frames.append(pd.DataFrame(dict(mvt_id=ids, month=month, day=d['day'], y=d['y'],
            control=f.cat_only.to_numpy(), candidate=candidate, v75=f.cat_only_transfer.to_numpy(), transferred=q)))
    f = pd.concat(frames, ignore_index=True)
    path = run.OUT / 'combined_public_cat.parquet'
    f.to_parquet(path, index=False)
    columns = ['control', 'candidate', 'v75', 'transferred']
    scores = {m: {c: float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in columns} for m, g in f.groupby('month')}
    pairs = {f'{b}_vs_{a}': run.parent.ts.paired_bootstrap(f.day, f.month, (f[a]-f.y)**2, (f[b]-f.y)**2)
             for a, b in [('control', 'candidate'), ('v75', 'transferred')]}
    result = dict(status='pass', model_reload=checks, independent_sql_scores=True,
        feature_audit=run.features.base.artifact(run.features.OUT / 'audit.json'),
        combined=dict(scores=scores, cv={c: float(np.mean([v[c] for v in scores.values()])) for c in columns}, paired=pairs),
        predictions=run.features.base.artifact(path), source=run.features.base.artifact(__file__))
    write_json(run.OUT / 'audit.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    with threadpool_limits(2):
        main()
