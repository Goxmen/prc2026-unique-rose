"""Fixed joint CV for the audited public main tree and public neural pool.

Reconstruct v75 and both previously scored single changes exactly. Combine
the new ordinary components before the original correction stack, then apply
the same fixed v74 transfer and plan bounds. No fits, weights or stopping search.
"""
import argparse
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import nn_public_pool3 as neural
import public_delay_lgb_jan_jul as main
from records import write_json, sha256

ROOT = neural.ROOT
OUT = ROOT / 'VALIDATION/runs/v3_20261011_public_stack_cv'
MANIFEST = OUT / 'identity.json'
art, verify, read = neural.single.artifact, neural.single.verify, neural.single.shared.ts.read
DESIGN = dict(parent='v75', change='public main LightGBM and public physical neural pool together',
    tree='equal main/airport/CatBoost', neural='equal callsign15/physical3', neural_weight=.71,
    rome='public Cat3 plus original R20, equal mixture',
    transfer='v74 plus new primary minus original v70; restore matched/recovered bounds',
    no_new_fit=True, no_weight_or_parameter_search=True, evaluation='reused Jan/Jul development',
    comparisons=['combined vs public-main alone', 'combined vs public-neural alone', 'combined vs v75'])


def guard():
    single_identity = neural.guard()
    main.guard()
    for path in [neural.OUT / 'audit.json', neural.OUT / 'score_audit.json', main.OUT / 'audit.json']:
        assert read(path)['status'] == 'pass'
    records = []
    old_identity = read(neural.control.single.MANIFEST)
    for month in neural.single.STOP:
        for k in range(3):
            for rec in [neural.completed(month, k, single_identity),
                        neural.control.completed(month, k, old_identity)]:
                assert rec is not None
                records.extend([rec['model'], rec['prediction'], rec['preprocessing']])
        path = main.OUT / 'predictions' / f'{month}_s42.json'
        rec = read(path)
        assert rec['identity'] == art(main.OUT / 'identity.json')
        records.extend([art(path), rec['model'], rec['prediction']])
    for record in records:
        verify(record)
    paths = [Path(__file__), Path(neural.__file__), Path(main.__file__), Path(neural.cat.__file__),
        Path(main.selected.constraints.__file__), Path(main.selected.constraints.exclusion.__file__),
        Path(main.selected.physical.replay.__file__), Path(main.selected.physical.replay.baseline.__file__),
        neural.MANIFEST, main.OUT / 'identity.json', neural.OUT / 'audit.json',
        neural.OUT / 'score_audit.json', neural.OUT / 'result.json', neural.OUT / 'scored.parquet',
        main.OUT / 'audit.json', main.OUT / 'combined_public_cat.parquet',
        neural.cat.OUT / 'scored.parquet', main.features.base.OUT / 'rows2025.parquet',
        main.selected.constraints.exclusion.OUT / 'matches2025.parquet']
    value = dict(design=DESIGN, sources={str(p.relative_to(ROOT)): sha256(p) for p in paths}, model_artifacts=records)
    OUT.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists():
        assert read(MANIFEST) == value
    else:
        assert not list(OUT.glob('*.parquet'))
        write_json(MANIFEST, value)
    return single_identity, old_identity


def score():
    assert not (OUT / 'result.json').exists(), 'Preserve completed joint score'
    identity, old_identity = guard()
    r20 = neural.cat.r20
    replay = r20.physical.replay
    sc, rome, _, trees = replay.baseline.baseline()
    prior = pd.read_parquet(neural.cat.OUT / 'scored.parquet').set_index('mvt_id')
    main_saved = pd.read_parquet(main.OUT / 'combined_public_cat.parquet').set_index('mvt_id')
    neural_saved = pd.read_parquet(neural.OUT / 'scored.parquet').set_index('mvt_id')
    raw = pd.read_parquet(main.features.base.OUT / 'rows2025.parquet').set_index('mvt_id')
    matches = pd.read_parquet(main.selected.constraints.exclusion.OUT / 'matches2025.parquet')
    frames = []
    for month in neural.single.STOP:
        d = sc.d[month]
        ids = d['ids']
        f = prior.loc[ids]
        pools = {}
        for name, runner, expected in [('old', neural.control, old_identity), ('new', neural, identity)]:
            values = []
            for k in range(3):
                rec = runner.completed(month, k, expected)
                p = pd.read_parquet(verify(rec['prediction']))
                np.testing.assert_array_equal(p.mvt_id, ids)
                values.append(p.base.to_numpy(float))
            pools[name] = np.mean(values, axis=0)
        calls = replay.values(replay.combo.sequence.CALLS_PRED, ids, 'candidate_nn')
        old_main = replay.values(replay.combo.parent.control.candidate.OUT / 'predictions' / f'{month}_s42.parquet', ids, 'pred')
        new_main = replay.values(main.OUT / 'predictions' / f'{month}_s42.parquet', ids, 'pred')
        old_tree = np.mean([old_main, trees['airport'][42][month], trees['catboost'][42][month]], axis=0)
        new_tree = np.mean([new_main, trees['airport'][42][month], trees['catboost'][42][month]], axis=0)
        p = .5*f.newcat.to_numpy()+.5*rome[month]
        permitted = matches[matches.donor_month.isin(main.features.base.FOLDS[month]+[month])]
        predictions = {}
        for name, pool, tree in [('baseline', pools['old'], old_tree), ('main', pools['old'], new_tree),
                                 ('neural', pools['new'], old_tree), ('combined', pools['new'], new_tree)]:
            ordinary = .71*.5*(calls+pool)+.29*tree
            final = sc.final(month, ordinary, p)
            transferred = r20.constraints.matched_bound(raw.loc[ids].reset_index(),
                f.v74.to_numpy()+final-f.control.to_numpy())
            transferred, _ = r20.constraints.exclusion.apply(ids, np.maximum(transferred, 0), permitted)
            predictions[name] = final
            predictions[name+'_transfer'] = transferred
        np.testing.assert_array_equal(predictions['baseline'], f.cat_only)
        np.testing.assert_array_equal(predictions['baseline_transfer'], f.cat_only_transfer)
        np.testing.assert_array_equal(predictions['main'], main_saved.loc[ids, 'candidate'])
        np.testing.assert_array_equal(predictions['main_transfer'], main_saved.loc[ids, 'transferred'])
        np.testing.assert_array_equal(predictions['neural'], neural_saved.loc[ids, 'pred'])
        np.testing.assert_array_equal(predictions['neural_transfer'], neural_saved.loc[ids, 'transferred'])
        frames.append(pd.DataFrame(dict(mvt_id=ids, month=month, day=d['day'], y=d['y'], **predictions)))
    f = pd.concat(frames, ignore_index=True)
    assert len(f) == 344419 and f.mvt_id.is_unique
    columns = list(predictions)
    assert np.isfinite(f[columns]).all().all() and f[columns].ge(0).all().all()
    scores = {m: {c: float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in columns} for m, g in f.groupby('month')}
    pairs = [(a+suffix, 'combined'+suffix) for suffix in ['', '_transfer'] for a in ['baseline', 'main', 'neural']]
    paired = {f'{b}_vs_{a}': neural.single.shared.ts.paired_bootstrap(f.day, f.month, (f[a]-f.y)**2, (f[b]-f.y)**2)
        for a, b in pairs}
    f.to_parquet(OUT / 'scored.parquet', index=False)
    result = dict(design=DESIGN, scores=scores, cv={c: float(np.mean([s[c] for s in scores.values()])) for c in columns},
        paired=paired, identity=art(MANIFEST), predictions=art(OUT / 'scored.parquet'),
        all_three_control_replays_exact=True,
        selection='Joint recipe is selectable only if it improves both individual transferred recipes; no upload here')
    write_json(OUT / 'result.json', result)
    print(result, flush=True)


def audit():
    assert not (OUT / 'audit.json').exists(), 'Preserve completed audit'
    guard()
    result = read(OUT / 'result.json')
    assert result['identity'] == art(MANIFEST) and result['predictions'] == art(OUT / 'scored.parquet')
    with duckdb.connect(config={'threads': 1}) as con:
        for name, path in [('candidate', OUT / 'scored.parquet'), ('truth', main.features.base.OUT / 'rows2025.parquet'),
                           ('rome', neural.cat.OUT / 'scored.parquet'), ('main', main.OUT / 'combined_public_cat.parquet'),
                           ('neural', neural.OUT / 'scored.parquet')]:
            con.read_parquet(str(path)).create_view(name)
        count, distinct_ids = con.execute('select count(*),count(distinct mvt_id) from candidate').fetchone()
        assert count == distinct_ids == 344419
        matched, errors = con.execute('''select count(*),sum(case when c.y is distinct from t.y or
            c.day is distinct from t.day or c.month is distinct from t.month or
            c.baseline is distinct from r.cat_only or c.baseline_transfer is distinct from r.cat_only_transfer or
            c.main is distinct from m.candidate or c.main_transfer is distinct from m.transferred or
            c.neural is distinct from n.pred or c.neural_transfer is distinct from n.transferred then 1 else 0 end)
            from candidate c join truth t using(mvt_id) join rome r using(mvt_id)
            join main m using(mvt_id) join neural n using(mvt_id)''').fetchone()
        assert matched == count and errors == 0
        columns = list(result['cv'])
        expressions = ','.join(f'sqrt(avg(pow("{c}"-y,2))) "{c}"' for c in columns)
        metrics = con.execute('select month,'+expressions+' from candidate group by month order by month').fetchdf()
        assert metrics.month.tolist() == ['2025-01', '2025-07']
        for _, row in metrics.iterrows():
            for c in columns:
                np.testing.assert_allclose(row[c], result['scores'][row.month][c], atol=1e-8, rtol=0)
        for c in columns:
            np.testing.assert_allclose(metrics[c].mean(), result['cv'][c], atol=1e-8, rtol=0)
    value = dict(status='pass', rows=count, truth_calendar_and_all_controls_exact=True,
        independent_sql_scores=True, identity=art(MANIFEST), result=art(OUT / 'result.json'),
        predictions=art(OUT / 'scored.parquet'), source=art(__file__),
        scope='Direct joint stack calculation and exact single-change replays; independent SQL label/control/score checks; retained model audits')
    write_json(OUT / 'audit.json', value)
    print(value, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['guard', 'score', 'audit'])
    args = parser.parse_args()
    with threadpool_limits(2):
        globals()[args.stage]()
