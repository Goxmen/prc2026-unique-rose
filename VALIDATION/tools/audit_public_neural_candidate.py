"""Independently replay the public neural candidate, constraints and CV."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'VALIDATION'))

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

import public_neural_candidate as run
from records import write_json, sha256


def audit(version):
    dest = run.OUT / f'submission_v{version}'
    assert not (dest / 'audit.json').exists(), 'Preserve completed audit'
    mp = dest / 'manifest.json'
    manifest = run.read(mp)
    assert manifest['state'] == 'prepared' and manifest['parent'] == 'v75'
    for path, digest in manifest['sources'].items():
        assert sha256(ROOT / path) == digest, path
    run.selected_cv(manifest['with_main'])
    identity = run.production.guard()
    production_audit = run.read(run.production.OUT / 'audit.json')
    assert production_audit['status'] == 'pass' and production_audit['identity'] == run.art(run.production.MANIFEST)
    data = run.rome.physical.parent.ranking_data('old')
    ids = data['rank_ids']
    reference = pd.read_parquet(run.verify(run.read(
        run.rome.physical.OUT / 'submission_v70/manifest.json')['parts']))
    calls = pd.read_parquet(run.verify(run.read(
        run.rome.physical.selected.OUT / 'candidates/old/manifest.json')['parts']))
    parent75 = pd.read_parquet(run.rome.OUT / 'submission_v75/predictions.parquet')
    parts = pd.read_parquet(dest / 'predictions.parquet')
    for frame in [reference, calls, parent75, parts]:
        np.testing.assert_array_equal(frame.mvt_id, ids)
    pools = {}
    for name in ['old', 'new']:
        bank = []
        for k in range(3):
            if name == 'new':
                record = run.production.completed(k)
                assert record['all_ranking_reload_exact']
                source = record['prediction']
            else:
                record = run.production.physical.completed(k, identity['parent_packet'])
                source = record['predictions']['old']
            prediction = pd.read_parquet(run.verify(source))
            np.testing.assert_array_equal(prediction.mvt_id, ids)
            bank.append(prediction.base.to_numpy(float))
        pools[name] = np.asarray(bank).mean(axis=0)
        np.testing.assert_array_equal(pools[name], parts['physical_'+name])
    old_neural = (calls.calls_nn.to_numpy(float)+pools['old'])/2
    new_neural = (calls.calls_nn.to_numpy(float)+pools['new'])/2
    np.testing.assert_array_equal(old_neural, reference.nn_base)
    np.testing.assert_array_equal(old_neural, parts.neural_old)
    np.testing.assert_array_equal(new_neural, parts.neural_new)
    np.testing.assert_array_equal(calls.calls_nn, parts.calls_nn)
    tree = reference.gbm_base.to_numpy().copy()
    if manifest['with_main']:
        main = run.main_candidate.production
        main.guard()
        rec = run.read(main.OUT / 'predictions/old.json')
        assert rec['all_ranking_reload_exact'] and rec['identity'] == run.art(main.OUT / 'identity.json')
        replacement = pd.read_parquet(run.verify(rec['prediction']))
        np.testing.assert_array_equal(replacement.mvt_id, ids)
        components, sources = main.old.parent_components('old', ids)
        for source in sources:
            run.verify(source)
        names = main.old.parent.hs.GBM
        values = components[names].to_numpy(float).copy()
        old_main = pd.read_parquet(run.verify(run.read(main.old.OUT / 'predictions/old.json')['prediction']))
        np.testing.assert_array_equal(old_main.mvt_id, ids)
        values[:, names.index('lgb')] = old_main.pred.to_numpy(float)
        np.testing.assert_array_equal(values.mean(axis=1), tree)
        values[:, names.index('lgb')] = replacement.pred.to_numpy(float)
        tree = values.mean(axis=1)
    np.testing.assert_array_equal(tree, parts.tree)
    full, _ = run.rome.physical.parent.final_frame(data, new_neural, tree, parent75.p.to_numpy())
    np.testing.assert_array_equal(full.pred, parts.candidate)
    np.testing.assert_array_equal(parent75.p, parts.p)
    np.testing.assert_array_equal(reference.pred, parts.control)
    parent = pd.read_parquet(ROOT / 'submissions/unique-rose_v74.parquet')
    np.testing.assert_array_equal(parent.MVT_ID_mvt, ids)
    np.testing.assert_array_equal(parent.TAXITIME_SEC_mvt, parts.parent74)
    q = parent.TAXITIME_SEC_mvt.to_numpy(float)+full.pred.to_numpy()-reference.pred.to_numpy()
    raw = pd.read_parquet(run.rome.features.base.OUT / 'ranking_inputs.parquet').set_index('mvt_id').loc[ids]
    known = raw.has_id.to_numpy(bool) & raw.nm_gap.notna().to_numpy()
    gap = raw.nm_gap.to_numpy(float)
    q[known] = np.maximum(0, np.minimum(np.maximum(q[known], gap[known]-3606), gap[known]+3606))
    q = np.maximum(q, 0)
    matches = pd.read_parquet(run.rome.r20.constraints.exclusion.OUT / 'ranking_matches.parquet')
    positions = {v: k for k, v in enumerate(ids)}
    for row in matches.loc[matches.accepted].itertuples():
        i = positions[row.mvt_id]
        lower, upper = row.recovered_gap-3606, row.recovered_gap+3606
        if lower < q[i] < upper:
            q[i] = min([v for v in [lower, upper] if v >= 0], key=lambda v: abs(q[i]-v))
    np.testing.assert_array_equal(q, parts.pred)
    expected = np.rint(q).astype('int32')
    submitted = pq.read_table(ROOT / manifest['file'])
    template = pq.read_table(ROOT / 'data/submitting.parquet')
    assert submitted.schema == template.schema and submitted['MVT_ID_mvt'].equals(template['MVT_ID_mvt'])
    np.testing.assert_array_equal(submitted['TAXITIME_SEC_mvt'].to_numpy(), expected)
    np.testing.assert_array_equal(parts.submitted, expected)
    assert sha256(ROOT / manifest['file']) == manifest['sha256']
    with duckdb.connect(config={'threads': 1}) as con:
        con.read_parquet(str(dest / 'cv.parquet')).create_view('candidate')
        con.read_parquet(str(run.cv.main.features.base.OUT / 'rows2025.parquet')).create_view('truth')
        con.read_parquet(str(run.cv.OUT / 'scored.parquet')).create_view('joint')
        column = 'combined_transfer' if manifest['with_main'] else 'neural_transfer'
        count, distinct_ids, errors = con.execute(f'''select count(*),count(distinct c.mvt_id),
            sum(case when c.y is distinct from t.y or c.day is distinct from t.day or
            c.month is distinct from t.month or c.pred is distinct from j.{column} or
            c.baseline_transfer is distinct from j.baseline_transfer then 1 else 0 end)
            from candidate c join truth t using(mvt_id) join joint j using(mvt_id)''').fetchone()
        assert count == distinct_ids == 344419 and errors == 0
        metrics = con.execute('''select month,count(*) n,sqrt(avg(pow(pred-y,2))) cv
            from candidate group by month order by month''').fetchdf()
        assert metrics.n.sum() == 344419 and metrics.month.tolist() == ['2025-01', '2025-07']
        assert abs(metrics.cv.mean()-manifest['jan_jul_cv']) < 1e-8
    value = dict(status='pass', source=run.art(__file__), output_sha256=sha256(ROOT / manifest['file']),
        full_prediction_replay_rows=len(ids), independent_neural_pool_exact=True,
        independent_tree_mean_exact=True, independent_constraints_exact=True,
        independent_sql_truth_calendar_controls_and_cv=True, cv=float(metrics.cv.mean()),
        schema_and_ids_exact=True, production_model_audit=run.art(run.production.OUT / 'audit.json'),
        scope='All final arithmetic and constraints replayed; retained production stages reload all new model predictions')
    write_json(dest / 'audit.json', value)
    manifest['sources'].update({str(Path(__file__).relative_to(ROOT)): sha256(__file__),
        str((dest / 'audit.json').relative_to(ROOT)): sha256(dest / 'audit.json')})
    manifest['audit']['independent_final_audit_pending'] = False
    manifest['audit']['independent_review'] = value
    write_json(mp, manifest)
    print(value, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', type=int, required=True)
    args = parser.parse_args()
    with threadpool_limits(2):
        audit(args.version)
