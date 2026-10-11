"""Independently replay the complete main/public-Cat candidate and its bounds."""
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
import public_main_candidate as run
from records import write_json, sha256


def main(version):
    dest = run.OUT / f'submission_v{version}'
    mp = dest / 'manifest.json'
    m = run.read(mp)
    assert m['state'] == 'prepared' and m['parent'] == 'v75'
    for path, digest in m['sources'].items():
        assert sha256(ROOT / path) == digest, path
    production = run.production
    recipe = production.guard()
    record = run.read(production.OUT / 'predictions/old.json')
    assert record['identity'] == run.artifact(production.OUT / 'identity.json')
    assert record['all_ranking_reload_exact'] and record['rows'] == 344841
    new = pd.read_parquet(run.verify(record['prediction']))
    run.verify(record['model'])
    old_main = pd.read_parquet(run.verify(run.read(production.old.OUT / 'predictions/old.json')['prediction']))
    data = run.rome.physical.parent.ranking_data('old')
    ids = data['rank_ids']
    for f in [new, old_main]:
        np.testing.assert_array_equal(f.mvt_id, ids)
    components, sources = production.old.parent_components('old', ids)
    for source in sources:
        run.verify(source)
    names = production.old.parent.hs.GBM
    old_values = components[names].to_numpy(float).copy()
    new_values = old_values.copy()
    old_values[:, names.index('lgb')] = old_main.pred.to_numpy(float)
    new_values[:, names.index('lgb')] = new.pred.to_numpy(float)
    old_tree, new_tree = old_values.mean(axis=1), new_values.mean(axis=1)
    reference = pd.read_parquet(run.verify(run.read(run.rome.physical.OUT / 'submission_v70/manifest.json')['parts']))
    np.testing.assert_array_equal(reference.mvt_id, ids)
    np.testing.assert_array_equal(old_tree, reference.gbm_base)
    parent75 = pd.read_parquet(run.rome.OUT / 'submission_v75/predictions.parquet')
    np.testing.assert_array_equal(parent75.mvt_id, ids)
    full, _ = run.rome.physical.parent.final_frame(data, reference.nn_base.to_numpy(), new_tree, parent75.p.to_numpy())
    out = pd.read_parquet(dest / 'predictions.parquet')
    np.testing.assert_array_equal(out.mvt_id, ids)
    for values, column in [(old_tree, 'tree_old'), (new_tree, 'tree_new'), (full.pred, 'candidate'),
                           (parent75.p, 'p'), (reference.pred, 'control')]:
        np.testing.assert_array_equal(values, out[column])
    best = pd.read_parquet(ROOT / 'submissions/unique-rose_v74.parquet')
    np.testing.assert_array_equal(best.MVT_ID_mvt, ids)
    q = best.TAXITIME_SEC_mvt.to_numpy(float)+full.pred.to_numpy()-reference.pred.to_numpy()
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
            allowed = [v for v in [lower, upper] if v >= 0]
            q[i] = min(allowed, key=lambda v: abs(q[i]-v))
    np.testing.assert_array_equal(q, out.pred)
    expected = np.rint(q).astype('int32')
    submitted = pq.read_table(ROOT / m['file'])
    template = pq.read_table(ROOT / 'data/submitting.parquet')
    assert submitted.schema == template.schema and submitted['MVT_ID_mvt'].equals(template['MVT_ID_mvt'])
    np.testing.assert_array_equal(submitted['TAXITIME_SEC_mvt'].to_numpy(), expected)
    np.testing.assert_array_equal(out.submitted, expected)
    assert sha256(ROOT / m['file']) == m['sha256']
    with duckdb.connect(config={'threads': 1}) as con:
        metrics = con.execute('''select month,count(*) n,sqrt(avg(pow(transferred-y,2))) cv
            from read_parquet(?) group by month order by month''', [str(dest / 'cv.parquet')]).fetchdf()
    assert metrics.n.sum() == 344419
    assert abs(metrics.cv.mean()-m['jan_jul_cv']) < 1e-8
    audit = dict(status='pass', source=run.artifact(__file__), production_model_audit=run.artifact(production.OUT / 'predictions/old.json'),
        full_prediction_replay_rows=len(ids), independent_tree_mean_exact=True, independent_constraints_exact=True,
        independent_sql_cv=True, cv=float(metrics.cv.mean()), schema_and_ids_exact=True,
        output_sha256=sha256(ROOT / m['file']),
        scope='Final arithmetic and constraints independently replayed; all-row model reload was completed during the retained production stage')
    write_json(dest / 'audit.json', audit)
    m['sources'].update({str(Path(__file__).relative_to(ROOT)): sha256(__file__),
        str((dest / 'audit.json').relative_to(ROOT)): sha256(dest / 'audit.json')})
    m['audit']['independent_final_audit_pending'] = False
    m['audit']['independent_review'] = audit
    write_json(mp, m)
    print(audit, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', type=int, required=True)
    args = parser.parse_args()
    with threadpool_limits(2):
        main(args.version)
