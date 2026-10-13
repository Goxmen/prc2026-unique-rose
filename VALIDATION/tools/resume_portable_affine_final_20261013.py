"""Complete only final ID alignment and serialization after the saved affine solve.

The original identity, source, weights and old/CV prediction files are immutable.
The original run failed before persisting its metric summary; summary statistics
are recalculated from saved CV arrays, without predicting or solving again.
"""
import argparse
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'VALIDATION'))
import portable_affine_20261013 as parent
from portable_affine_20261013 import (np, pd, pq, common, seasonal, fill_template,
    threadpool_limits, write_json, art, read, verify, OUT, MANIFEST)

COMPLETION = OUT/'completion_final_order_20261013'
IDENTITY = COMPLETION/'identity.json'
AUDITOR = ROOT/'VALIDATION/tools/audit_portable_affine_completion_20261013.py'
ORIGINAL_AUDITOR = ROOT/'VALIDATION/tools/audit_portable_affine_20261013.py'
DESIGN = dict(change='Unique-ID checked final alignment only',
    reuse=['original identity', 'one saved weight solve', 'old predictions', 'CV predictions'],
    metrics='Recalculate lost summary from saved CV arrays; original summary was not persisted',
    numerical='Original float64 matrix product and original serialization',
    no_solve=True, no_fit=True, no_model_inference=True, no_old_or_cv_prediction_recompute=True,
    no_upload=True)


def freeze():
    assert not IDENTITY.exists()
    original = parent.guard()
    assert not (OUT/'result.json').exists()
    assert not (OUT/'final.parquet').exists() and not (OUT/'final_candidate.parquet').exists()
    assert read(OUT/'geometry.json')['identity'] == art(MANIFEST)
    paths = {Path(__file__), Path(parent.__file__), ORIGINAL_AUDITOR, AUDITOR,
        MANIFEST, OUT/'geometry.json', OUT/'old.parquet', OUT/'cv.parquet',
        OUT/'run.log', ROOT/'data/final_submitting.parquet', ROOT/'VALIDATION/submit.py',
        Path(seasonal.__file__), Path(common.__file__)}
    paths.update(verify(c['final']) for c in original['components'].values())
    COMPLETION.mkdir(parents=True, exist_ok=True)
    write_json(IDENTITY, dict(design=DESIGN, original_identity=art(MANIFEST),
        sources=[art(p) for p in sorted(paths)]))
    print(dict(stage='frozen', identity=art(IDENTITY)), flush=True)


def guard():
    original = parent.guard()
    frozen = read(IDENTITY)
    assert frozen['design'] == DESIGN and frozen['original_identity'] == art(MANIFEST)
    assert art(__file__) in frozen['sources'] and art(AUDITOR) in frozen['sources']
    for record in frozen['sources']:
        verify(record)
    return original


def run():
    original = guard()
    assert not (COMPLETION/'execution.json').exists(), 'No automatic retry'
    assert not any((OUT/n).exists() for n in ['result.json','final.parquet','final_candidate.parquet'])
    started = time.monotonic()
    execution = dict(state='running', pid=os.getpid(), identity=art(IDENTITY))
    write_json(COMPLETION/'execution.json', execution)
    try:
        versions = original['versions']
        geometry = read(OUT/'geometry.json')
        assert geometry['identity'] == art(MANIFEST)
        solved = geometry['solution']
        weights = np.array([solved['weights'][str(v)] for v in versions])
        ids = common.template('final')
        assert pd.Index(ids).is_unique and not pd.Index(ids).hasnans and len(ids) == 670790
        matrix = np.empty((len(ids), len(versions)), dtype=float)
        alignments = []
        for j, version in enumerate(versions):
            record = original['components'][str(version)]
            id_col, value_col = record['id_column'], record['value_column']
            frame = pd.read_parquet(verify(record['final']), columns=[id_col, value_col])
            input_ids = pd.Index(frame[id_col])
            assert input_ids.is_unique and not input_ids.hasnans and len(input_ids) == len(ids)
            locations = input_ids.get_indexer(ids)
            assert (locations >= 0).all() and len(np.unique(locations)) == len(ids)
            np.testing.assert_array_equal(frame[id_col].to_numpy()[locations], ids)
            values = frame[value_col].to_numpy(dtype=float)[locations]
            assert np.isfinite(values).all() and (values >= 0).all()
            np.testing.assert_array_equal(values, np.rint(values))
            matrix[:, j] = values
            alignments.append(dict(version=version, input=record['final'], id_column=id_col,
                value_column=value_col, rows=len(ids), reordered_rows=int(np.count_nonzero(input_ids.to_numpy()!=ids)),
                unique_ids=True, exact_id_set=True))
        raw = matrix @ weights
        assert np.isfinite(raw).all()
        pd.DataFrame(dict(mvt_id=ids, raw=raw, submitted=common.serialized(raw))).to_parquet(OUT/'final.parquet', index=False)
        template = pq.read_table(ROOT/'data/final_submitting.parquet')
        table = fill_template(template, ids, raw)
        assert table.schema == template.schema and table['MVT_ID_mvt'].equals(template['MVT_ID_mvt'])
        pq.write_table(table, OUT/'final_candidate.parquet')
        assert pq.read_table(OUT/'final_candidate.parquet').equals(table)
        # Reuse the original saved outputs byte-for-byte. Only summary work was
        # lost in the earlier exception, so do not invoke parent.run().
        old = pd.read_parquet(OUT/'old.parquet')
        cv = pd.read_parquet(OUT/'cv.parquet')
        metrics = {m:{k:float(np.sqrt(np.mean((g[k]-g.y)**2))) for k in ['v70','v80','submitted']} for m,g in cv.groupby('month')}
        paired = {k:seasonal.paired_bootstrap(cv.day, cv.month, (cv[k]-cv.y)**2,
            (cv.submitted-cv.y)**2) for k in ['v70','v80']}
        distance = float(np.sqrt(np.mean((old.submitted.astype(float)-old.raw)**2)))
        lo, hi = solved['score_rounding_rmse_interval']
        guard()
        completed = dict(identity=art(IDENTITY), original_identity=art(MANIFEST),
            geometry=art(OUT/'geometry.json'), old=art(OUT/'old.parquet'), cv=art(OUT/'cv.parquet'),
            final=art(OUT/'final.parquet'), final_candidate=art(OUT/'final_candidate.parquet'),
            alignments=alignments, summary_only_recalculated=True, **DESIGN)
        write_json(COMPLETION/'result.json', completed)
        write_json(OUT/'result.json', dict(identity=art(MANIFEST), geometry=art(OUT/'geometry.json'),
            completion_identity=art(IDENTITY), completion_result=art(COMPLETION/'result.json'),
            predictions={view:art(OUT/f'{view}.parquet') for view in ['old','cv','final']},
            final_candidate=art(OUT/'final_candidate.parquet'), versions=versions,
            rows=dict(old=len(old),cv=len(cv),final=len(ids)),
            cv={k:float(np.mean([d[k] for d in metrics.values()])) for k in ['v70','v80','submitted']},
            months=metrics, paired=paired, projected_old_rmse=solved['projected_rmse'],
            serialization_rms=distance, conservative_old_rmse_interval=[max(0,lo-distance),hi+distance],
            independent_audit_pending=True, selected_for_final=False, no_upload=True))
        execution.update(state='complete', seconds=time.monotonic()-started,
            result=art(COMPLETION/'result.json'), combined_result=art(OUT/'result.json'))
        write_json(COMPLETION/'execution.json', execution)
        print(dict(stage='final_complete', rows=len(ids), components=len(versions),
            reordered_components=sum(a['reordered_rows']>0 for a in alignments),
            cv=read(OUT/'result.json')['cv']), flush=True)
    except BaseException:
        execution.update(state='failed', seconds=time.monotonic()-started, failure=traceback.format_exc())
        write_json(COMPLETION/'execution.json', execution)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze','run'])
    args = parser.parse_args()
    with threadpool_limits(1):
        (freeze if args.stage=='freeze' else run)()
