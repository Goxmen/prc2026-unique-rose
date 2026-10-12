"""Independent v56 copy-input SQL, checkpoint reload, fallback and final audit."""
import gc
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]
import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import callsign_submit as parent
import nn_improve_submit as finish

OUT = ROOT / 'VALIDATION/runs/v3_20261012_final_missing_flightnumber_reconstruct'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def checked(record):
    path = ROOT / record['path']
    assert sha(path) == record['sha256'], record['path']
    return path


def main():
    assert not (OUT / 'independent_audit.json').exists()
    result = read(OUT / 'result.json'); identity = read(OUT / 'identity.json')
    assert result['state'] == 'exact_old_replay_final_inference_complete'
    checked(result['identity'])
    for record in identity['sources']:
        checked(record)
    trained = read(OUT / 'trained.json'); preflight = read(OUT / 'preflight.json')
    assert trained['identity'] == preflight['identity'] == result['identity']
    checked(trained['preflight']); checked(preflight['history'])
    assert preflight['rows'] == 160577 and preflight['positives'] == 33205
    assert preflight['independent_SQL_leave_month_out_rate_and_counts_exact']
    assert preflight['original_ordered_LIRF_membership_and_raw_gaps_targets_exact']
    model = lgb.Booster(model_file=str(checked(trained['model'])))
    assert model.num_trees() == 1700 and len(model.feature_name()) == 70
    assert model.feature_name() == preflight['columns'] == trained['columns']
    base_columns = model.feature_name()[:-2]
    history = pd.read_parquet(OUT / 'history.parquet')
    checks = {}
    with duckdb.connect(config={'threads': 1, 'memory_limit': '512MB'}) as con:
        con.execute("SET TimeZone='UTC'")
        con.register('history', history)
        con.execute('CREATE TEMP TABLE totals AS SELECT fn,sum(copied) copied,count(*) n FROM history GROUP BY fn')
        prior = con.execute('SELECT avg(copied) FROM history').fetchone()[0]
        for view, report in result['views'].items():
            path = checked(report['prediction']); saved = pd.read_parquet(path).set_index('mvt_id')
            data = parent.ranking_data(view)
            neural, tree, old_probability, sources = parent.parent_parts(view, data)
            assert sources == report['parent_parts']
            for source in sources:
                checked(source)
            offset = 0; rome_rows = 0; fallback_rows = 0
            for month, item in data['months'].items():
                V, raw = item['V'], item['raw']; n = len(raw); sl = slice(offset, offset+n)
                current = saved.loc[raw.mvt_id]
                visible_gap = (pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True)-pd.to_datetime(raw.SCHED_TIME_UTC_mvt, utc=True)).dt.total_seconds().to_numpy()
                np.testing.assert_array_equal(visible_gap, V['mvt_minus_sched_sec'].to_numpy())
                use = raw.ADEP_mvt.eq('LIRF').to_numpy() & np.isfinite(visible_gap)
                q = raw.loc[use, ['mvt_id', 'FLIGHT_mvt']].copy()
                con.register('query_rows', q)
                sql = con.execute('''SELECT q.mvt_id,coalesce(t.n,0) n,
                    (coalesce(t.copied,0)+10.0*?)/(coalesce(t.n,0)+10.0) rate
                    FROM query_rows q LEFT JOIN totals t ON coalesce(q.FLIGHT_mvt,'')=t.fn''', [prior]).df().set_index('mvt_id').loc[q.mvt_id]
                X = V.loc[use, base_columns].copy()
                X['fn_copy_rate'], X['fn_copy_n'] = sql.rate.to_numpy(), sql.n.to_numpy(float)
                probability = np.zeros(n); unseen = np.zeros(n, bool)
                probability[use] = model.predict(X, num_threads=1)
                unseen[use] = sql.n.to_numpy() == 0
                probability[unseen] = old_probability[sl][unseen]
                np.testing.assert_array_equal(current.p, probability)
                np.testing.assert_array_equal(current.old_p, old_probability[sl])
                np.testing.assert_array_equal(current.unseen, unseen)
                base = .6*neural[sl]+.4*tree[sl]
                np.testing.assert_array_equal(current.base, base)
                # This is a checkpoint/input replay using the unchanged, already
                # audited postprocessor, not a new independent arithmetic proof.
                frame, _ = finish.final(base, probability, V, raw, data['parameters'], data['fitted'])
                np.testing.assert_array_equal(current.pred, frame.pred)
                np.testing.assert_array_equal(current.rule_triggered, frame.rule_triggered)
                offset += n; rome_rows += int(use.sum()); fallback_rows += int(unseen.sum())
            assert offset == len(saved) == report['rows']
            con.read_parquet(str(path)).create_view('p', replace=True)
            raw_path = ROOT / 'data' / ('ranking.parquet' if view == 'old' else 'final_ranking.parquet')
            template = ROOT / 'data' / ('submitting.parquet' if view == 'old' else 'final_submitting.parquet')
            con.read_parquet(str(raw_path)).create_view('raw', replace=True)
            con.read_parquet(str(template)).create_view('template', replace=True)
            n, unique, bad = con.execute('''SELECT count(*),count(distinct p.mvt_id),sum(CASE WHEN
                NOT isfinite(p.pred) OR p.pred<0 OR p.submitted IS DISTINCT FROM round_even(greatest(0,p.pred),0)
                OR p.month IS DISTINCT FROM strftime(r.MVT_TIME_UTC_mvt,'%Y-%m') THEN 1 ELSE 0 END)
                FROM p FULL JOIN template t ON p.mvt_id=t.MVT_ID_mvt
                JOIN raw r ON p.mvt_id=r.MVT_ID_mvt WHERE r.PHASE_mvt='DEP' ''').fetchone()
            assert n == unique == report['rows'] == (344841 if view == 'old' else 670790) and bad == 0
            if view == 'old':
                ref = read(ROOT / 'data/derived/scored_submission_cache_20261011/manifest.json')['56']
                assert sha(ROOT / ref['file']) == ref['sha256']
                con.read_parquet(str(ROOT / ref['file'])).create_view('original', replace=True)
                bad = con.execute('SELECT count(*) FROM p FULL JOIN original o ON p.mvt_id=o.MVT_ID_mvt WHERE p.mvt_id IS NULL OR o.MVT_ID_mvt IS NULL OR p.submitted IS DISTINCT FROM o.TAXITIME_SEC_mvt').fetchone()[0]
                assert bad == 0 and fallback_rows == 162+7603
            checks[view] = dict(rows=n, rome_model_rows=rome_rows, fallback_rows=fallback_rows,
                independent_SQL_copy_query_inputs_and_model_reload_exact=True,
                fallback_parent_parts_and_existing_postprocessor_replay_exact=True,
                SQL_ids_dates_rounding_exact=True, old_integer_original_exact=view=='old')
            del data, neural, tree, saved; gc.collect()
    (OUT / 'independent_audit.json').write_text(json.dumps(dict(status='pass', checks=checks,
        source_sha256=sha(__file__), result_sha256=sha(OUT / 'result.json'), model_sha256=sha(OUT / 'model.txt'),
        caveats=['Final base networks reuse the prior audited October8 refits, not a claim of original weight identity.',
                 'Postprocessing uses the unchanged previously audited recipe; its arithmetic is replayed, not independently re-derived here.']), indent=2)+'\n')
    print(json.dumps(dict(status='pass', rows=sum(v['rows'] for v in checks.values()))))


if __name__ == '__main__':
    with threadpool_limits(1):
        main()
