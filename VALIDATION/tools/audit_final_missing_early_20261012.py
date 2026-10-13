"""Audit one recovered early version's old replay and final integer assembly.

    .venv/bin/python VALIDATION/tools/audit_final_missing_early_20261012.py --version 4

Requires the producer's exact-old gate and model-reload receipts; independently
joins raw IDs/months and reconstructs all integer rounding. No model fit or
inference is repeated. Temporal-history features retain their pinned original
helper implementation and are not independently rederived here.
"""
import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'VALIDATION/runs/v3_20261012_final_missing_early_reconstruct'


def read(path):
    return json.loads(path.read_text())


def art(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            digest.update(block)
    return dict(path=str(path.relative_to(ROOT)), sha256=digest.hexdigest())


def main(version):
    audit_identity = read(OUT / 'output_auditor_identity.json')
    assert audit_identity['source'] == art(Path(__file__))
    assert audit_identity['identity'] == art(OUT / 'identity.json')
    identity = read(OUT / 'identity.json')
    assert version in identity['design']['versions']
    for source in identity['sources']:
        assert source == art(ROOT / source['path']), source['path']
    directory = OUT / f'v{version}'
    target = directory / 'independent_audit.json'
    assert not target.exists()
    result, trained, preflight = [read(directory / f'{name}.json')
                                  for name in ['result', 'trained', 'preflight']]
    for record in [result, trained, preflight]:
        assert record['identity'] == art(OUT / 'identity.json') and record['version'] == version
    assert result['state'] == 'exact_old_replay_final_inference_complete'
    assert result['views']['old']['exact_original_integer_replay']
    assert result['views']['old']['changed_rows'] == 0
    assert trained['model'] == art(directory / 'model.txt') and trained['rounds'] == 800
    assert trained['saved_before_inference'] and trained['preflight'] == art(directory / 'preflight.json')
    assert preflight['training'] == art(directory / 'training.pkl')
    assert preflight['original_metadata_feature_order_config_and_count_exact']
    assert preflight['independent_SQL_raw_membership_labels_basic18_and_fpl_exact']
    assert preflight['raw_order_matches_retained_200000_cap_control']
    assert preflight['rows'] == identity['expected'][str(version)]['rows']
    assert preflight['features'] == identity['expected'][str(version)]['features']
    cache = read(ROOT / 'data/derived/scored_submission_cache_20261011/manifest.json')[str(version)]
    checks = {}
    with duckdb.connect(config={'threads': 1, 'memory_limit': '512MB'}) as con:
        con.execute("SET TimeZone='UTC'")
        for view, rows in [('old', 344841), ('final', 670790)]:
            receipt = read(directory / f'{view}.json')
            assert receipt == result['views'][view]
            assert receipt['prediction'] == art(directory / f'{view}.parquet')
            assert receipt['rows'] == rows and receipt['all_model_reload_predictions_exact']
            assert receipt['original_feature_order_exact'] and receipt['template_order_exact']
            template = ROOT / 'data' / ('submitting.parquet' if view == 'old' else 'final_submitting.parquet')
            raw = ROOT / 'data' / ('ranking.parquet' if view == 'old' else 'final_ranking.parquet')
            pred = pd.read_parquet(directory / f'{view}.parquet')
            ids = pd.read_parquet(template, columns=['MVT_ID_mvt']).MVT_ID_mvt
            assert list(pred) == ['mvt_id', 'month', 'pred', 'submitted']
            assert len(pred) == rows and pred.mvt_id.is_unique and ids.is_unique
            np.testing.assert_array_equal(pred.mvt_id, ids)
            con.register('prediction', pred)
            con.read_parquet(str(raw)).create_view('raw', replace=True)
            count, unique, invalid = con.execute('''WITH departures AS
                (SELECT MVT_ID_mvt mvt_id,strftime(MVT_TIME_UTC_mvt,'%Y-%m') AS "month" FROM raw WHERE PHASE_mvt='DEP')
                SELECT count(*),count(DISTINCT p.mvt_id),sum(CASE WHEN p.mvt_id IS NULL OR r.mvt_id IS NULL
                OR p.month IS DISTINCT FROM r.month OR p.pred IS NULL OR NOT isfinite(p.pred)
                OR p.pred<0 OR p.submitted IS NULL OR NOT isfinite(p.submitted)
                OR p.submitted IS DISTINCT FROM round_even(p.pred,0) OR p.submitted>2147483647
                THEN 1 ELSE 0 END) FROM prediction p FULL JOIN departures r USING(mvt_id)''').fetchone()
            assert count == unique == rows and invalid == 0, (count, unique, invalid)
            months = dict(con.execute('SELECT month,count(*) FROM prediction GROUP BY month ORDER BY month').fetchall())
            assert months == receipt['months']
            assert set(months) == ({'2026-01', '2026-07'} if view == 'old' else
                                   {'2026-01', '2026-02', '2026-06', '2026-07'})
            if view == 'old':
                original = ROOT / cache['file']
                assert art(original)['sha256'] == cache['sha256']
                con.read_parquet(str(original)).create_view('original')
                bad = con.execute('''SELECT count(*) FROM prediction p FULL JOIN original o
                    ON p.mvt_id=o.MVT_ID_mvt WHERE p.mvt_id IS NULL OR o.MVT_ID_mvt IS NULL
                    OR p.submitted IS DISTINCT FROM o.TAXITIME_SEC_mvt''').fetchone()[0]
                assert bad == 0
            checks[view] = dict(rows=rows, prediction=receipt['prediction'], original_integer_replay_exact=view=='old',
                all_raw_departure_IDs_months_and_domains_exact=True, all_rounding_exact=True,
                template_order_exact=True, saved_model_reload_receipt_verified=True)
            con.unregister('prediction')
    report = dict(status='pass', version=version, source=art(Path(__file__)),
        auditor_identity=art(OUT / 'output_auditor_identity.json'), identity=art(OUT / 'identity.json'),
        result=art(directory / 'result.json'), checks=checks, prediction_rows_checked=1015631,
        no_fit=True, no_inference=True, original_model_hash_unavailable=True,
        caveats=['Exact integer replay is established, not original model byte identity.',
                 'v5-v9 original modeling.py source is absent; exact replay is their acceptance criterion.',
                 'Model reloads are verified producer receipts; this auditor independently derives ID/month/domain/rounding arithmetic.',
                 'Temporal-history features use pinned original helpers; no independent rederivation is claimed.'])
    target.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status='pass', version=version, rows=1015631)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', type=int, required=True, choices=[4, 5, 6, 7, 8, 9])
    args = parser.parse_args()
    with threadpool_limits(1):
        main(args.version)
