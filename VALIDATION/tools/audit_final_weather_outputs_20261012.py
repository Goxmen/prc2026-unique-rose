"""Independent SQL reconstruction of retained weather mixtures and all final rules.

    .venv/bin/python VALIDATION/tools/audit_final_weather_outputs_20261012.py

Checks every old component and v60/v61 integer, final identities/domains, saved
model reload receipts and independently derived mixture/correction arithmetic.
No model inference, training or calibration is repeated by this auditor.
"""
import gc
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'VALIDATION/runs'
OUT = RUNS / 'v3_20261012_final_weather_inference'
LEGACY = RUNS / 'v3_20261008_final_submit'
V58 = RUNS / 'v3_20261006_blend_weight_submit/submission_v58'
PARTS = ['rome', 'lgb', 'airport', 'catboost', 'nn']
RAW_SHA = {
    'old': '9db75e3a56fa09b87aab72e191a53da3fe763d9887ad2b23aea9cde6a6325f86',
    'final': '4b0692a792846079516a2749e034b0abda55df72c213eabdcda52a4d1f6bc69f',
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            h.update(block)
    return h.hexdigest()


def art(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size)


def main():
    target = OUT / 'independent_audit.json'
    assert not target.exists()
    audit_identity = read(OUT / 'output_auditor_identity.json')
    assert audit_identity['source'] == art(Path(__file__))
    assert audit_identity['identity'] == art(OUT / 'identity.json')
    identity = read(OUT / 'identity.json')
    verified = set()

    def verify(record):
        path = ROOT / record['path']
        if 'bytes' in record:
            assert path.stat().st_size == record['bytes'], path
        key = (str(path), record['sha256'])
        if key not in verified:
            assert sha(path) == record['sha256'], path
            verified.add(key)
        return path

    for source in identity['sources']:
        verify(source)
    assert identity['design']['versions'] == [60, 61]
    assert len(identity['models']) == 28
    assert sum(name.endswith('.pt') for name in identity['models']) == 15
    for source in identity['models'].values():
        verify(source)
    prepared = read(OUT / 'prepared.json')
    assert prepared['identity'] == art(OUT / 'identity.json')
    verify(prepared['preprocessing'])
    assert prepared['original167_input_values_exact']
    assert prepared['checkpoint_bins_cards_scales_exact'] == 15
    assert prepared['training_rows'] == 2051327 and prepared['old_rows'] == 344841
    checks, component_checks = {}, []
    with duckdb.connect(config={'threads': 1, 'memory_limit': '1GB'}) as con:
        con.execute("SET TimeZone='UTC'")
        for view in ['old', 'final']:
            n_expected = 344841 if view == 'old' else 670790
            template = ROOT / 'data' / ('submitting.parquet' if view == 'old' else 'final_submitting.parquet')
            raw_path = ROOT / 'data' / ('ranking.parquet' if view == 'old' else 'final_ranking.parquet')
            assert verify(audit_identity['raw'][view]) == raw_path
            assert audit_identity['raw'][view]['sha256'] == RAW_SHA[view], raw_path
            ids = pd.read_parquet(template, columns=['MVT_ID_mvt']).MVT_ID_mvt.to_numpy()
            assert len(ids) == len(np.unique(ids)) == n_expected
            con.read_parquet(str(template)).create_view('template', replace=True)
            con.read_parquet(str(raw_path)).create_view('raw', replace=True)
            for part in PARTS:
                receipt = read(OUT / view / f'{part}.json')
                assert receipt['identity'] == art(OUT / 'identity.json')
                assert receipt['view'] == view and receipt['part'] == part
                assert receipt['rows'] == n_expected and receipt['no_model_fit']
                assert receipt['all_original_values_exact']
                if view == 'final':
                    assert receipt['all_final_reloads_exact']
                path = verify(receipt['prediction'])
                con.read_parquet(str(path)).create_view(part, replace=True)
                cols = [x[0] for x in con.execute(f'DESCRIBE {part}').fetchall()]
                expected = ['mvt_id'] + ([f'n{g}_{k}' for g in [42, 142, 242] for k in range(5)] if part == 'nn' else ['pred'])
                assert cols == expected
                assert con.execute(f'SELECT count(*),count(distinct mvt_id) FROM {part}').fetchone() == (n_expected, n_expected)
                assert con.execute(f'''SELECT count(*) FROM {part} p FULL JOIN template t
                    ON p.mvt_id=t.MVT_ID_mvt WHERE p.mvt_id IS NULL OR t.MVT_ID_mvt IS NULL''').fetchone()[0] == 0
                invalid = ' OR '.join(f'"{c}" IS NULL OR NOT isfinite("{c}") OR "{c}"<0' for c in cols[1:])
                assert con.execute(f'SELECT count(*) FROM {part} WHERE {invalid}').fetchone()[0] == 0
                if part == 'rome':
                    assert con.execute('''SELECT count(*) FROM rome p JOIN raw r ON p.mvt_id=r.MVT_ID_mvt
                        WHERE p.pred>1 OR (r.ADEP_mvt<>'LIRF' AND p.pred<>0)''').fetchone()[0] == 0
                if view == 'old':
                    con.read_parquet(str(verify(identity['old_parts'][part]))).create_view('original', replace=True)
                    mismatch = ' OR '.join(f'p."{c}" IS DISTINCT FROM o."{c}"' for c in cols[1:])
                    assert con.execute(f'''SELECT count(*) FROM {part} p FULL JOIN original o USING(mvt_id)
                        WHERE p.mvt_id IS NULL OR o.mvt_id IS NULL OR {mismatch}''').fetchone()[0] == 0
                component_checks.append(dict(view=view, part=part, rows=n_expected, values=n_expected*(len(cols)-1),
                    prediction=receipt['prediction'], original_all_values_exact=view=='old',
                    producer_saved_model_reload_receipt_exact=view=='final'))

            packet_path = LEGACY / 'work' / f'inputs_{view}.pkl'
            packet_record = read(LEGACY / f'inputs_{view}.json')
            assert sha(packet_path) == packet_record['sha256']
            packet = pd.read_pickle(packet_path)
            handlers = pd.concat([pd.DataFrame(dict(mvt_id=item['raw'].mvt_id.to_numpy(),
                handler_share=item['V'].handler_arr_sched_share.to_numpy(),
                handler_n=item['V'].handler_arr_n.to_numpy())) for item in packet['months'].values()], ignore_index=True)
            assert handlers.mvt_id.is_unique and set(handlers.mvt_id) == set(ids)
            fitted, late_mean = packet['fitted'], packet['parameters']['late_mean']
            assert fitted['settings'] == dict(clock_agreement_sec=1800, clock_fit_min_sec=7200,
                clock_prior_rows=25, day_min_gap_sec=50400, day_prior_rows=5, handler_threshold=.5,
                handler_min_arrivals=20, handler_min_training_dates=5, rule_max_gap_sec=43200,
                rule_max_probability=.05)
            del packet
            gc.collect()
            con.register('handlers', handlers)
            con.execute('CREATE OR REPLACE TEMP TABLE clock_offsets(airport VARCHAR,offset_sec DOUBLE)')
            con.executemany('INSERT INTO clock_offsets VALUES(?,?)', list(fitted['clock']['airport_residual_sec'].items()))
            if view == 'old':
                parents = pd.concat([pd.read_parquet(V58 / f'ranking_{month}.parquet', columns=['mvt_id', 'blend_base'])
                                     for month in ['2026-01', '2026-07']], ignore_index=True)
                con.register('parent_base', parents)
            else:
                con.read_parquet(str(LEGACY / 'work/parts/nn.parquet')).create_view('parent_nn', replace=True)
                con.read_parquet(str(LEGACY / 'work/parts/trees.parquet')).create_view('parent_tree', replace=True)
                con.execute('''CREATE OR REPLACE TEMP VIEW parent_base AS SELECT n.mvt_id,
                    cast(.71 AS DOUBLE)*cast(n.pred AS DOUBLE)+cast(.29000000000000004 AS DOUBLE)*
                    ((cast(t.lgb AS DOUBLE)+cast(t.airport AS DOUBLE)+cast(t.catboost AS DOUBLE))/3) blend_base
                    FROM parent_nn n JOIN parent_tree t USING(mvt_id)''')
            nn_mean = '(' + '+'.join(f'cast(n."n{g}_{k}" AS DOUBLE)' for g in [42, 142, 242] for k in range(5)) + ')/15'
            con.execute(f'''CREATE OR REPLACE TEMP TABLE components AS SELECT n.mvt_id,
                cast(p.pred AS DOUBLE) probability,cast(parent_base.blend_base AS DOUBLE) base60,
                cast(.71 AS DOUBLE)*({nn_mean})+cast(.29 AS DOUBLE)*
                ((cast(l.pred AS DOUBLE)+cast(a.pred AS DOUBLE)+cast(c.pred AS DOUBLE))/3) base61
                FROM nn n JOIN rome p USING(mvt_id) JOIN lgb l USING(mvt_id)
                JOIN airport a USING(mvt_id) JOIN catboost c USING(mvt_id) JOIN parent_base USING(mvt_id)''')
            assert con.execute('SELECT count(*),count(distinct mvt_id) FROM components').fetchone() == (n_expected,n_expected)
            con.execute('''CREATE OR REPLACE TEMP TABLE visible AS SELECT c.*,r.ADEP_mvt airport,
                r.FLIGHT_mvt flight,r.FLIGHT_ID_mvt IS NULL noid,r.MVT_TIME_UTC_mvt t,
                strftime(r.MVT_TIME_UTC_mvt,'%Y-%m') AS month,
                epoch(r.MVT_TIME_UTC_mvt-r.SCHED_TIME_UTC_mvt) gap,
                epoch(r.MVT_TIME_UTC_mvt-r.LOBT_flt) lobt,
                epoch(r.MVT_TIME_UTC_mvt-r.AOBT_3_flt) aobt,
                date_diff('day',timezone('Europe/Rome',cast(r.SCHED_TIME_UTC_mvt AS TIMESTAMPTZ))::DATE,
                    timezone('Europe/Rome',cast(r.MVT_TIME_UTC_mvt AS TIMESTAMPTZ))::DATE) local_days,
                h.handler_share,h.handler_n FROM components c JOIN raw r ON c.mvt_id=r.MVT_ID_mvt
                JOIN handlers h USING(mvt_id) WHERE r.PHASE_mvt='DEP' ''')
            con.execute('''CREATE OR REPLACE TEMP TABLE twins AS
                WITH gaps AS (SELECT *,epoch(t-lag(t) OVER(PARTITION BY month,airport,flight ORDER BY t,mvt_id)) distance
                    FROM visible WHERE flight IS NOT NULL),
                starts AS (SELECT *,CASE WHEN distance<=180 THEN 0 ELSE 1 END start_cluster FROM gaps),
                grouped AS (SELECT *,sum(start_cluster) OVER(PARTITION BY month,airport,flight ORDER BY t,mvt_id) AS twin_group FROM starts),
                sizes AS (SELECT *,count(*) OVER(PARTITION BY month,airport,flight,twin_group) n,
                    sum(CASE WHEN noid THEN 0 ELSE 1 END) OVER(PARTITION BY month,airport,flight,twin_group) with_id FROM grouped)
                SELECT mvt_id FROM sizes WHERE n=2 AND with_id=1 AND NOT noid AND isfinite(lobt)''')
            dp, cp = fitted['day'], fitted['clock']
            for version in [60, 61]:
                report = read(OUT / f'{view}_v{version}.json')
                assert report['identity'] == art(OUT / 'identity.json')
                assert report['view'] == view and report['version'] == version
                assert report['rows'] == n_expected and report['no_model_fit']
                assert report['all_original_predictions_exact'] and report['all_original_component_values_exact']
                path = verify(report['prediction'])
                frame_ids = pd.read_parquet(path, columns=['mvt_id']).mvt_id.to_numpy()
                np.testing.assert_array_equal(frame_ids, ids)
                con.read_parquet(str(path)).create_view('p', replace=True)
                con.execute(f'''CREATE OR REPLACE TEMP TABLE expected AS
                    WITH flags AS (SELECT v.*,coalesce(v.airport='LIRF' AND noid AND gap>3600 AND gap<14400,false) late,
                        coalesce(v.airport='LIRF' AND noid AND gap>=14400,false) fired,
                        coalesce(v.airport='LIRF' AND noid AND gap>=50400 AND local_days=1,false) day_changed,
                        coalesce(v.airport='LIRF' AND noid AND gap>=14400 AND gap<43200 AND probability<.05,false) rule_changed,
                        coalesce(NOT noid AND isfinite(lobt) AND isfinite(aobt) AND least(lobt,aobt)>=43200 AND abs(lobt-aobt)<=1800,false) clock_changed,
                        twins.mvt_id IS NOT NULL twin_changed,
                        coalesce(co.offset_sec,cast({cp['pooled_residual_sec']!r} AS DOUBLE)) offset_sec
                        FROM visible v LEFT JOIN twins USING(mvt_id) LEFT JOIN clock_offsets co ON v.airport=co.airport),
                    hybrid AS (SELECT *,CASE WHEN fired THEN gap WHEN airport='LIRF' AND isfinite(gap)
                        THEN greatest(0,probability*gap+(1-probability)*
                            (CASE WHEN late THEN least(cast({late_mean!r} AS DOUBLE),greatest(gap-3600,60)) ELSE base{version} END))
                        ELSE base{version} END mixed FROM flags),
                    windowed AS (SELECT *,CASE WHEN NOT noid AND isfinite(lobt)
                        THEN greatest(0,least(greatest(mixed,lobt-3660),lobt+3660)) ELSE greatest(0,mixed) END bounded FROM hybrid),
                    capped AS (SELECT *,least(bounded,
                        CASE WHEN noid AND airport NOT IN ('LIRF','LTFM') AND gap>4500 THEN gap-3600 ELSE 'Infinity'::DOUBLE END,
                        CASE WHEN noid AND airport<>'LIRF' AND gap>1800 THEN gap ELSE 'Infinity'::DOUBLE END) control FROM windowed),
                    weighted AS (SELECT *,CASE WHEN isfinite(handler_share) AND handler_n>=20
                        THEN CASE WHEN handler_share<.5 THEN cast({dp['handler_weights']['low']!r} AS DOUBLE)
                                  ELSE cast({dp['handler_weights']['high']!r} AS DOUBLE) END
                        ELSE cast({dp['pooled_weight']!r} AS DOUBLE) END day_weight FROM capped)
                    SELECT *,CASE WHEN clock_changed THEN greatest(0,least(greatest(aobt+offset_sec,lobt-3660),lobt+3660))
                        WHEN rule_changed THEN probability*gap+(1-probability)*least(cast({fitted['late_mean']!r} AS DOUBLE),greatest(gap-3600,60))
                        WHEN day_changed THEN gap+day_weight*(86400+cast({fitted['late_mean']!r} AS DOUBLE)-gap)
                        WHEN twin_changed THEN greatest(lobt-330,0) ELSE control END prediction FROM weighted''')
                n, unique, bad, error = con.execute('''SELECT count(*),count(distinct p.mvt_id),sum(CASE WHEN
                    p.mvt_id IS NULL OR e.mvt_id IS NULL OR p.pred IS NULL OR NOT isfinite(p.pred) OR p.pred<0
                    OR abs(p.pred-e.prediction)>1e-7 OR p.month IS DISTINCT FROM e.month
                    OR p.submitted IS DISTINCT FROM round_even(greatest(0,e.prediction),0) THEN 1 ELSE 0 END),
                    max(abs(p.pred-e.prediction)) FROM p FULL JOIN expected e USING(mvt_id)''').fetchone()
                assert n == unique == n_expected and bad == 0, (view,version,n,unique,bad,error)
                if view == 'old':
                    con.read_parquet(str(verify(identity['original_versions'][str(version)]))).create_view('original',replace=True)
                    assert con.execute('''SELECT count(*) FROM p FULL JOIN original o ON p.mvt_id=o.MVT_ID_mvt
                        WHERE p.mvt_id IS NULL OR o.MVT_ID_mvt IS NULL OR p.submitted IS DISTINCT FROM o.TAXITIME_SEC_mvt''').fetchone()[0] == 0
                months = dict(con.execute('SELECT month,count(*) FROM p GROUP BY month ORDER BY month').fetchall())
                want = {'2026-01':152719,'2026-07':192122} if view=='old' else {'2026-01':152719,'2026-02':144158,'2026-06':181791,'2026-07':192122}
                assert months == want
                checks[f'{view}_v{version}'] = dict(rows=n,months=months,max_SQL_float_error=float(error),
                    all_mixture_correction_rules_integer_rounding_exact=True,template_order_exact=True,
                    old_original_integers_exact=view=='old',prediction=report['prediction'])
            con.unregister('handlers')
            if view == 'old':
                con.unregister('parent_base')
            del handlers
            gc.collect()
    target.write_text(json.dumps(dict(status='pass',source=art(Path(__file__)),identity=art(OUT/'identity.json'),
        auditor_identity=art(OUT/'output_auditor_identity.json'),
        checks=checks,components=component_checks,sources_verified=len(verified),models=28,raw_sha256=RAW_SHA,
        original_component_values=sum(x['values'] for x in component_checks if x['view']=='old'),
        final_component_values=sum(x['values'] for x in component_checks if x['view']=='final'),
        prediction_rows_checked=sum(x['rows'] for x in checks.values()),no_fit=True,no_inference=True,
        caveats=['v60 retains already-audited October8 parent neural refit provenance.',
                 'Saved-model reloads are verified producer receipts; this auditor independently derives output arithmetic.',
                 'Handler inputs reuse hashed previously audited input packets; raw clocks and twins are independently reconstructed.']),indent=2)+'\n')
    print(json.dumps(dict(status='pass',prediction_rows_checked=sum(x['rows'] for x in checks.values()))))


if __name__ == '__main__':
    with threadpool_limits(1):
        main()
