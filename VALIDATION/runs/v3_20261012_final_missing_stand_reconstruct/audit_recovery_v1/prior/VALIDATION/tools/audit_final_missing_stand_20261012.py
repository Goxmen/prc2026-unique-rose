"""Independent all-row SQL reconstruction of v30/33/34/37 final semantics."""
import hashlib
import json
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'VALIDATION/runs/v3_20261012_final_missing_stand_reconstruct'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def checked(record):
    assert sha(ROOT / record['path']) == record['sha256'], record['path']


def main():
    target = OUT / 'independent_audit.json'
    assert not target.exists()
    result = read(OUT / 'result.json'); identity = read(OUT / 'identity.json')
    preflight = read(OUT / 'preflight.json')
    assert result['state'] == 'exact_old_replay_final_inference_complete'
    checked(result['identity'])
    for record in identity['sources']:
        checked(record)
    assert preflight['identity'] == result['identity']
    assert preflight['rows'] == 2084659
    for key in ['raw_SQL_targets_membership_and_original_ID_order_exact',
                'basic18_inputs_and_classifier_membership_independent_SQL_exact',
                'all_five_stand_features_independent_SQL_exact']:
        assert preflight[key]
    model_hashes = {}
    for part, rounds, features in [('ordinary', 800, 83), ('classifier', 300, 54)]:
        record = read(OUT / f'{part}_trained.json')
        assert record['identity'] == result['identity']
        for key in ['model', 'preflight']:
            checked(record[key])
        assert record['trees'] == rounds and len(record['features']) == features
        model_hashes[part] = record['model']['sha256']
    refs = read(ROOT / 'data/derived/scored_submission_cache_20261011/manifest.json')
    h = identity['dayshift_history']
    assert h['rows'] == 19 and h['wrong_date'] == 11 and h['taxi_sec'] == 1080
    late_mean = identity['parameters']['late_mean']
    checks = {}
    with duckdb.connect(config={'threads': 1, 'memory_limit': '512MB'}) as con:
        con.execute("SET TimeZone='UTC'")
        con.execute('CREATE TABLE airline_history(airline VARCHAR,wrong DOUBLE,n DOUBLE)')
        con.executemany('INSERT INTO airline_history VALUES(?,?,?)', [(key, *value) for key, value in h['airline'].items()])
        for view, versions in result['views'].items():
            source = ROOT / 'data' / ('ranking.parquet' if view == 'old' else 'final_ranking.parquet')
            template = ROOT / 'data' / ('submitting.parquet' if view == 'old' else 'final_submitting.parquet')
            con.read_parquet(str(source)).create_view('raw', replace=True)
            con.read_parquet(str(template)).create_view('template', replace=True)
            con.read_parquet(str(ROOT / versions['30']['prediction']['path'])).create_view('base', replace=True)
            con.execute(f"""CREATE OR REPLACE TEMP TABLE expected AS
                WITH r AS (SELECT b.*,r.ADEP_mvt airport,r.FLIGHT_ID_mvt IS NULL noid,
                    r.FLIGHT_mvt flight,r.MVT_TIME_UTC_mvt t,
                    epoch(r.MVT_TIME_UTC_mvt-r.SCHED_TIME_UTC_mvt) gap,
                    epoch(r.MVT_TIME_UTC_mvt-r.LOBT_flt) centre,
                    date_diff('day',timezone('Europe/Rome',cast(r.SCHED_TIME_UTC_mvt AS TIMESTAMPTZ))::DATE,
                        timezone('Europe/Rome',cast(r.MVT_TIME_UTC_mvt AS TIMESTAMPTZ))::DATE) days,
                    CASE WHEN length(r.FLIGHT_mvt)<3 THEN NULL
                        WHEN regexp_full_match(substr(r.FLIGHT_mvt,1,3),'[A-Za-z]{{3}}') THEN substr(r.FLIGHT_mvt,1,3)
                        ELSE substr(r.FLIGHT_mvt,1,2) END airline,
                    coalesce(length(r.FLIGHT_mvt)=8 AND regexp_full_match(substr(r.FLIGHT_mvt,1,3),'[A-Za-z]{{3}}')
                        AND (substr(r.FLIGHT_mvt,4,1)=substr(r.FLIGHT_mvt,3,1)
                          OR substr(r.FLIGHT_mvt,4,2)=substr(r.FLIGHT_mvt,2,2)
                          OR substr(r.FLIGHT_mvt,4,3)=substr(r.FLIGHT_mvt,1,3)),false) mangled
                    FROM base b JOIN raw r ON b.mvt_id=r.MVT_ID_mvt WHERE r.PHASE_mvt='DEP'),
                flags AS (SELECT *,coalesce(airport='LIRF' AND noid AND gap>3600 AND gap<14400,false) late,
                    coalesce(airport='LIRF' AND noid AND gap>=14400,false) fired,
                    coalesce(airport='LIRF' AND noid AND gap>=50400 AND days>=1,false) hedged FROM r),
                mix AS (SELECT *,CASE WHEN airport='LIRF' AND isfinite(gap)
                    THEN greatest(0,p*gap+(1-p)*(CASE WHEN late THEN least(cast({late_mean!r} AS DOUBLE),greatest(gap-3600,60)) ELSE ordinary END))
                    ELSE ordinary END mix FROM flags),
                v30 AS (SELECT *,CASE WHEN fired THEN gap ELSE mix END e30 FROM mix),
                v33 AS (SELECT *,CASE WHEN NOT noid AND isfinite(centre)
                    THEN greatest(0,least(greatest(e30,centre-3660),centre+3660)) ELSE greatest(0,e30) END e33 FROM v30),
                v34 AS (SELECT *,least(e33,
                    CASE WHEN noid AND airport NOT IN ('LIRF','LTFM') AND gap>4500 THEN gap-3600 ELSE 'Infinity'::DOUBLE END,
                    CASE WHEN noid AND airport<>'LIRF' AND gap>1800 THEN gap ELSE 'Infinity'::DOUBLE END) e34 FROM v33),
                prior AS (SELECT *,CASE WHEN mangled
                    THEN (cast({h['format']['mangled'][0]} AS DOUBLE)+2.0*(({h['wrong_date']}+1.0)/({h['rows']}+2.0)))/({h['format']['mangled'][1]}+2.0)
                    ELSE (cast({h['format']['clean'][0]} AS DOUBLE)+2.0*(({h['wrong_date']}+1.0)/({h['rows']}+2.0)))/({h['format']['clean'][1]}+2.0) END format_prior FROM v34),
                prob AS (SELECT prior.*,(coalesce(ah.wrong,0)+2.0*format_prior)/(coalesce(ah.n,0)+2.0) wrong_p
                    FROM prior LEFT JOIN airline_history ah USING(airline))
                SELECT *,CASE WHEN hedged THEN wrong_p*(days*86400+{h['taxi_sec']})+(1-wrong_p)*gap ELSE e34 END e37 FROM prob""")
            for version, report in versions.items():
                checked(report['prediction'])
                con.read_parquet(str(ROOT / report['prediction']['path'])).create_view('p', replace=True)
                n, unique, bad, max_error = con.execute(f"""SELECT count(*),count(distinct p.mvt_id),
                    sum(CASE WHEN NOT isfinite(p.pred) OR NOT isfinite(p.submitted)
                        OR NOT isfinite(p.ordinary) OR NOT isfinite(p.p) OR p.p<0 OR p.p>1
                        OR (e.airport<>'LIRF' AND p.p<>0) OR p.submitted<0
                        OR abs(p.pred-e.e{version})>1e-8 OR abs(p.blended-e.mix)>1e-8
                        OR p.submitted IS DISTINCT FROM round_even(greatest(0,e.e{version}),0)
                        OR p.rule_triggered IS DISTINCT FROM e.fired OR p.late_rows IS DISTINCT FROM e.late
                        OR p.month IS DISTINCT FROM strftime(e.t,'%Y-%m')
                        OR p.ordinary IS DISTINCT FROM e.ordinary OR p.p IS DISTINCT FROM e.p
                        THEN 1 ELSE 0 END),max(abs(p.pred-e.e{version}))
                    FROM p JOIN expected e USING(mvt_id) JOIN template t ON p.mvt_id=t.MVT_ID_mvt""").fetchone()
                assert n == unique == report['rows'] == (344841 if view == 'old' else 670790)
                assert bad == 0, (view, version, bad, max_error)
                months = dict(con.execute('SELECT month,count(*) FROM p GROUP BY month ORDER BY month').fetchall())
                want = {'2026-01': 152719, '2026-07': 192122} if view == 'old' else {
                    '2026-01': 152719, '2026-02': 144158, '2026-06': 181791, '2026-07': 192122}
                assert months == want
                if view == 'old':
                    ref = refs[version]; assert sha(ROOT / ref['file']) == ref['sha256']
                    con.read_parquet(str(ROOT / ref['file'])).create_view('original', replace=True)
                    old_n, old_bad = con.execute('SELECT count(*),sum(CASE WHEN p.submitted IS DISTINCT FROM o.TAXITIME_SEC_mvt THEN 1 ELSE 0 END) FROM p JOIN original o ON p.mvt_id=o.MVT_ID_mvt').fetchone()
                    assert old_n == 344841 and old_bad == 0
                checks[view+'_v'+version] = dict(rows=n, max_SQL_float_error=float(max_error),
                    all_integer_rounding_rules_IDs_dates_exact=True, original_integer_vector_exact=(view=='old'))
    target.write_text(json.dumps(dict(status='pass', source_sha256=sha(__file__),
        result_sha256=sha(OUT / 'result.json'), checks=checks, model_hashes=model_hashes,
        caveats=['All original integer vectors exact; original model text identity is reported separately.',
                 'Eight threads are the original validation and CLI default; historical full-year CLI override was not recorded.',
                 'Model reloads were independently executed by producer; SQL auditor independently reconstructs all postprocessing.',
                 'Remaining temporal/weather/queue/handler input values rely on unchanged original source and prior cached-input audits.']), indent=2)+'\n')
    print(json.dumps(dict(status='pass', rows=sum(v['rows'] for v in checks.values()))))


if __name__ == '__main__':
    main()
