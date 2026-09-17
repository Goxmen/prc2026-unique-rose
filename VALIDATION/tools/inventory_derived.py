"""Verify notebook recovery caches without regenerating or using them for training."""

import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from records import ROOT, write_json, sha256
sys.path.insert(0, str(ROOT / 'pipeline'))
from recovery import PLAN_FIELDS


def main():
    con = duckdb.connect(config={'threads': 2})
    sources = [str(ROOT/'data/training_2025-*.parquet'), str(ROOT/'data/ranking.parquet')]
    con.read_parquet(sources, union_by_name=True).create_view('raw')
    result = {}
    for name in ['recovered_flt_blocks', 'recovered_flt_relaxed']:
        path = ROOT / 'data/derived' / (name+'.parquet')
        con.read_parquet(str(path)).create_view('cached', replace=True)
        equality = ' AND '.join(f'c.{x} IS NOT DISTINCT FROM a.{x}' for x in PLAN_FIELDS)
        sql = f'''SELECT count(*) n_rows,
            count(*) FILTER (d.MVT_ID_mvt IS NULL) missing_departures,
            count(*) FILTER (a.MVT_ID_mvt IS NULL) missing_donors,
            count(*) FILTER (c.FLIGHT_ID_recovered IS DISTINCT FROM a.FLIGHT_ID_mvt) donor_id_mismatches,
            count(*) FILTER (NOT ({equality})) payload_mismatches,
            count(*) FILTER (c.ADEP_flt IS DISTINCT FROM d.ADEP_mvt) origin_guard_rejections,
            count(*) FILTER (c.ADES_flt IS DISTINCT FROM d.ADES_mvt) destination_guard_rejections,
            count(*) FILTER (strftime(d.MVT_TIME_UTC_mvt,'%Y-%m') != strftime(a.MVT_TIME_UTC_mvt,'%Y-%m')) cross_month_donors
            FROM cached c LEFT JOIN raw d ON d.MVT_ID_mvt=c.MVT_ID_mvt
            LEFT JOIN raw a ON a.MVT_ID_mvt=c.recovered_from_arr_mvt_id'''
        item = con.sql(sql).df().iloc[0].to_dict()
        item['sha256'] = sha256(path)
        item['dataset_counts'] = dict(con.sql('SELECT dataset,count(*) FROM cached GROUP BY 1 ORDER BY 1').fetchall())
        item['claimed_by_other_departures'] = con.sql('''SELECT count(*) FROM cached c WHERE EXISTS (
            SELECT 1 FROM raw d WHERE d.PHASE_mvt='DEP' AND d.FLIGHT_ID_mvt=c.FLIGHT_ID_recovered
            AND d.MVT_ID_mvt != c.MVT_ID_mvt)''').fetchone()[0]
        if name.endswith('relaxed'):
            item['stages'] = con.sql('SELECT dataset,matcher_variant,count(*) n FROM cached GROUP BY 1,2 ORDER BY 1,2').df().to_dict('records')
        result[name] = item
        print(name, item, flush=True)
    for name in ['imputed_flt_static', 'rule_recovered', 'dest_recovered']:
        path = ROOT / 'data/derived' / (name+'.parquet')
        df = con.read_parquet(str(path)).df()
        result[name] = dict(rows=len(df), sha256=sha256(path), dataset_counts=df.groupby('dataset').size().to_dict())
        if name == 'imputed_flt_static':
            result[name]['low_confidence_flight_type'] = int(df.FLIGHT_TYPE_low_conf.fillna(False).sum())
            result[name]['low_confidence_aircraft_type'] = int(df.AIRCRAFT_TYPE_low_conf.fillna(False).sum())
        print(name, result[name], flush=True)
    write_json(ROOT / 'VALIDATION/reports/stage1_derived_inventory.json', result)


if __name__ == '__main__':
    main()
