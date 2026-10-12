"""Input-only census of mechanical runway-key mismatches in public taxi joins.

Fixed before reading counts: strip whitespace, uppercase, remove a leading zero
from a single-digit runway designator, preserving L/C/R. No runway reassignment,
labels, predictions, models, calibration, or downloads. Ambiguous source keys
are excluded rather than summed. One SQL thread with a256MB memory bound.
"""
import hashlib
import json
from pathlib import Path
import duckdb

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'VALIDATION/runs/v3_20261012_public_runway_key_census'
SOURCE=ROOT/'data/external/eurocontrol_daily_taxi_20261011/daily_runway.parquet'
VIEWS={name:ROOT/f'VALIDATION/runs/v3_20261011_public_daily_taxi_features/inputs{name}.parquet'
       for name in ['2025','ranking']}


def art(path):
    p=Path(path)
    return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    assert not (OUT/'design.json').exists() and not (OUT/'result.json').exists(),'Preserve one fixed census'
    OUT.mkdir(parents=True,exist_ok=True)
    design=dict(hypothesis='Literal runway formatting may hide already available public taxi cells',
        transform='trim uppercase; remove zero only for ^0[1-9][LCR]?$',
        gate='Describe recoverable unambiguous source joins only; no predictive claim',
        ambiguity='Discard canonical source keys with more than one raw source row',
        sources=[art(p) for p in [Path(__file__),SOURCE,*VIEWS.values()]],
        no_labels=True,no_predictions=True,no_fit=True,no_download=True)
    write(OUT/'design.json',design)
    reports={}
    with duckdb.connect(config={'threads':1,'memory_limit':'256MB'}) as con:
        con.execute("CREATE MACRO canonical(x) AS CASE WHEN regexp_full_match(upper(trim(x)), '0[1-9][LCR]?') THEN substr(upper(trim(x)),2) ELSE upper(trim(x)) END")
        con.read_parquet(str(SOURCE)).create_view('source')
        con.execute('CREATE TEMP TABLE normalized AS SELECT airport,day,canonical(runway) run_key,count(*) n FROM source GROUP BY ALL')
        collisions=con.execute('SELECT count(*) FROM normalized WHERE n>1').fetchone()[0]
        for name,path in VIEWS.items():
            con.read_parquet(str(path)).create_view('inputs',replace=True)
            con.execute('''CREATE OR REPLACE TEMP TABLE joined AS
                SELECT i.mvt_id,i.airport,i.local_day,i.runway,
                s.runway IS NOT NULL original_match,
                coalesce(n.n=1,false) canonical_match,coalesce(n.n>1,false) ambiguous
                FROM inputs i LEFT JOIN source s ON i.airport=s.airport AND i.local_day=s.day AND i.runway=s.runway
                LEFT JOIN normalized n ON i.airport=n.airport AND i.local_day=n.day AND canonical(i.runway)=n.run_key''')
            rows,unique,original,canonical,recovered,ambiguous=con.execute('''SELECT count(*),count(distinct mvt_id),
                count(*) FILTER(WHERE original_match),count(*) FILTER(WHERE canonical_match),
                count(*) FILTER(WHERE NOT original_match AND canonical_match),count(*) FILTER(WHERE ambiguous)
                FROM joined''').fetchone()
            expected=2085047 if name=='2025' else 344841
            assert rows==unique==expected
            by_airport=con.execute('''SELECT airport,count(*) row_count,count(*) FILTER(WHERE NOT original_match AND canonical_match) recovered_count
                FROM joined GROUP BY airport ORDER BY airport''').fetchall()
            cases=con.execute('''SELECT airport,runway,canonical(runway) normalized_runway,count(*) row_count
                FROM joined WHERE NOT original_match AND canonical_match GROUP BY ALL ORDER BY row_count DESC,airport,runway''').fetchall()
            reports[name]=dict(rows=rows,original_matches=original,canonical_matches=canonical,
                recoverable_rows=recovered,ambiguous_rows=ambiguous,
                by_airport=[dict(zip(['airport','rows','recoverable_rows'],r)) for r in by_airport],
                recoverable_keys=[dict(zip(['airport','runway','normalized_runway','rows'],r)) for r in cases])
    write(OUT/'result.json',dict(design=art(OUT/'design.json'),views=reports,source_key_collisions=collisions,
        no_label_or_prediction_read=True,no_fit=True,
        conclusion='No formatting-only missing join exists; close without model work' if
        sum(r['recoverable_rows'] for r in reports.values())==0 else 'Source-join discrepancy only; independent audit required before any model decision'))
    print(json.dumps(dict(recovered={k:v['recoverable_rows'] for k,v in reports.items()},source_collisions=collisions)))


if __name__=='__main__':main()
