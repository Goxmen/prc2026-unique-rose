"""Test the source-only opposite-clock perturbation mechanism on visible arrivals.

No model, departure labels, prediction transport, coefficient fitting or tuning.
The departure block clock is hidden; its transformation cannot be verified here.
"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'VALIDATION'))
import json
import duckdb
from records import sha256,write_json
OUT=ROOT/'VALIDATION/reports/final_shared_clocks_20261013'


def main():
    assert not (OUT/'coupling.json').exists()
    source=json.loads((OUT/'identity.json').read_text())
    for record in source['inputs'].values():
        assert sha256(ROOT/record['path'])==record['sha256']
    identity=dict(source=dict(path=str(Path(__file__).relative_to(ROOT)),sha256=sha256(__file__)),
        previous_source_diagnostic=dict(path=str((OUT/'result.json').relative_to(ROOT)),sha256=sha256(OUT/'result.json')),
        hypothesis='Arrival clock changes are opposite up to timestamp integer rounding',
        fixed_residual='delta_block + delta_movement',
        no_model=True,no_departure_targets=True,no_candidate=True)
    write_json(OUT/'coupling_identity.json',identity)
    with duckdb.connect(config={'threads':1,'memory_limit':'1GB'}) as con:
        con.execute("SET TimeZone='UTC'")
        for name,record in source['inputs'].items():
            con.read_parquet(str(ROOT/record['path'])).create_view(name)
        con.execute('''create temp table paired as select a.MVT_ID_mvt id,a.ADEP_mvt departure_airport,
            a.ADES_mvt arrival_airport,a.PHASE_mvt phase,
            epoch(b.MVT_TIME_UTC_mvt)-epoch(a.MVT_TIME_UTC_mvt) dt,
            epoch(b.BLOCK_TIME_UTC_mvt)-epoch(a.BLOCK_TIME_UTC_mvt) db,
            cast(b.TAXITIME_SEC_mvt as double)-a.TAXITIME_SEC_mvt dy
            from ranking a join final_ranking b using(MVT_ID_mvt)''')
        hist=con.execute('''select dt+db residual,count(*) n from paired where phase='ARR'
            group by residual order by residual''').fetchdf().to_dict('records')
        airports=con.execute('''select arrival_airport airport,count(*) n,corr(dt,db) correlation,
            sqrt(avg(pow(dt+db,2))) residual_rms,max(abs(dt+db)) maximum_absolute,
            count(*) filter(where dy is distinct from db-dt) label_identity_failures,
            count(*) filter(where abs(dt+db)>1) outside_one_second
            from paired where phase='ARR' group by arrival_airport order by arrival_airport''').fetchdf().to_dict('records')
        overall=con.execute('''select count(*) n,corr(dt,db) correlation,
            sqrt(avg(pow(dt+db,2))) residual_rms,max(abs(dt+db)) maximum_absolute,
            count(*) filter(where dy is distinct from db-dt) label_identity_failures,
            count(*) filter(where abs(dt+db)>1) outside_one_second
            from paired where phase='ARR' ''').fetchdf().to_dict('records')[0]
        dep=con.execute("select count(*) from paired where phase='DEP' and (db is not null or dy is not null)").fetchone()[0]
        assert dep==0
    write_json(OUT/'coupling.json',dict(identity=dict(path=str((OUT/'coupling_identity.json').relative_to(ROOT)),sha256=sha256(OUT/'coupling_identity.json')),
        overall=overall,airports=airports,residual_histogram=hist,
        conclusion='Visible arrival transform only. Opposite departure block-clock change remains an inference requiring explicit qualification.',
        no_model=True,no_candidate=True,no_departure_label_claim=True))
    print(dict(overall=overall,residual_histogram=hist,airports=len(airports)))


if __name__=='__main__':main()
