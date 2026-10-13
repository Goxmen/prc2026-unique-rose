"""Source-only paired-view clock diagnostic; no departure targets or predictions.

Determine which public fields changed for shared movement IDs. Visible arrival
labels can describe the published transform, but cannot prove how hidden final
departure labels were generated. This is not a model or candidate evaluation.
"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'VALIDATION'))
import duckdb
from records import write_json,sha256

OUT=ROOT/'VALIDATION/reports/final_shared_clocks_20261013'


def main():
    assert not (OUT/'result.json').exists()
    paths={v:ROOT/'data'/f'{v}.parquet' for v in ['ranking','final_ranking']}
    OUT.mkdir(parents=True,exist_ok=True)
    write_json(OUT/'identity.json',dict(source=dict(path=str(Path(__file__).relative_to(ROOT)),sha256=sha256(__file__)),
        inputs={k:dict(path=str(p.relative_to(ROOT)),sha256=sha256(p)) for k,p in paths.items()},
        scope='All shared IDs, field changes by phase, visible arrival target transform only',
        no_departure_target=True,no_model=True,no_candidate=True,no_selection=True))
    with duckdb.connect(config={'threads':1,'memory_limit':'1GB'}) as con:
        con.execute("SET TimeZone='UTC'")
        for name,path in paths.items():
            con.read_parquet(str(path)).create_view(name)
        columns=[r[0] for r in con.execute('describe ranking').fetchall()]
        assert columns==[r[0] for r in con.execute('describe final_ranking').fetchall()]
        def qi(s):return '"'+s.replace('"','""')+'"'
        rows=con.execute('select count(*),count(distinct MVT_ID_mvt) from ranking').fetchone()
        joined=con.execute('select count(*) from ranking a join final_ranking b using(MVT_ID_mvt)').fetchone()[0]
        assert rows[0]==rows[1]==joined==689534
        fields=[]
        for column in columns:
            for phase,count,changed in con.execute(f'''select a.PHASE_mvt,count(*),
                count(*) filter(where a.{qi(column)} is distinct from b.{qi(column)})
                from ranking a join final_ranking b using(MVT_ID_mvt) group by 1 order by 1''').fetchall():
                fields.append(dict(field=column,phase=phase,rows=count,changed=changed))
        clocks=[]
        for column in ['MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','SCHED_TIME_UTC_mvt']:
            for phase,n,lo,hi,mean,rms,qs in con.execute(f'''select phase,count(delta),min(delta),max(delta),avg(delta),
                sqrt(avg(delta*delta)),quantile_cont(delta,[0.01,0.5,0.99]) from
                (select a.PHASE_mvt phase,epoch(b.{qi(column)})-epoch(a.{qi(column)}) delta
                from ranking a join final_ranking b using(MVT_ID_mvt)) group by phase order by phase''').fetchall():
                clocks.append(dict(field=column,phase=phase,nonnull_pairs=n,min=lo,max=hi,mean=mean,rms=rms,quantiles=qs))
        arrivals=con.execute('''select count(*) as n_rows,count(dy) target_pairs,
            count(*) filter(where dy!=0) target_changes,
            count(*) filter(where db!=0) block_changes,
            count(*) filter(where dy is not null and abs(dy-(db-dt))>0.000001) duration_identity_changes,
            sqrt(avg(dy*dy)) target_delta_rms,sqrt(avg(pow(dy-(db-dt),2))) identity_residual_rms
            from (select cast(b.TAXITIME_SEC_mvt as double)-a.TAXITIME_SEC_mvt dy,
              epoch(b.MVT_TIME_UTC_mvt)-epoch(a.MVT_TIME_UTC_mvt) dt,
              epoch(b.BLOCK_TIME_UTC_mvt)-epoch(a.BLOCK_TIME_UTC_mvt) db
              from ranking a join final_ranking b using(MVT_ID_mvt) where a.PHASE_mvt='ARR')''').fetchdf().to_dict('records')[0]
        departure=con.execute('''select count(*),count(a.TAXITIME_SEC_mvt),count(b.TAXITIME_SEC_mvt),
            count(a.BLOCK_TIME_UTC_mvt),count(b.BLOCK_TIME_UTC_mvt)
            from ranking a join final_ranking b using(MVT_ID_mvt) where a.PHASE_mvt='DEP' ''').fetchone()
        assert departure== (344841,0,0,0,0)
    report=dict(identity=dict(path=str((OUT/'identity.json').relative_to(ROOT)),sha256=sha256(OUT/'identity.json')),
        shared_rows=joined,fields=fields,clocks=clocks,arrivals=arrivals,departure_hidden_values_all_null=True,
        conclusion='Descriptive published-source transform only; hidden departure label transform remains unobserved.',
        no_prediction_transfer=True,no_model=True,no_candidate=True,no_cv_claim=True)
    write_json(OUT/'result.json',report)
    print(dict(shared_rows=joined,changed_fields=[r for r in fields if r['changed']],clocks=clocks,arrivals=arrivals))


if __name__=='__main__':main()
