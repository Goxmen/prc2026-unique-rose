"""Light independent source/binding review; no model inference, fit or labels."""
from pathlib import Path
import hashlib
import json
import sys
import types

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'VALIDATION'))
import duckdb
import final_daily_main_inference as implementation


def art(path):
    path=Path(path)
    return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read(path):return json.loads(Path(path).read_text())


def main():
    out=ROOT/'VALIDATION/reports/final_daily_main_review_20261012.json'
    assert not out.exists()
    daily=implementation.daily
    clone=types.FunctionType(daily.audit.__code__,dict(daily.audit.__globals__,OUT=implementation.FEATURES),daily.audit.__name__)
    assert clone.__code__ is daily.audit.__code__
    assert clone.__globals__['__file__']==daily.__file__
    assert clone.__globals__['source'] is daily.source
    assert clone.__globals__['build'] is daily.build
    assert clone.__globals__['OUT']==implementation.FEATURES
    recipe=read(implementation.selected.MANIFEST)
    parent=read(implementation.trees.MANIFEST)
    model=read(implementation.selected.OUT/'model.json')
    original_prediction=read(implementation.selected.OUT/'predictions/old.json')
    assert model['identity']==original_prediction['identity']==art(implementation.selected.MANIFEST)
    assert model['model']==original_prediction['model']
    assert model['columns']==recipe['columns'] and model['rounds']==10000
    assert recipe['columns'][:-10]==parent['main_recipe']['columns']
    assert recipe['columns'][-10:]==daily.FEATURES
    parent_sources={r['path']:r['sha256'] for r in parent['sources']}
    for name in ['callsign_ranking','public_ranking']:
        item=recipe['inputs'][name]
        assert parent_sources[item['path']]==item['sha256']
    assert recipe['original_columns']==parent['original']['original_columns']
    zones='case r.ADEP_mvt '+' '.join(f"when '{a}' then '{z}'" for a,z in daily.source.ZONES.items())+' end'
    with duckdb.connect(config={'threads':1,'memory_limit':'512MB'}) as con:
        con.execute("SET TimeZone='UTC'")
        con.read_parquet(str(ROOT/'data/final_ranking.parquet')).create_view('r')
        con.read_parquet(str(ROOT/'data/final_submitting.parquet')).create_view('t')
        con.read_parquet(str(daily.source.SOURCE/'daily_runway.parquet')).create_view('s')
        assert con.execute("select count(*),count(distinct MVT_ID_mvt) from r where PHASE_mvt='DEP'").fetchone()==(670790,670790)
        assert con.execute("select count(*) from (select MVT_ID_mvt from r where PHASE_mvt='DEP') r full join t using(MVT_ID_mvt) where r.MVT_ID_mvt is null or t.MVT_ID_mvt is null").fetchone()[0]==0
        rows=con.execute(f'''with a as (select airport,day,sum(n) n from s group by airport,day)
            select strftime(r.MVT_TIME_UTC_mvt,'%Y-%m') AS month,count(*),
            sum((s.n is null)::int),sum((a.n is null)::int)
            from r left join s on r.ADEP_mvt=s.airport and r.RUNWAY_mvt=s.runway
                and strftime(r.MVT_TIME_UTC_mvt at time zone ({zones}),'%Y-%m-%d')=s.day
            left join a on r.ADEP_mvt=a.airport and strftime(r.MVT_TIME_UTC_mvt at time zone ({zones}),'%Y-%m-%d')=a.day
            where r.PHASE_mvt='DEP' group by 1 order by 1''').fetchall()
        assert [r[0] for r in rows]==['2026-01','2026-02','2026-06','2026-07']
        assert sum(r[1] for r in rows)==670790
    report=dict(status='pass_static_and_source_only',source=art(__file__),implementation=art(implementation.__file__),
        selected_model=model['model'],selected_identity=art(implementation.selected.MANIFEST),
        public_source=art(daily.source.SOURCE/'daily_runway.parquet'),
        rows=670790,coverage=[dict(zip(['month','rows','missing_runway_cells','missing_airport_cells'],r)) for r in rows],
        clone_same_code_and_original_source_globals=True,only_audit_output_path_rebound=True,
        selected184_column_order_and_retained174_parent_exact=True,old_prediction_record_model_identity_exact=True,
        final_template_ID_set_exact=True,missing_public_cells_retained_as_NaN=True,
        actual_inference_not_run=True,labels_not_read=True,
        pending='Require every old feature replay, old prediction equality, final feature SQL audit and all-row saved-model reload receipts from execution; this review is not those execution proofs.')
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
