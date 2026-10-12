"""Independent SQL audit of final prediction identities/populations/domains."""
import hashlib,json
from pathlib import Path
import duckdb

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'VALIDATION/runs/v3_20261012_final_additional_public_components'


def read(p):return json.loads(p.read_text())


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()


def art(p):return dict(path=str(p.relative_to(ROOT)),sha256=sha(p))


def verify(a):
    p=ROOT/a['path'];assert sha(p)==a['sha256'];return p


def main():
    assert not (OUT/'audit.json').exists()
    runs=ROOT/'VALIDATION/runs';records=[];views=[]
    prefixes=['v3_20261012_final_public_ple_inference','v3_20261012_final_daily_main_inference']
    seen=set()
    for prefix in prefixes:
        folder=runs/prefix;a=read(folder/'old.json');b=read(folder/'final.json')
        assert a['identity']==b['identity'];identity=read(verify(a['identity']))
        assert a.get('all_old_predictions_exact',a.get('all_original_predictions_exact'))
        assert b.get('all_final_reloads_exact',b.get('all_model_reload_predictions_exact'))
        assert a['rows']==344841 and b['rows']==670790
        for source in identity['sources']:
            if source['path'] not in seen:verify(source);seen.add(source['path'])
        records += [art(folder/'old.json'),art(folder/'final.json')]
        for item in b.get('predictions',[b.get('prediction')]):views.append((prefix,verify(item)))
    assert len(views)==4
    checks=[]
    with duckdb.connect(config={'threads':1,'memory_limit':'1GB'}) as con:
        con.execute("SET TimeZone='UTC'")
        con.read_parquet(str(ROOT/'data/final_submitting.parquet')).create_view('template')
        con.read_parquet(str(ROOT/'data/final_ranking.parquet')).create_view('raw')
        assert con.execute('select count(*),count(distinct MVT_ID_mvt) from template').fetchone()==(670790,670790)
        for name,path in views:
            con.read_parquet(str(path)).create_view('p',replace=True)
            columns=[r[0] for r in con.execute('describe p').fetchall()]
            assert columns[0]=='mvt_id' and len(columns)==2;value=columns[1]
            n,u=con.execute('select count(*),count(distinct mvt_id) from p').fetchone();assert n==u==670790
            bad=con.execute(f'''select count(*) from p full join template t on p.mvt_id=t.MVT_ID_mvt
                where p.mvt_id is null or t.MVT_ID_mvt is null or p.{value} is null
                or not isfinite(p.{value}) or p.{value}<0''').fetchone()[0]
            assert bad==0
            if value=='p':
                assert con.execute('select count(*) from p join raw r on p.mvt_id=r.MVT_ID_mvt where p.p>1 or (r.ADEP_mvt!=\'LIRF\' and p.p!=0)').fetchone()[0]==0
            months=dict(con.execute('''select strftime(r.MVT_TIME_UTC_mvt,'%Y-%m'),count(*)
                from p join raw r on p.mvt_id=r.MVT_ID_mvt where r.PHASE_mvt='DEP' group by1'''.replace('by1','by 1')).fetchall())
            assert months=={'2026-01':152719,'2026-02':144158,'2026-06':181791,'2026-07':192122}
            bounds=con.execute(f'select min({value}),max({value}) from p').fetchone()
            checks.append(dict(component=name,prediction=art(path),rows=n,months=months,range=list(bounds)))
    OUT.mkdir(parents=True,exist_ok=True)
    result=dict(status='pass',old_controls_and_final_reloads_exact=True,source_hashes_checked=len(seen),
        components=checks,receipts=records,total_prediction_values=4*670790,
        all_SQL_template_membership_and_domains_exact=True,
        interpretation='Audited components only; no complete affine-stack port or final file is claimed.',
        no_fit=True,no_upload=True)
    (OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','source_hashes_checked','total_prediction_values']}))


if __name__=='__main__':main()
