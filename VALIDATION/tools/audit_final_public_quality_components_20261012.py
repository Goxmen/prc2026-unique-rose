"""Independent SQL quality-transport audit, without original inference helpers.

Checks all matched/recovered projections, daily cells and source gates, integer
rounding and old production stages. No model fit, posterior or prediction run.
"""
import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT/'VALIDATION/runs'
OUT = RUNS/'v3_20261012_final_public_quality_components'
DIRECT = RUNS/'v3_20261012_final_public_direct_components'
CORRECTIONS = RUNS/'v3_20261012_final_public_corrections'
INPUTS = RUNS/'v3_20261012_final_correction_inputs'
ORIGINAL = RUNS/'v3_20261011_public_daily_taxi_submit'
ARMS = ['airport', 'cat', 'main', 'tabm']
AIRPORTS = ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LEMD', 'LFPG', 'LIRF', 'LSZH', 'LTFM']
MONTHS = {'old': {'2026-01':152719,'2026-07':192122},
    'final': {'2026-01':152719,'2026-02':144158,'2026-06':181791,'2026-07':192122}}
RAW = {'old':dict(path='data/ranking.parquet',sha256='9db75e3a56fa09b87aab72e191a53da3fe763d9887ad2b23aea9cde6a6325f86'),
    'final':dict(path='data/final_ranking.parquet',sha256='4b0692a792846079516a2749e034b0abda55df72c213eabdcda52a4d1f6bc69f')}


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def art(path):
    path=Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)),sha256=sha(path))


def verify(record):
    path=(ROOT/record['path']).resolve();assert path.is_relative_to(ROOT)
    assert sha(path)==record['sha256'],record['path']
    return path


def constrain(expression):
    # Exact original order: matched plan clip, nonnegative clip, then exclusion
    # from the recovered open interval. Ties use its nonnegative lower bound.
    matched=f'''greatest(0.,case when d.has_id and isfinite(d.nm_gap)
        then greatest(0.,least(greatest(({expression}),d.nm_gap-3606.),d.nm_gap+3606.))
        else ({expression}) end)'''
    lo,hi='(w.recovered_gap-3606.)','(w.recovered_gap+3606.)'
    return f'''case when isfinite(w.recovered_gap) and ({matched})>{lo} and ({matched})<{hi}
        then case when {lo}>=0 and ({matched})-{lo}<={hi}-({matched}) then {lo} else {hi} end
        else ({matched}) end'''


def preflight(view):
    folder=OUT/view;identity=read(folder/'identity.json');result=read(folder/'result.json')
    assert identity['view']==result['view']==view and identity['source']['path']=='VALIDATION/final_public_quality_components.py'
    verify(identity['source'])
    for record in identity['sources']:verify(record)
    assert result['identity']==art(folder/'identity.json') and result['old_replay_exact']
    assert result['no_fit'] and result['no_submission'] and identity['no_new_choice']
    assert result['rows']==identity['rows']==sum(MONTHS[view].values())
    assert result['months']==MONTHS[view]
    assert result['old_values_checked']==(26*344841 if view=='old' else 0)
    assert result['prediction']==art(folder/'stages.parquet') and result['cells']==art(folder/'cells.parquet')
    original=read(ORIGINAL/'result.json');parameters=read(ORIGINAL/'parameters.json')
    assert art(ORIGINAL/'parameters.json') in original['artifacts']
    assert parameters['allowed']==[f'2025-{m:02d}' for m in range(1,13)]
    assert len(parameters['coefficients'])==20 and set(parameters['noise'])==set(AIRPORTS)
    direct_audit=read(DIRECT/'independent_audit.json')
    assert direct_audit['status']=='pass' and direct_audit['identity']==art(DIRECT/'identity.json')
    assert direct_audit['views'][view]['prediction']==identity['direct']
    correction_result=read(CORRECTIONS/view/'result.json');correction_audit=read(CORRECTIONS/view/'audit.json')
    assert correction_result['identity']==art(CORRECTIONS/'identity.json') and correction_audit['status']=='pass'
    assert correction_audit['result']==art(CORRECTIONS/view/'result.json')
    assert correction_audit['prediction']==correction_result['prediction']
    verify(correction_result['prediction'])
    prepared=read(INPUTS/view/'prepared.json');input_audit=read(INPUTS/'audit.json')
    assert prepared['identity']==input_audit['identity']==art(INPUTS/'identity.json') and input_audit['status']=='pass'
    for record in prepared['outputs']:verify(record)
    feature_directory=verify(identity['raw_inputs']).parent
    feature_audit=read(feature_directory/'audit.json')
    assert feature_audit['status']=='pass' and feature_audit['manifest']==art(feature_directory/'manifest.json')
    manifest=read(feature_directory/'manifest.json')
    assert manifest['records']['ranking' if view=='old' else 'final']['inputs']==identity['raw_inputs']
    verify(RAW[view])
    if view=='final':
        old=read(OUT/'old/audit.json');assert old['status']=='pass' and old['result']==art(OUT/'old/result.json')
    return identity,result,parameters


def audit(view):
    folder=OUT/view;assert not (folder/'audit.json').exists()
    identity,result,parameters=preflight(view);n=result['rows']
    sources={Path(r['path']).name:r for r in identity['sources']}
    daily=verify(sources['daily_runway.parquet']);monthly=verify(sources['taxi_out_additional_time_2026.csv'])
    original=read(ORIGINAL/'identity.json')
    assert all(original['sources'][sources[name]['path']]==sources[name]['sha256'] for name in ['daily_runway.parquet','taxi_out_additional_time_2026.csv'])
    template=ROOT/'data'/('submitting.parquet' if view=='old' else 'final_submitting.parquet')
    ranges={};calibration_errors={}
    with duckdb.connect(config={'threads':1,'memory_limit':'1GB','preserve_insertion_order':True}) as con:
        con.execute("SET TimeZone='UTC'")
        def bind(name,path,unique=True):
            con.read_parquet(str(path)).create_view(name)
            if unique:
                assert con.execute(f'select count(*),count(distinct mvt_id) from {name}').fetchone()==(n,n)
                if name!='f':
                    assert con.execute(f'''select count(*) from {name} full join f using(mvt_id)
                        where {name}.mvt_id is null or f.mvt_id is null''').fetchone()[0]==0
        bind('f',folder/'stages.parquet')
        bind('a',verify(identity['direct']))
        bind('s',CORRECTIONS/view/'stages.parquet')
        for alias,name in [('d','daily'),('m','monthly'),('c','clocks')]:bind(alias,INPUTS/view/(name+'.parquet'))
        bind('i',verify(identity['raw_inputs']))
        bind('matches',INPUTS/view/'matches.parquet',False)
        con.execute('create temp view w as select * from matches where accepted')
        assert con.execute('select count(*)=count(distinct mvt_id) from w').fetchone()[0]
        con.read_parquet(str(verify(RAW[view]))).create_view('raw')
        con.read_parquet(str(template)).create_view('template')
        assert con.execute('''select count(*) from (select row_number() over() as pos,mvt_id from f) x
            full join (select row_number() over() as pos,MVT_ID_mvt from template) t using(pos)
            where x.mvt_id is distinct from t.MVT_ID_mvt''').fetchone()[0]==0
        actual_months=dict(con.execute('''select strftime(r.MVT_TIME_UTC_mvt,'%Y-%m'),count(*)
            from f join raw r on f.mvt_id=r.MVT_ID_mvt where r.PHASE_mvt='DEP' group by 1''').fetchall())
        assert actual_months==MONTHS[view]
        con.read_csv(str(monthly),header=True).create_view('source_monthly')
        con.execute('''create temp view q as select APT_ICAO as airport,
            cast(YEAR as varchar)||'-'||lpad(cast(MONTH_NUM as varchar),2,'0') as local_month,
            (cast(TF as double)-cast(VALID_FL as double))/cast(TF as double) as invalid
            from source_monthly''')
        assert con.execute('''select count(*) from f join i using(mvt_id) join m using(mvt_id)
            left join q on f.airport=q.airport and substr(f.local_day,1,7)=q.local_month
            where f.airport is distinct from i.airport or f.runway is distinct from i.runway
            or f.local_day is distinct from i.local_day or f.month is distinct from strftime(i.takeoff,'%Y-%m')
            or f.eligible is distinct from m.eligible or f.invalid_share is distinct from q.invalid
            or f.quality_eligible is distinct from coalesce(q.invalid<.05,false)''').fetchone()[0]==0
        numeric=['v76_float','v76']+[arm+'_'+key for arm in ARMS for key in ['refined_float','refined','shift','applied','bounded','pred']]
        bad=' or '.join(f'f."{k}" is null or not isfinite(f."{k}")' for k in numeric)
        assert con.execute('select count(*) from f where '+bad).fetchone()[0]==0
        joins='from f join a using(mvt_id) join s using(mvt_id) join d using(mvt_id) left join w using(mvt_id)'
        expected=constrain('cast(s.v74 as double)+cast(a.v76_direct as double)-cast(a.control as double)')
        assert con.execute('select count(*) '+joins+' where f.v76_float is distinct from ('+expected+') or f.v76 is distinct from round_even(greatest(0.,f.v76_float),0)').fetchone()[0]==0
        con.read_parquet(str(daily)).create_view('source_daily')
        con.read_parquet(str(folder/'cells.parquet')).create_view('actual_cells')
        assert {r[0] for r in con.execute('select distinct arm from actual_cells').fetchall()}==set(ARMS)
        con.execute('''create temp view z as select airport,day as local_day,runway,n as source_n,
            60.*total_min/n as source_mean from source_daily''')
        def airport_case(values):
            return 'case g.airport '+ ' '.join("when '"+a+"' then "+repr(float(v)) for a,v in zip(AIRPORTS,values))+' end'
        intercept=airport_case(parameters['coefficients'][:10])
        slope=airport_case(parameters['coefficients'][10:])
        noise=airport_case([parameters['noise'][a] for a in AIRPORTS])
        for arm in ARMS:
            prefix='f.'+arm+'_'
            transported=constrain('cast(s.refined74 as double)+cast(a.'+arm+'_direct as double)-cast(a.control as double)')
            bounded=constrain(prefix+'refined+'+prefix+'applied')
            bad=f'''{prefix}refined_float is distinct from ({transported})
                or {prefix}refined is distinct from round_even(greatest(0.,{prefix}refined_float),0)
                or {prefix}applied is distinct from case when f.quality_eligible then {prefix}shift else 0. end
                or {prefix}bounded is distinct from ({bounded})
                or {prefix}pred is distinct from round_even(greatest(0.,{prefix}bounded),0)'''
            assert con.execute('select count(*) '+joins+' where '+bad).fetchone()[0]==0
            con.execute(f'''create or replace temp view expected_cells as
                with g as (select airport,local_day,runway,count(*) as n,avg({arm}_refined) as mean
                    from f where eligible and {arm}_refined between 0 and 7200 group by 1,2,3),
                enriched as (select g.*,z.source_n,z.source_mean,
                    {intercept}+{slope}*(1.-cast(z.source_n as double)/g.n)/.1 as composition,
                    {noise} as noise,40000./g.n+225. as prior_var
                    from g left join z using(airport,local_day,runway))
                select *,coalesce(n>=20 and source_n>=20 and cast(source_n as double)/n between .5 and 1.2
                    and isfinite(source_mean),false) as supported,
                    prior_var/(prior_var+noise*noise) as gain,source_mean+composition-mean as raw_gap,
                    case when n>=20 and source_n>=20 and cast(source_n as double)/n between .5 and 1.2
                    and isfinite(source_mean) then prior_var/(prior_var+noise*noise)*
                    least(120.,greatest(-120.,source_mean+composition-mean)) else 0. end as shift
                from enriched''')
            saved_count,saved_unique=con.execute('''select count(*),count(distinct (airport,local_day,runway))
                from actual_cells where arm=?''',[arm]).fetchone()
            assert saved_count==saved_unique==con.execute('select count(*) from expected_cells').fetchone()[0]
            fixed=' or '.join('q.'+k+' is distinct from e.'+k for k in ['n','source_n','supported','noise'])
            numeric_cells=['mean','source_mean','composition','prior_var','gain','raw_gap','shift']
            numeric_bad=' or '.join(f'''((q.{k} is null) is distinct from (e.{k} is null))
                or (q.{k} is not null and (not isfinite(q.{k}) or abs(q.{k}-e.{k})>1e-9))''' for k in numeric_cells)
            assert con.execute('''select count(*) from (select * from actual_cells where arm=?) q
                full join expected_cells e using(airport,local_day,runway)
                where q.airport is null or e.airport is null or '''+fixed+' or '+numeric_bad,[arm]).fetchone()[0]==0
            # Only two composition terms are nonzero. BLAS and scalar SQL may
            # associate these differently; this fixed absolute algebra bound
            # is far below the original one-second serialization precision.
            error=con.execute(f'''select max(abs(f.{arm}_shift-case when f.eligible and f.{arm}_refined between 0 and 7200
                then coalesce(e.shift,0.) else 0. end)) from f left join expected_cells e using(airport,local_day,runway)''').fetchone()[0]
            assert error is not None and error<=1e-9,(view,arm,error)
            calibration_errors[arm]=float(error)
            assert con.execute(f'select count(*) from f where abs({arm}_shift)>120 or {arm}_pred<0').fetchone()[0]==0
            ranges[arm]=con.execute(f'select min({arm}_pred),max({arm}_pred) from f').fetchone()
        old_count=0
        if view=='old':
            for name,record in identity['controls'].items():bind('o_'+name,verify(record))
            for output,column in [('v76_float','pred'),('v76','submitted')]:
                assert con.execute(f'select count(*) from f join o_v76 o using(mvt_id) where f.{output} is distinct from o.{column}').fetchone()[0]==0
                old_count+=n
            for arm in ARMS:
                for key in ['refined_float','refined','shift','applied','bounded','pred']:
                    column={'shift':'unrestricted_shift','applied':'applied_shift','pred':'submitted'}.get(key,key) if arm=='airport' else ('tabm_' if arm=='tabm' else 'candidate_')+key
                    assert con.execute(f'select count(*) from f join o_{arm} o using(mvt_id) where f.{arm}_{key} is distinct from o.{column}').fetchone()[0]==0
                    old_count+=n
                assert con.execute(f'select count(*) from f join o_{arm} o using(mvt_id) where f.quality_eligible is distinct from o.quality_eligible').fetchone()[0]==0
            assert con.execute('select count(*) from f join o_airport o using(mvt_id) where f.invalid_share is distinct from o.invalid_share').fetchone()[0]==0
    assert old_count==result['old_values_checked']
    value=dict(status='pass',view=view,source=art(Path(__file__)),identity=art(folder/'identity.json'),
        result=art(folder/'result.json'),prediction=result['prediction'],rows=n,months=actual_months,
        old_stage_values_exact=old_count,all_SQL_transport_projection_rounding_exact=True,
        fixed_full2025_calibration_exact=art(ORIGINAL/'parameters.json'),
        all_row_independent_daily_calibration_max_abs=calibration_errors,calibration_algebra_tolerance=1e-9,
        all_template_ID_order_months_and_quality_gates_exact=True,ranges=ranges,
        no_fits_model_inference_or_posterior_execution=True,no_submission=True)
    (folder/'audit.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({k:value[k] for k in ['status','view','rows','old_stage_values_exact','all_row_independent_daily_calibration_max_abs']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--view',choices=['old','final'],required=True)
    audit(parser.parse_args().view)
