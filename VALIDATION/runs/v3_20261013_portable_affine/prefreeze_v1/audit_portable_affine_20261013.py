"""Independent portable-affine output/SQL/geometry audit. Never solves weights."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import duckdb
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT/'VALIDATION/runs'
OUT = RUNS/'v3_20261013_portable_affine'
BASE = RUNS/'v3_20261012_target205_geometry'
V80 = RUNS/'v3_20261012_tabm_pool3_other_runway_combo'
RAW_SHA = {'old':'9db75e3a56fa09b87aab72e191a53da3fe763d9887ad2b23aea9cde6a6325f86',
           'final':'4b0692a792846079516a2749e034b0abda55df72c213eabdcda52a4d1f6bc69f'}
MONTHS = {'old':['2026-01','2026-07'],'final':['2026-01','2026-02','2026-06','2026-07']}


def read(path):
    return json.loads(Path(path).read_text())


def art(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8388608),b''):
            digest.update(block)
    return dict(path=str(path.relative_to(ROOT)),sha256=digest.hexdigest())


def verify(record):
    path = ROOT/record['path']
    assert art(path) == {k:record[k] for k in ['path','sha256']}, path
    if 'bytes' in record:
        assert path.stat().st_size == record['bytes']
    return path


def includes(tree, record):
    if isinstance(tree,dict):
        return (tree.get('path')==record['path'] and tree.get('sha256')==record['sha256']) or any(includes(v,record) for v in tree.values())
    return isinstance(tree,list) and any(includes(v,record) for v in tree)


def qi(name):
    return '"'+name.replace('"','""')+'"'


def close(a,b,atol=1e-8,rtol=1e-11):
    np.testing.assert_allclose(a,b,atol=atol,rtol=rtol)


def component(record, view, ids):
    frame = pd.read_parquet(verify(record[view]),columns=[record['id_column'],record['value_column']])
    np.testing.assert_array_equal(frame[record['id_column']],ids)
    assert frame[record['id_column']].is_unique
    value = frame[record['value_column']].to_numpy(float)
    assert np.isfinite(value).all() and (value>=0).all()
    np.testing.assert_array_equal(value,np.rint(value))
    return value


def geometry_audit(matrix, scores, versions, weights, solution, receipt, result):
    assert np.isfinite(weights).all() and abs(weights.sum()-1)<1e-8
    assert np.all(weights>=-3-1e-8) and np.all(weights<=4+1e-8)
    assert solution['constraints'] == dict(lower=-3.,upper=4.,ridge_mse=1.)
    raw = matrix@weights
    # Direct affine MSE identity, independent of the solver's reference form.
    expanded = float(np.dot(weights,scores*scores-(matrix*matrix).mean(axis=0)) + np.mean(raw*raw))
    assert expanded>=0
    close(expanded,receipt['expanded_mse'],atol=1e-7)
    close(np.sqrt(expanded),solution['projected_rmse'])
    close(result['projected_old_rmse'],np.sqrt(expanded))
    delta = matrix-matrix[:,versions.index(64),None]
    gram = delta.T@delta/len(matrix)
    reference = scores[versions.index(64)]**2
    covariance = (scores*scores-reference-np.diag(gram))/2
    close(reference+2*covariance@weights+weights@gram@weights,expanded,atol=1e-7)
    initial = np.zeros(len(versions));initial[versions.index(64)]=1
    gradient = 2*(covariance+gram@weights+weights-initial)
    lower = weights<=-3+1e-7
    upper = weights>=4-1e-7
    free = ~(lower|upper)
    assert free.any()
    residual = gradient-gradient[free].mean()
    residual[lower] = np.minimum(residual[lower],0)
    residual[upper] = np.maximum(residual[upper],0)
    # Ridge=1 gives a fixed strong-convexity optimality certificate; no re-solve.
    maximum_objective_gap = float(residual@residual/4)
    assert maximum_objective_gap<0.01, ('KKT certificate exceeds fixed0.01 MSE tolerance',maximum_objective_gap)
    radius = float(np.abs(weights)@(np.abs(scores)*1e-4+2.5e-9))
    interval = [np.sqrt(max(expanded-radius,0)),np.sqrt(expanded+radius)]
    close(solution['score_rounding_rmse_interval'],interval)
    close(solution['coefficient_l1'],abs(weights).sum())
    assert solution['negative_predictions']==int((raw<0).sum())
    close(solution['minimum_prediction'],raw.min())
    close(solution['maximum_prediction'],raw.max())
    return dict(expanded_mse=expanded,projected_rmse=float(np.sqrt(expanded)),
                score_rounding_interval=interval,kkt_objective_gap_upper_mse=maximum_objective_gap,
                no_optimizer_or_target_reconstruction=True)


def output_audit(con, view, frame, basis, versions, weights, ids):
    np.testing.assert_array_equal(frame.mvt_id,ids)
    assert frame.mvt_id.is_unique and np.isfinite(frame[['raw','submitted']]).all().all()
    assert (frame.submitted>=0).all() and (frame.submitted<=np.iinfo(np.int32).max).all()
    con.register('output',frame)
    con.register('basis',basis)
    expression = ' + '.join(f'CAST(b.v{v} AS DOUBLE)*({float(w)!r})' for v,w in zip(versions,weights))
    count,raw_difference,bad_round,round_order_differences = con.execute(f'''
      select count(*),max(abs(o.raw-({expression}))),
        sum(case when o.submitted!=round_even(greatest(o.raw,0.),0) then 1 else 0 end),
        sum(case when o.submitted!=round_even(greatest(({expression}),0.),0) then 1 else 0 end)
      from output o join basis b using(mvt_id)''').fetchone()
    assert count==len(ids) and raw_difference<1e-7 and bad_round==0
    if view!='cv':
        raw_path = ROOT/'data'/('ranking.parquet' if view=='old' else 'final_ranking.parquet')
        assert art(raw_path)['sha256']==RAW_SHA[view]
        con.read_parquet(str(raw_path)).create_view('raw_movements',replace=True)
        n,bad = con.execute('''select count(*),sum(case when r.PHASE_mvt!='DEP'
            or r.TAXITIME_SEC_mvt is not null then 1 else 0 end)
            from output o join raw_movements r on o.mvt_id=r.MVT_ID_mvt''').fetchone()
        assert n==len(ids) and bad==0
        months = [x[0] for x in con.execute("select distinct strftime(MVT_TIME_UTC_mvt,'%Y-%m') from raw_movements order by 1").fetchall()]
        assert months==MONTHS[view]
    return dict(rows=count,sql_max_raw_difference=raw_difference,serialization_exact=True,
                sql_sum_rounding_order_differences=round_order_differences,
                rounding_scope='Stored float64 dot-product is serialized exactly; independent SQL sum differs by <1e-7, with near-half order differences reported, never patched.')


def cv_audit(con, frame, result, identity):
    con.register('cv',frame.assign(row_position=np.arange(len(frame))))
    paths=[]
    for month in ['2025-01','2025-07']:
        selected=[s for s in identity['sources'] if Path(s['path']).name.startswith(f'training_{month}-01_')]
        assert len(selected)==1
        paths.append(str(ROOT/selected[0]['path']))
    con.read_parquet(paths).create_view('truth')
    n,bad = con.execute('''select count(*),sum(case when c.y is distinct from r.TAXITIME_SEC_mvt
       or c.month is distinct from strftime(r.MVT_TIME_UTC_mvt,'%Y-%m')
       or c.day is distinct from strftime(r.MVT_TIME_UTC_mvt,'%Y-%m-%d')
       or c.airport is distinct from r.ADEP_mvt or r.PHASE_mvt!='DEP' then 1 else 0 end)
       from cv c join truth r on c.mvt_id=r.MVT_ID_mvt''').fetchone()
    assert n==344419 and bad==0
    scores = con.execute('''select month,sqrt(avg(pow(v70-y,2))) v70,
      sqrt(avg(pow(v80-y,2))) v80,sqrt(avg(pow(submitted-y,2))) submitted
      from cv group by month order by month''').fetchdf().set_index('month')
    assert list(scores.index)==['2025-01','2025-07']
    for month,row in scores.iterrows():
        for key in ['v70','v80','submitted']:
            close(row[key],result['months'][month][key],atol=1e-9)
    for key in ['v70','v80','submitted']:
        close(scores[key].mean(),result['cv'][key],atol=1e-9)
    close(scores.v80.mean(),289.7627782102906,atol=1e-9)
    day = con.execute('''select month,day,count(*) n,sum(pow(v70-y,2)) v70,
      sum(pow(v80-y,2)) v80,sum(pow(submitted-y,2)) submitted,min(row_position) first_row
      from cv group by month,day order by month,first_row''').fetchdf()
    paired={}
    for control in ['v70','v80']:
        rng=np.random.default_rng(0);draws=np.zeros(2000)
        for month in ['2025-01','2025-07']:
            block=day[day.month.eq(month)]
            index=rng.integers(0,len(block),size=(2000,len(block)))
            n=block.n.to_numpy()[index].sum(axis=1)
            draws+=(np.sqrt(block.submitted.to_numpy()[index].sum(axis=1)/n)-np.sqrt(block[control].to_numpy()[index].sum(axis=1)/n))/2
        ci=np.quantile(draws,[.025,.975]);change=float((scores.submitted-scores[control]).mean())
        expected=result['paired'][control]
        close(ci,expected['ci95'],atol=1e-9);close(change,expected['change'],atol=1e-9)
        decision='improvement' if ci[1]<0 else 'regression' if ci[0]>0 else 'inconclusive'
        assert expected['decision']==decision
        paired[control]=dict(change=change,ci95=ci.tolist(),decision=decision)
    return dict(raw_truth_rows=344419,paired=paired,reused_development=True)


def main():
    started=time.monotonic();destination=OUT/'independent_audit.json'
    assert not destination.exists(), 'Preserve completed audit'
    identity=read(OUT/'identity.json');result=read(OUT/'result.json')
    assert art(Path(__file__)) in identity['sources']
    assert identity['source']==art(ROOT/'VALIDATION/portable_affine_20261013.py')
    assert result['identity']==art(OUT/'identity.json') and result['no_upload'] and not result['selected_for_final']
    for record in identity['sources']:
        verify(record)
    assert identity['design']['reference']==64 and identity['design']['solve_count']==1
    catalog=read(verify(identity['catalog']));prepared=read(BASE/'prepared.json')
    versions=identity['versions']
    assert versions==[v for v in prepared['versions'] if str(v) in catalog['components']]
    assert len(versions)==len(set(versions))>=27 and 64 in versions and 70 in versions
    assert all(v not in versions for v in [62,66,68,75])
    assert identity['unavailable_from_retained_basis']==[v for v in prepared['versions'] if v not in versions]
    assert identity['excluded_from_catalog']==sorted(set(map(int,catalog['components']))-set(versions))
    assert read(BASE/'audit.json')['prepared']==art(BASE/'prepared.json')
    live=read(verify(identity['snapshot']));history=live['history']
    assert len({x['file'] for x in history})==len(history)
    assert identity['snapshot_at']==live['updatedAt']
    history={x['file']:x for x in history}
    for v in versions:
        rec=identity['components'][str(v)]
        assert rec==catalog['components'][str(v)]
        audit=read(verify(rec['audit']));assert audit['status']=='pass'
        upstream=read(verify(rec['result'])) if 'result' in rec else None
        for view in ['old','final']:
            verify(rec[view])
            direct=includes(audit,rec[view])
            chain=upstream is not None and (includes(audit,rec['result']) or audit.get('result_sha256')==rec['result']['sha256']) and includes(upstream,rec[view])
            assert direct or chain,(v,view)
        score=history[f'unique-rose_v{v}.parquet']
        assert score['status']=='Succeeded' and score['usedPairs']==344841
        assert float(score['score'])==identity['scores'][str(v)]
    receipt=read(verify(result['geometry']));assert receipt['identity']==art(OUT/'identity.json')
    solution=receipt['solution'];assert set(solution['weights'])==set(map(str,versions))
    weights=np.array([solution['weights'][str(v)] for v in versions])
    frames={view:pd.read_parquet(verify(record)) for view,record in result['predictions'].items()}
    assert set(frames)=={'old','cv','final'} and result['versions']==versions
    for view,count in [('old',344841),('cv',344419),('final',670790)]:
        assert len(frames[view])==result['rows'][view]==count
    old=pd.read_parquet(verify(identity['old_basis']),columns=['mvt_id']+[f'v{v}' for v in versions])
    old_ids=pq.read_table(ROOT/'data/submitting.parquet')['MVT_ID_mvt'].to_numpy()
    np.testing.assert_array_equal(old.mvt_id,old_ids)
    for v in versions:
        np.testing.assert_array_equal(old[f'v{v}'],component(identity['components'][str(v)],'old',old_ids))
    held=pd.read_parquet(verify(identity['cv_basis']),columns=['mvt_id','month','day','airport','y']+[f'v{v}' for v in versions])
    controls=pd.read_parquet(verify(identity['control_cv'])).set_index('mvt_id').loc[held.mvt_id]
    np.testing.assert_array_equal(frames['cv'][['mvt_id','month','day','airport','y','v70']],held[['mvt_id','month','day','airport','y','v70']])
    np.testing.assert_array_equal(frames['cv'].v80,controls.three_tabm_runway)
    assert frames['cv'].mvt_id.is_unique
    assert identity['control_cv']==read(V80/'result.json')['scored']
    production=RUNS/'v3_20261012_tabm_pool3_other_runway_combo_submit'
    paudit=read(production/'independent_audit.json');pres=read(production/'result.json');pid=read(production/'identity.json')
    assert paudit['status']=='pass' and paudit['identity']==pres['identity']==art(production/'identity.json')
    assert paudit['result']==art(production/'result.json')
    assert pid['cv']==pres['cv_source']==art(V80/'result.json')
    assert read(V80/'result.json')['identity']==art(V80/'identity.json')
    assert identity['control_cv'] in pid['sources']
    final_template=pq.read_table(ROOT/'data/final_submitting.parquet');final_ids=final_template['MVT_ID_mvt'].to_numpy()
    final=pd.DataFrame({'mvt_id':final_ids})
    for v in versions:
        final[f'v{v}']=component(identity['components'][str(v)],'final',final_ids)
    checks={}
    with threadpool_limits(limits=1),duckdb.connect(config={'threads':1,'memory_limit':'1GB'}) as con:
        con.execute("SET TimeZone='UTC'")
        geometry=geometry_audit(old[[f'v{v}' for v in versions]].to_numpy(float),
            np.array([identity['scores'][str(v)] for v in versions]),versions,weights,solution,receipt,result)
        for view,basis,ids in [('old',old,old_ids),('cv',held,held.mvt_id.to_numpy()),('final',final,final_ids)]:
            checks[view]=output_audit(con,view,frames[view],basis,versions,weights,ids)
        cv=cv_audit(con,frames['cv'],result,identity)
    serialized=frames['old'].submitted.to_numpy(float);raw=frames['old'].raw.to_numpy(float)
    distance=float(np.sqrt(np.mean((serialized-raw)**2)))
    close(distance,result['serialization_rms'])
    lo,hi=geometry['score_rounding_interval']
    close([max(0,lo-distance),hi+distance],result['conservative_old_rmse_interval'])
    candidate=pq.read_table(verify(result['final_candidate']))
    assert candidate.schema==final_template.schema and candidate['MVT_ID_mvt'].equals(final_template['MVT_ID_mvt'])
    for name in candidate.column_names:
        if name!='TAXITIME_SEC_mvt':
            assert candidate[name].equals(final_template[name])
    np.testing.assert_array_equal(candidate['TAXITIME_SEC_mvt'].to_numpy(),frames['final'].submitted)
    report=dict(status='pass',source=art(Path(__file__)),identity=art(OUT/'identity.json'),result=art(OUT/'result.json'),
        predictions=result['predictions'],final_candidate=result['final_candidate'],versions=versions,
        checks=checks,geometry=geometry,cv=cv,seconds=time.monotonic()-started,
        independent_of_old_v80_recipe=True,no_missing_component_substitution=True,no_fit=True,no_solve=True,no_upload=True,
        limitations=['Leaderboard-selected weights; projected score is not an official score or independent validation.',
                    'Jan/Jul reused development with original reconstruction caveat.',
                    'Final four-month predictions are algebraically audited; hidden final accuracy is unknown.'])
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(audit=art(destination),versions=versions,geometry=geometry)))


if __name__=='__main__':
    main()
