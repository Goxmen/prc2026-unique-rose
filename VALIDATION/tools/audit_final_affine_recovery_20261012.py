"""Independent algebra/SQL checks for the label-free portability diagnostic."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'VALIDATION/runs/v3_20261012_final_affine_recovery_diagnostic'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    identity=read(OUT/'identity.json');result=read(OUT/'result.json')
    assert sha(OUT/'identity.json')==result['identity_sha256']
    for path,h in identity['sources'].items():assert sha(ROOT/path)==h,path
    assert not (OUT/'independent_audit.json').exists()
    paths={71:ROOT/'VALIDATION/runs/v3_20261011_physical_affine/submission_v71/geometry.json',
           73:ROOT/'VALIDATION/runs/v3_20261011_expanded_affine/submission_v73/geometry.json',
           79:ROOT/'VALIDATION/runs/v3_20261012_target205_candidate/submission_v79/manifest.json'}
    affine={v:{int(k):w for k,w in read(p)['weights'].items()} for v,p in paths.items()}
    # Expand by graph substitution, separately from the producer's closed-form
    # effective coefficient formula; keep all nonlinear increments symbolically.
    parents={80:79,78:77,77:76,76:74,74:73,72:71}
    expression=defaultdict(float,{80:1.})
    for v in [80,79,78,77,76,74,73,72,71]:
        scale=expression.pop(v,0.)
        if v in affine:
            expression['rounding_'+str(v)]+=scale
            for p,w in affine[v].items():expression[p]+=scale*w
        else:
            expression['delta_'+str(v)]+=scale
            expression[parents[v]]+=scale
    leaf={v:w for v,w in expression.items() if isinstance(v,int)}
    np.testing.assert_allclose([leaf[v] for v in sorted(leaf)],
        [result['effective_leaf_weights'][str(v)] for v in sorted(leaf)],atol=1e-14,rtol=0)
    for v,w in result['stage_delta_coefficients'].items():np.testing.assert_allclose(w,expression['delta_'+v],atol=1e-14,rtol=0)
    for v,w in result['rounding_coefficients'].items():np.testing.assert_allclose(w,expression['rounding_'+v],atol=1e-14,rtol=0)
    missing=result['missing_versions'];predictors=list(map(int,result['all_ranking_beta']))
    teacher=' + '.join(f'({leaf[v]!r})*(b.v{v}::DOUBLE-b.v38::DOUBLE)' for v in missing)
    def pred(beta):
        return ' + '.join(f'({beta[str(v)]!r})*(b.v{v}::DOUBLE-b.v38::DOUBLE)' for v in predictors)
    checks={}
    with duckdb.connect(config={'threads':1,'memory_limit':'512MB'}) as con:
        con.execute("SET TimeZone='UTC'")
        base=ROOT/'VALIDATION/runs/v3_20261012_target205_geometry'
        con.read_parquet(str(base/'ranking_basis.parquet')).create_view('basis')
        for v in [66,68]:
            con.read_parquet(str(ROOT/f'submissions/unique-rose_v{v}.parquet')).create_view(f's{v}')
        con.sql('SELECT basis.*,s66.TAXITIME_SEC_mvt v66,s68.TAXITIME_SEC_mvt v68 FROM basis JOIN s66 ON basis.mvt_id=s66.MVT_ID_mvt JOIN s68 ON basis.mvt_id=s68.MVT_ID_mvt').create_view('b')
        con.read_parquet(str(OUT/'ranking_fidelity.parquet')).create_view('f')
        expected=pred(result['all_ranking_beta'])
        n,u,teacher_error,prediction_error=con.execute(f'''SELECT count(*),count(distinct b.mvt_id),
            max(abs(f.missing_teacher-({teacher}))),max(abs(f.ridge_prediction-({expected})))
            FROM b JOIN f USING(mvt_id)''').fetchone()
        assert n==u==344841 and teacher_error<1e-7 and prediction_error<1e-7
        checks['ranking_all_rows']=dict(rows=n,teacher_max_error=teacher_error,prediction_max_error=prediction_error)
        for label,record in result['monthly_transfer_fidelity'].items():
            train,held=label.split('->');p=pred(record['beta'])
            rms=con.execute(f"SELECT sqrt(avg(pow(({p})-({teacher}),2))) FROM b JOIN f USING(mvt_id) WHERE f.month=?",[held]).fetchone()[0]
            np.testing.assert_allclose(rms,record['ridge_residual_error']['rms'],atol=1e-8,rtol=0)
            # Ridge stationarity independently aggregated from the actual
            # training month. The held month never enters fitting statistics.
            gradients=[]
            for v in predictors:
                g=con.execute(f"SELECT avg((b.v{v}::DOUBLE-b.v38::DOUBLE)*(({p})-({teacher}))) FROM b JOIN f USING(mvt_id) WHERE f.month=?",[train]).fetchone()[0]
                gradients.append(g+record['beta'][str(v)])
            assert max(abs(np.asarray(gradients)))<1e-6
            checks[label]=dict(held_rms=rms,maximum_training_normal_equation_error=max(abs(np.asarray(gradients))))
        con.read_parquet(str(base/'cv_basis.parquet')).create_view('cvbasis')
        con.read_parquet(str(ROOT/'VALIDATION/runs/v3_20261011_physical_candidate/single/cv/predictions.parquet')).create_view('cv66')
        con.read_parquet(str(ROOT/'VALIDATION/runs/v3_20261011_calls_only_candidate/submission_v68/cv.parquet')).create_view('cv68')
        # Explicitly project away taxi labels from all views used below.
        columns=','.join('cvbasis.v'+str(v) for v in leaf if v not in [66,68])
        con.sql(f'SELECT cvbasis.mvt_id,cvbasis.month,{columns},cv66.submitted v66,cv68.submitted v68 FROM cvbasis JOIN cv66 USING(mvt_id) JOIN cv68 USING(mvt_id)').create_view('b',replace=True)
        con.read_parquet(str(OUT/'historical_fidelity.parquet')).create_view('f',replace=True)
        n,u,teacher_error,prediction_error=con.execute(f'''SELECT count(*),count(distinct b.mvt_id),
            max(abs(f.missing_teacher-({teacher}))),max(abs(f.ridge_prediction-({expected})))
            FROM b JOIN f USING(mvt_id)''').fetchone()
        assert n==u==344419 and teacher_error<1e-7 and prediction_error<1e-7
        checks['historical_all_rows']=dict(rows=n,teacher_max_error=teacher_error,prediction_max_error=prediction_error)
        for month,record in result['historical_prediction_only_fidelity'].items():
            rms=con.execute(f'SELECT sqrt(avg(pow(({expected})-({teacher}),2))) FROM b WHERE month=?',[month]).fetchone()[0]
            np.testing.assert_allclose(rms,record['ridge_residual_error']['rms'],atol=1e-8,rtol=0)
    audit=dict(status='pass',source_sha256=sha(__file__),result_sha256=sha(OUT/'result.json'),
        independent_symbolic_nested_expansion=True,checks=checks,
        scope='Portability fidelity only; no new taxi-out score or candidate is supported')
    (OUT/'independent_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(dict(status='pass',prediction_rows=344841+344419)))


if __name__=='__main__':main()
