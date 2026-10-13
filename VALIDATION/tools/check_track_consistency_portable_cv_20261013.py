"""Synthetic secondary-portability SQL/bootstrap/guard checks; no actual data.

Run once before freeze. Uses only fabricated arrays and AST-extracted guard
functions. It never opens any experimental prediction, checkpoint or label file.
"""
import ast
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import duckdb
import numpy as np
import pandas as pd

import audit_track_consistency_portable_cv_20261013 as audit

OUT = audit.ROOT/'VALIDATION/reports/cohort_learning_20261013/track_consistency_portable_fixture.json'


def function(path, name, namespace):
    tree = ast.parse(Path(path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def rejected(callback):
    try:
        callback()
    except AssertionError:
        return
    raise AssertionError('corruption was not rejected')


def scalar_constraint(values, raw, recovered):
    answer = []
    for value, row, gap in zip(values, raw.itertuples(), recovered):
        value = float(value)
        if row.has_id and np.isfinite(row.nm_gap):
            value = min(max(value, row.nm_gap-3606), row.nm_gap+3606)
        value = max(value, 0)
        if np.isfinite(gap) and gap-3606 < value < gap+3606:
            lower, upper = gap-3606, gap+3606
            value = lower if lower >= 0 and value-lower <= upper-value else upper
        answer.append(np.rint(max(value, 0)))
    return np.array(answer)


def suffix_fixture():
    rows = 96
    months = np.repeat(audit.MONTHS, 48)
    within = np.tile(np.arange(48), 2)
    airports = np.where(within < 40, 'EDDF', 'EGLL')
    days = [month+('-03' if at < 20 else '-01' if at < 40 else '-02') for month, at in zip(months, within)]
    frame = pd.DataFrame(dict(mvt_id=np.arange(rows), month=months, airport=airports,
        M=650.+within, direct_control=500.+within, robust_active=within%3 != 0,
        joint_increment=(within%4-1)*3., robust_delta=np.where(within%3 != 0, within%5-2., 0.)))
    raw = frame[['mvt_id', 'airport']].assign(has_id=False, nm_gap=np.nan)
    raw.loc[0, ['has_id', 'nm_gap']] = [True, -3500.]
    recovered = np.full(rows, np.nan)
    recovered[[1, 2]] = [200., 4200.]
    matches = pd.DataFrame(dict(mvt_id=frame.mvt_id, recovered_gap=recovered))
    inputs = frame[['mvt_id', 'airport']].assign(local_day=days, runway='09')
    monthly = frame[['mvt_id', 'airport']].assign(eligible=True)
    monthly.loc[5, 'eligible'] = False
    sources = inputs[['airport', 'local_day', 'runway']].drop_duplicates().copy()
    sources['source_n'] = [20 if day.endswith('03') else 21 if day.endswith('01') else 8 for day in sources.local_day]
    sources['source_mean'] = [900. if day.endswith('03') else 600. for day in sources.local_day]
    ms = pd.DataFrame(dict(airport=['EDDF', 'EDDF'], local_month=audit.MONTHS, invalid=[.02, .05]))
    params = {month: dict(coefficients=list(np.linspace(-2., 2., 20)),
        noise={airport: float(10+i) for i, airport in enumerate(audit.historical.AIRPORTS)},
        allowed=audit.historical.FOLDS[month]) for month in audit.MONTHS}
    cells = []
    for ai, arm in enumerate(audit.ALL):
        direct = frame.direct_control.to_numpy() + ai*1.5
        nraw = frame.M.to_numpy()+direct-frame.direct_control.to_numpy()
        n = scalar_constraint(nraw, raw, recovered)
        shift = np.zeros(rows)
        for month in audit.MONTHS:
            p = params[month]
            eligible = (months == month) & monthly.eligible.to_numpy() & (n >= 0) & (n <= 7200)
            groups = inputs.loc[eligible].assign(pred=n[eligible]).groupby(['airport', 'local_day', 'runway'], sort=True)
            for (airport, day, runway), group in groups:
                source = sources[(sources.airport == airport) & (sources.local_day == day) & (sources.runway == runway)].iloc[0]
                count, mean = len(group), float(np.mean(group.pred))
                index = audit.historical.AIRPORTS.index(airport)
                composition = p['coefficients'][index]+p['coefficients'][index+10]*(1-source.source_n/count)/.1
                noise = p['noise'][airport]
                prior = 40000/count+225
                gain = prior/(prior+noise**2)
                supported = bool(count >= 20 and source.source_n >= 20 and .5 <= source.source_n/count <= 1.2 and np.isfinite(source.source_mean))
                gap = source.source_mean+composition-mean
                delta = gain*np.clip(gap, -120, 120) if supported else 0.
                shift[group.index] = delta
                cells.append(dict(arm=arm, fold=month, airport=airport, local_day=day, runway=runway,
                    n=count, mean=mean, source_n=source.source_n, source_mean=source.source_mean,
                    composition=composition, noise=noise, prior_var=prior, supported=supported,
                    gain=gain, raw_gap=gap, shift=delta))
        invalid = np.where(airports == 'EDDF', np.where(months == '2025-01', .02, .05), np.nan)
        quality = invalid < .05
        applied = np.where(quality, shift, 0.)
        q = scalar_constraint(n+applied, raw, recovered)
        j = scalar_constraint(q+.5*frame.joint_increment.to_numpy(), raw, recovered)
        final = scalar_constraint(j+.5*frame.robust_delta.to_numpy(), raw, recovered)
        values = dict(N_raw=nraw, N=n, quality_eligible=quality, invalid_share=invalid,
            daily_shift=shift, applied_shift=applied, Q=q, J=j, submitted=final, direct=direct)
        for key, value in values.items():
            frame[arm+'_'+key] = value
    cells = pd.DataFrame(cells)
    with duckdb.connect(config={'threads': 1, 'memory_limit': '128MB'}) as con:
        for key, value in [('d', raw), ('w', matches), ('i', inputs), ('m', monthly), ('ds', sources), ('ms', ms)]:
            con.register(key, value)
        result = audit.suffix_checks(con, frame, cells, params)
        broken = frame.copy()
        broken.loc[1, 'target_consistency_Q'] += 1
        rejected(lambda: audit.suffix_checks(con, broken, cells, params))
        broken_cells = cells.copy()
        broken_cells.loc[0, 'source_n'] += 1
        rejected(lambda: audit.suffix_checks(con, frame, broken_cells, params))
    return dict(rows=rows, arms=len(result), cells=len(cells), qualified_duplicate_airport_joins_pass=True,
        independent_scalar_constraints_and_daily_cells=True, corrupted_stage_and_cell_rejected=True,
        strict_quality_edge_and_missing_quality_checked=True)


def metric_fixture():
    frame = pd.DataFrame(dict(mvt_id=np.arange(8),month=['2025-01']*4+['2025-07']*4,
        day=['03','01','03','02','02','01','02','03'],y=[1.5,4.25,7.75,8.25,3.5,1.5,5.25,8.75],
        v82=[0.,4.,9.,12.,2.,4.,8.,13.],supervised_control=[0.,3.,9.,11.,1.,3.,8.,12.],
        training_consistency=[0.,4.,8.,12.,2.,4.,7.,11.],target_consistency=[2.,4.,8.,8.,4.,2.,5.,9.]))
    columns = ['v82', *audit.ARMS]
    scores = {month:{name:float(np.sqrt(np.mean((group[name]-group.y)**2))) for name in columns}
              for month, group in frame.groupby('month')}
    cv = {name:float(np.mean([scores[month][name] for month in audit.MONTHS])) for name in columns}
    paired = {}
    for reference in audit.REFERENCES:
        random = np.random.default_rng(0)
        draws = np.zeros(2000)
        for month in audit.MONTHS:
            group = frame[frame.month.eq(month)]
            clusters = [group[group.day.eq(day)] for day in dict.fromkeys(group.day)]
            selections = random.integers(0,len(clusters),size=(2000,len(clusters)))
            for k, chosen in enumerate(selections):
                # Oracle resamples actual fabricated rows, not SQL daily sums.
                truth = np.concatenate([clusters[int(at)].y.to_numpy() for at in chosen])
                target = np.concatenate([clusters[int(at)].target_consistency.to_numpy() for at in chosen])
                other = np.concatenate([clusters[int(at)][reference].to_numpy() for at in chosen])
                draws[k] += (np.sqrt(np.mean((target-truth)**2))-np.sqrt(np.mean((other-truth)**2)))/2
        interval = np.percentile(draws,[2.5,97.5])
        paired[reference] = dict(change=cv['target_consistency']-cv[reference],ci95=interval.tolist(),
            decision='improvement' if interval[1]<0 else 'regression' if interval[0]>0 else 'inconclusive')
    gate = dict(primary=True,mean_improves=True,both_months=True,interval_below_zero=True,passed=True)
    result = dict(scores=scores,cv=cv,paired=paired,gate=gate)
    with duckdb.connect(config={'threads':1, 'memory_limit':'128MB'}) as con:
        audit.metrics(con,frame,result)
        bad = deepcopy(result);bad['gate']['passed'] = False
        rejected(lambda: audit.metrics(con,frame,bad))
        bad = deepcopy(result);bad['paired']['v82']['ci95'][1] += .01
        rejected(lambda: audit.metrics(con,frame,bad))
    return dict(rows=len(frame),bootstrap_replicates=2000,references=3,
        unsorted_first_occurrence_days_and_fractional_truth=True,raw_row_cluster_bootstrap_exact=True,
        corrupted_gate_and_interval_rejected=True)


def guard_fixture():
    def forbidden(*args, **kwargs):
        raise RuntimeError('unexpected real-data decode or numerical expansion')
    records = {}
    def fake_art(path):
        return {'path':str(path),'sha256':'synthetic'}
    members = [{'path':f'{month}/{arm}','sha256':'synthetic'} for month in audit.MONTHS for arm in audit.ARMS]
    with tempfile.TemporaryDirectory(prefix='portable_guard_fixture_') as directory:
        out = Path(directory)
        primary_out, candidate_out = out/'primary', out/'candidate'
        manifest = out/'identity.json'
        records[str(out/'allocation.json')] = dict(authority='/root',identity=fake_art(manifest),state='released_for_fixed_portable_consistency_cv')
        records[str(primary_out/'result.json')] = dict(identity=fake_art(primary_out/'identity.json'),gate={'passed':False})
        records[str(primary_out/'independent_audit.json')] = dict(status='pass',result=fake_art(primary_out/'result.json'),identity=fake_art(primary_out/'identity.json'),gate={'passed':False})
        writes = {}
        namespace = dict(guard=lambda:{}, OUT=out, MANIFEST=manifest, read=lambda path:records[str(path)],
            art=fake_art, collect_fit_receipts=lambda:({},members),primary=SimpleNamespace(OUT=primary_out),
            candidate=SimpleNamespace(OUT=candidate_out),write=lambda path,value:writes.setdefault(path.name,value),
            portable=SimpleNamespace(load=forbidden),pd=SimpleNamespace(read_parquet=forbidden,read_pickle=forbidden),print=lambda *a,**k:None)
        score = function(audit.PRODUCER,'score',namespace)
        score()
        assert set(writes)=={'closed.json'}
        assert writes['closed.json']['new_prediction_vectors_decoded'] is False
        assert writes['closed.json']['members']==members
        saved = []
        links = dict(identity=fake_art(manifest),members=members)
        with patch.object(audit,'OUT',out),patch.object(audit,'provenance',lambda:({},None,{},links)),patch.object(audit,'art',fake_art),patch.object(audit,'write',lambda path,value:saved.append(value)),patch.object(audit.pd,'read_parquet',forbidden),patch.object(audit.pd,'read_pickle',forbidden):
            audit.audit()
        assert len(saved)==1 and saved[0]['status']=='pass_closed_primary_gate_failed'
        assert saved[0]['no_new_prediction_decode'] is True
        # Execute the actual producer's six-export gate against synthetic receipts.
        candidate = SimpleNamespace(OUT=candidate_out,MANIFEST=out/'train_identity.json',
            paths=lambda month,arm:(None,None,Path(f'{month}/{arm}')),
            completed=lambda month,arm,identity:{'month':month,'arm':arm})
        training = dict(status='pass',identity=fake_art(candidate.MANIFEST),members=members)
        reads = {str(candidate.OUT/'audit.json'):training,str(candidate.MANIFEST):{}}
        collect = function(audit.PRODUCER,'collect_fit_receipts',dict(candidate=candidate,MONTHS=audit.MONTHS,ARMS=audit.ARMS,art=fake_art,read=lambda path:reads[str(path)]))
        assert len(collect()[1])==6
        training['members']=members[:-1]
        rejected(collect)
        training['members']=members
        candidate.completed=lambda month,arm,identity:None
        rejected(collect)
    return dict(producer_primary_failure_closes_before_decode=True,auditor_closed_route_before_decode=True,
        all_six_exports_required=True,missing_member_and_missing_completion_rejected=True)


def pool_fixture():
    rng = np.random.default_rng(42)
    source = rng.uniform(0,1e5,size=(3,257)).astype(np.float32)
    source[:,0] = [1e8,1.,1.]
    arrays = [pd.Series(part).to_numpy(float) for part in source]
    explicit = (arrays[0]+arrays[1]+arrays[2])/3.
    native = np.mean(arrays,axis=0)
    np.testing.assert_array_equal(explicit,native)
    calls, tree = arrays[:2]
    np.testing.assert_array_equal(.355*(calls+explicit)+.29*tree,.71*.5*(calls+native)+.29*tree)
    assert np.any(source.mean(axis=0).astype(float)!=explicit), 'fixture must detect premature float32 mean'
    return dict(rows=257,float64_conversion_and_left_to_right_three_pool_exact=True,
        native_coefficient_operation_exact=True,premature_float32_pool_distinguished=True)


def main():
    assert not OUT.exists()
    report = dict(status='pass',source=audit.art(__file__),auditor=audit.art(audit.__file__),
        producer=audit.art(audit.PRODUCER),suffix=suffix_fixture(),metrics=metric_fixture(),
        guards=guard_fixture(),pool=pool_fixture(),no_actual_labels_vectors_models_read=True,
        no_fit_no_inference_no_network=True)
    with OUT.open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status='pass',receipt=audit.art(OUT))))


if __name__=='__main__':
    main()
