"""Training-calibrated reconciliation with public local-day/runway taxi totals.

One fixed screen, not a parameter search. The published cohort is different
from the competition cohort. Its mean is an imperfect observation, not truth.
"""
import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

import public_refinements_submission_cv as parent
from records import write_json

ROOT = parent.ROOT
OUT = ROOT / 'VALIDATION/runs/v3_20261011_public_daily_taxi'
CACHE = ROOT / 'tmp/research_20261011/daily_taxi_source/training_rows.parquet'
SOURCE = ROOT / 'data/external/eurocontrol_daily_taxi_20261011'
REPORT = ROOT / 'VALIDATION/reports/research_network_20261010'
base = parent.original.tail.base
constraints = parent.original.constraints
ZONES = constraints.monthly.ZONES
KEY = ['airport', 'local_day', 'runway']
art, read, verify = parent.art, parent.read, parent.verify
DESIGN = dict(
    primary_parent='v76 plus selected airport/t5/full-monthly refinements, production rounding',
    secondary_parent='v76 with production rounding; same method without any retuning',
    source='EUROCONTROL public taxi-out dashboard; local takeoff day and runway; minutes to seconds',
    train_selection='stand/runway present, true taxi in [0,7200]; only allowed outer training months',
    inference_selection='stand/runway present, incoming prediction in [0,7200]; no labels',
    minimum_cell_rows=20, minimum_source_rows=20, source_to_prediction_count_range=[0.5,1.2],
    calibration='airport intercept and airport count-deficit coefficient; equal cells; Huber20; ridge0.001',
    noise='per-airport training residual MAD times1.4826; floor10 seconds',
    prior_mean_variance='200^2 / eligible cell count + 15^2; fixed existing daily prior scales',
    gain='prior mean variance / (prior mean variance + calibrated source noise variance)',
    correction='gain times source+composition minus predicted mean; raw gap capped at120 seconds',
    cap_provenance='same120-second raw cap as existing monthly stage; no cap/weight search',
    incomplete_training_days='require entire local day to lie in allowed UTC months',
    postprocessing='reapply matched/recovered plan bounds, then nonnegative nearest integer',
    limitations='Source cohort differs; true/predicted truncation differs; repeatedly reused Jan/Jul development',
    no_taxi_model_fit=True, no_production=True, no_upload=True)


def guard():
    assert read(parent.OUT/'audit.json')['status'] == 'pass'
    assert read(REPORT/'public_daily_taxi_source_audit.json')['status'] == 'pass'
    paths = [Path(__file__), Path(parent.__file__), Path(constraints.__file__),
        Path(constraints.exclusion.__file__), Path(constraints.monthly.__file__),
        SOURCE/'manifest.json', SOURCE/'daily_runway.parquet', CACHE,
        REPORT/'public_daily_taxi_source_audit.json', REPORT/'public_daily_taxi_population.json',
        parent.OUT/'scored.parquet', parent.OUT/'result.json', parent.OUT/'audit.json',
        base.OUT/'rows2025.parquet', constraints.exclusion.OUT/'matches2025.parquet']
    value = dict(design=DESIGN, folds=base.FOLDS, sources=[art(p) for p in paths])
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT/'identity.json'
    if p.exists():
        assert read(p) == value
    else:
        assert not list(OUT.glob('*.parquet'))
        write_json(p, value)
    return value


def rows():
    f = pd.read_parquet(CACHE, columns=['mvt_id','airport','takeoff','runway','stand','y','local_takeoff'])
    f = f.rename(columns={'local_takeoff':'local_day'})
    f['month'] = f.takeoff.dt.strftime('%Y-%m')
    f['eligible'] = f.stand.notna() & f.runway.notna()
    assert f.mvt_id.is_unique
    return f.drop(columns=['stand','takeoff'])


def public():
    f = pd.read_parquet(SOURCE/'daily_runway.parquet').rename(columns={'day':'local_day','n':'source_n'})
    assert not f.duplicated(KEY).any() and f.phase.eq('DEP').all()
    f['source_mean'] = 60*f.total_min/f.source_n
    return f[KEY+['source_n','source_mean']]


def supported(c):
    return (c.n.ge(20) & c.source_n.ge(20) & (c.source_n/c.n).between(.5,1.2)
            & np.isfinite(c.source_mean))


def matrix(c):
    one = np.column_stack([c.airport.eq(a).to_numpy(float) for a in ZONES])
    return np.column_stack([one, one*(1-c.source_n.to_numpy()/c.n.to_numpy())[:,None]/.1])


def complete_days(c, allowed):
    keep = np.zeros(len(c),bool)
    for a,z in ZONES.items():
        ix = np.flatnonzero(c.airport.eq(a))
        date = pd.DatetimeIndex(c.local_day.iloc[ix])
        start = date.tz_localize(z).tz_convert('UTC').strftime('%Y-%m')
        end = ((date+pd.Timedelta(days=1)).tz_localize(z)-pd.Timedelta(nanoseconds=1)).tz_convert('UTC').strftime('%Y-%m')
        keep[ix] = start.isin(allowed)&end.isin(allowed)
    return keep


def fit(train, source, allowed):
    assert set(train.month).issubset(allowed)
    eligible = train.eligible & train.y.between(0,7200)
    c = train.loc[eligible].groupby(KEY,as_index=False).agg(truth=('y','mean'),n=('y','size'))
    c = c.merge(source,on=KEY,how='inner',validate='one_to_one')
    c = c.loc[supported(c)&complete_days(c,allowed)].reset_index(drop=True)
    X = matrix(c); target = (c.truth-c.source_mean).to_numpy()
    fun = lambda b: np.r_[(X@b-target)/20, np.sqrt(.001)*b/20]
    result = least_squares(fun,np.zeros(20),loss='huber',f_scale=1.)
    assert result.success
    c['composition'] = X@result.x
    c['residual'] = target-c.composition
    noises = {}
    for a,q in c.groupby('airport'):
        noises[a] = max(10.,float(1.4826*np.median(abs(q.residual-q.residual.median()))))
        assert len(q)>=30
    assert set(noises)==set(ZONES)
    return dict(coefficients=result.x.tolist(),noise=noises,allowed=allowed,
        cells=len(c),fit_message=result.message),c


def apply(inputs, prediction, source, parameters):
    assert 'y' not in inputs
    p = np.asarray(prediction,float)
    f = inputs.assign(pred=p)
    eligible = f.eligible & f.pred.between(0,7200)
    c = f.loc[eligible].groupby(KEY,as_index=False).agg(mean=('pred','mean'),n=('pred','size'))
    c = c.merge(source,on=KEY,how='left',validate='one_to_one')
    c['supported'] = supported(c)
    c['composition'] = matrix(c)@np.asarray(parameters['coefficients'])
    c['noise'] = c.airport.map(parameters['noise'])
    c['prior_var'] = 200.**2/c.n+15.**2
    c['gain'] = c.prior_var/(c.prior_var+c.noise**2)
    c['raw_gap'] = c.source_mean+c.composition-c['mean']
    c['shift'] = np.where(c.supported,c.gain*c.raw_gap.clip(-120,120),0.)
    join = f[KEY].merge(c[KEY+['shift']],on=KEY,how='left',validate='many_to_one')
    shift = np.where(eligible,join['shift'].fillna(0),0.)
    assert np.isfinite(shift).all() and np.max(abs(shift))<=120
    return p+shift,c,shift


def constrain(raw, ids, month, predictions, matches):
    p = constraints.matched_bound(raw,predictions)
    for m,allowed in base.FOLDS.items():
        ix = np.flatnonzero(month.eq(m))
        permit = matches[matches.donor_month.isin(allowed+[m])]
        p[ix],_ = constraints.exclusion.apply(ids[ix],np.maximum(p[ix],0),permit)
    return np.rint(np.maximum(p,0))


def screen():
    guard(); assert not (OUT/'result.json').exists(), 'Preserve completed screen'
    train = rows(); source = public(); cv = pd.read_parquet(parent.OUT/'scored.parquet')
    ids = cv.mvt_id.to_numpy()
    aligned = train.set_index('mvt_id').loc[ids].reset_index()
    raw = pd.read_parquet(base.OUT/'rows2025.parquet').set_index('mvt_id').loc[ids].reset_index()
    np.testing.assert_array_equal(aligned.y,cv.y)
    np.testing.assert_array_equal(raw.y,cv.y)
    np.testing.assert_array_equal(aligned.month,cv.month)
    matches = pd.read_parquet(constraints.exclusion.OUT/'matches2025.parquet')
    f = cv[['mvt_id','month','day','y']].copy()
    f['airport'] = aligned.airport; f['local_day'] = aligned.local_day; f['runway'] = aligned.runway
    f['eligible'] = aligned.eligible
    parameters, calibration, cells = {},[],[]
    for month,allowed in base.FOLDS.items():
        parameters[month],c = fit(train[train.month.isin(allowed)],source,allowed)
        calibration.append(c.assign(fold=month))
    for name,column in [('v76','control'),('refined','candidate')]:
        p = cv[column].to_numpy(float)
        f[name] = p
        q = p.copy(); shifts = np.zeros(len(p))
        for month in base.FOLDS:
            ix = np.flatnonzero(cv.month.eq(month))
            q[ix],c,shifts[ix] = apply(aligned.iloc[ix].drop(columns='y'),p[ix],source,parameters[month])
            cells.append(c.assign(fold=month,arm=name))
        f[name+'_shift'] = shifts
        f[name+'_daily'] = constrain(raw,ids,cv.month,q,matches)
        # Postprocessing alone must preserve the serialized parent exactly.
        np.testing.assert_array_equal(constrain(raw,ids,cv.month,p,matches),p)
    columns = ['v76','v76_daily','refined','refined_daily']
    scores = {m:{c:float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in columns} for m,g in f.groupby('month')}
    paired = {name:parent.original.airport.parent.ts.paired_bootstrap(f.day,f.month,
        (f[name]-f.y)**2,(f[name+'_daily']-f.y)**2) for name in ['v76','refined']}
    f.to_parquet(OUT/'scored.parquet',index=False)
    pd.concat(calibration).to_parquet(OUT/'calibration.parquet',index=False)
    pd.concat(cells).to_parquet(OUT/'cells.parquet',index=False)
    write_json(OUT/'parameters.json',parameters)
    value = dict(design=DESIGN,identity=art(OUT/'identity.json'),scores=scores,
        cv={c:float(np.mean([v[c] for v in scores.values()])) for c in columns},paired=paired,
        changed_rows={n:int(f[n].ne(f[n+'_daily']).sum()) for n in ['v76','refined']},
        artifacts=[art(OUT/p) for p in ['scored.parquet','calibration.parquet','cells.parquet','parameters.json']])
    write_json(OUT/'result.json',value);print(value,flush=True)


def audit():
    guard(); assert not (OUT/'audit.json').exists(), 'Preserve completed audit'
    result=read(OUT/'result.json')
    for a in result['artifacts']: verify(a)
    # Independent SQL recomputes labels, local calendar, parents and scores.
    with duckdb.connect(config={'threads':1}) as con:
        for name,path in [('c',OUT/'scored.parquet'),('p',parent.OUT/'scored.parquet'),
                          ('r',CACHE),('s',SOURCE/'daily_runway.parquet'),('cal',OUT/'calibration.parquet')]:
            con.read_parquet(str(path)).create_view(name)
        zone = 'case r.airport '+ ' '.join(f"when '{a}' then '{z}'" for a,z in ZONES.items())+' end'
        n,u,errors=con.execute(f'''select count(*), count(distinct c.mvt_id),sum(case when
            c.y is distinct from r.y or c.month != strftime(r.takeoff AT TIME ZONE 'UTC','%Y-%m')
            or c.local_day != strftime(r.takeoff AT TIME ZONE ({zone}),'%Y-%m-%d')
            or c.eligible != (r.stand is not null and r.runway is not null)
            or c.v76 is distinct from p.control or c.refined is distinct from p.candidate
            then 1 else 0 end) from c join r using(mvt_id) join p using(mvt_id)''').fetchone()
        assert n==u==344419 and errors==0
        for column in result['cv']:
            values=con.execute(f'select month,sqrt(avg(pow({column}-y,2))) from c group by month').fetchall()
            for m,v in values: np.testing.assert_allclose(v,result['scores'][m][column],atol=1e-8,rtol=0)
            np.testing.assert_allclose(np.mean([v for m,v in values]),result['cv'][column],atol=1e-8,rtol=0)
        for fold,allowed in base.FOLDS.items():
            q=con.execute('''select cal.airport,cal.local_day,cal.runway,cal.n,cal.truth,
                count(*) as sql_n,avg(r.y) as sql_mean from cal join r
                on cal.airport=r.airport and cal.local_day=r.local_takeoff and cal.runway=r.runway
                where cal.fold=? and strftime(r.takeoff AT TIME ZONE 'UTC','%Y-%m') in
                (select unnest(?)) and r.stand is not null and r.runway is not null and r.y between 0 and7200
                group by all'''.replace('and7200','and 7200'),[fold,allowed]).df()
            assert len(q)==read(OUT/'parameters.json')[fold]['cells']
            np.testing.assert_array_equal(q.n,q.sql_n)
            np.testing.assert_allclose(q.truth,q.sql_mean,atol=1e-8,rtol=0)
        count,err=con.execute('''select count(*),max(abs(cal.source_mean-60*s.total_min/s.n))
            from cal join s on cal.airport=s.airport and cal.local_day=s.day and cal.runway=s.runway''').fetchone()
        assert count==con.execute('select count(*) from cal').fetchone()[0] and err<1e-8
    # Separate scalar implementation of every group's composition/gain/update.
    c=pd.read_parquet(OUT/'cells.parquet'); parameters=read(OUT/'parameters.json')
    for row in c.itertuples(index=False):
        k=list(ZONES).index(row.airport);par=parameters[row.fold]
        ok=row.n>=20 and row.source_n>=20 and .5<=row.source_n/row.n<=1.2 and np.isfinite(row.source_mean)
        shift=0.
        if ok:
            composition=par['coefficients'][k]+par['coefficients'][10+k]*(1-row.source_n/row.n)/.1
            variance=40000./row.n+225.
            gain=variance/(variance+par['noise'][row.airport]**2)
            shift=gain*min(120.,max(-120.,row.source_mean+composition-row.mean))
        np.testing.assert_allclose(row.shift,shift,atol=1e-9,rtol=0)
        assert row.supported==ok
    frame=pd.read_parquet(OUT/'scored.parquet'); ids=frame.mvt_id.to_numpy()
    raw=pd.read_parquet(base.OUT/'rows2025.parquet').set_index('mvt_id').loc[ids].reset_index()
    matches=pd.read_parquet(constraints.exclusion.OUT/'matches2025.parquet')
    for name in ['v76','refined']:
        expected=constrain(raw,ids,frame.month,frame[name].to_numpy()+frame[name+'_shift'].to_numpy(),matches)
        np.testing.assert_array_equal(expected,frame[name+'_daily'])
    value=dict(status='pass',rows=n,source_cells=count,inference_cells=len(c),
        independent_sql_truth_local_calendar_training_membership_means_parents_scores=True,
        independent_scalar_cell_updates=True,all_row_final_constraints_replay=True,
        result=art(OUT/'result.json'),identity=art(OUT/'identity.json'),source=art(__file__))
    write_json(OUT/'audit.json',value);print(value,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['guard','screen','audit'])
    args=parser.parse_args()
    with threadpool_limits(1): globals()[args.stage]()
