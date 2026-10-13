"""One fixed training-only falsifier of cross-flight Rome mode independence.

Retained honest R20 probabilities are reused. There is no model fitting,
public posterior, held-outer taxi scoring, candidate generation or upload.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import resource
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import rome_signals
from records import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'VALIDATION/runs/v3_20261013_correlated_modes_pilot'
MANIFEST = OUT/'identity.json'
PLAN = ROOT/'VALIDATION/reports/cohort_learning_20261013/correlated_modes_pilot_design.json'
REVIEW = OUT/'source_review.json'
AUDITOR = ROOT/'VALIDATION/tools/audit_correlated_modes_pilot_20261013.py'
CACHE = ROOT/'VALIDATION/runs/v3_20261013_posterior_aware_crossfit'
OUTERS = ['2025-01','2025-07']
PERMUTATIONS, BOOTSTRAPS = 199, 2000


def art(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)),sha256=sha256(path))


def read(path):
    return json.loads(Path(path).read_text())


def verify(record):
    path = ROOT/record['path']
    assert art(path) == record, record['path']
    return path


def ids_hash(ids):
    return hashlib.sha256(np.asarray(ids,dtype=np.int64).tobytes()).hexdigest()


def freeze():
    assert not MANIFEST.exists() and not (OUT/'started.json').exists()
    plan, review = read(PLAN), read(REVIEW)
    assert review['status'] == 'pass' and review['source'] == art(__file__)
    assert review['auditor'] == art(AUDITOR) and review['design'] == art(PLAN)
    metadata_paths = {verify(r) for r in plan['metadata_sources']}
    prepared = read(CACHE/'prepared.json')
    assert prepared['cross_month_aircraft_aliases'] == 0
    cached = {Path(r['path']).name:r for r in prepared['outputs']}
    paths = {Path(__file__),PLAN,REVIEW,AUDITOR,CACHE/'prepared.json',
             ROOT/'VALIDATION/rome_hybrid.py',Path(rome_signals.__file__),ROOT/'VALIDATION/records.py'}
    paths.update(metadata_paths)
    raw = {}
    for record in prepared['raw_sources']:
        path = Path(record['path'])
        match = re.fullmatch(r'training_(2025-\d{2})-01_(\d{4}-\d{2})-01\.parquet',path.name)
        if path.parent == Path('data') and match:
            month = match.group(1)
            assert str(pd.Period(month,freq='M')+1) == match.group(2)
            assert month not in raw
            raw[month] = record
    needed_months = {month for outer in OUTERS for rec in plan['folds'][outer] for month in rec['predicted_months']}
    assert set(raw) == needed_months
    fold_inputs = {}
    peer_identity = None
    for outer in OUTERS:
        records = plan['folds'][outer]
        assert len(records) == 3
        months = sorted(m for rec in records for m in rec['predicted_months'])
        assert len(months) == len(set(months)) and outer not in months
        for rec in records:
            receipt = read(verify(rec['receipt']))
            if peer_identity is None:
                peer_identity = receipt['identity']
            assert receipt['identity'] == peer_identity
            paths.add(verify(peer_identity))
            for key in ['prediction','model','fitting_months','predicted_months','held_rows','held_ids_sha256']:
                assert rec[key] == receipt[key]
            assert receipt['all_held_reload_exact'] and len(receipt['columns']) == 63
            assert not set(rec['fitting_months']) & set(rec['predicted_months'])
            assert set(rec['fitting_months']) | set(rec['predicted_months']) == set(months)
            assert outer not in rec['fitting_months']
            paths.update(verify(rec[k]) for k in ['receipt','prediction','model'])
        fold_inputs[outer] = dict(months=months,records=records,
            labels={month:cached[f'labels_{month}.parquet'] for month in months})
        paths.update(verify(r) for r in fold_inputs[outer]['labels'].values())
        paths.update(verify(raw[month]) for month in months)
    needed = {k:cached[k] for k in ['rows.parquet','features.parquet']}
    paths.update(verify(r) for r in needed.values())
    OUT.mkdir(parents=True,exist_ok=True)
    write_json(MANIFEST,dict(source=art(__file__),design=art(PLAN),auditor=art(AUDITOR),
        review=art(REVIEW),folds=fold_inputs,inputs=needed,raw=raw,
        peer_identity=peer_identity,
        sources=[art(p) for p in sorted(paths)],no_numeric_read_before_freeze=True,
        no_new_model_fit=True,no_held_outer_labels=True))
    print(dict(identity=art(MANIFEST),sources=len(paths)),flush=True)


def guard():
    m = read(MANIFEST)
    assert m['source'] == art(__file__) and m['design'] == art(PLAN)
    assert m['auditor'] == art(AUDITOR) and m['review'] == art(REVIEW)
    for rec in m['sources']:
        verify(rec)
    return m


def airline_inputs(m):
    frames = []
    for month, record in sorted(m['raw'].items()):
        f = pd.read_parquet(verify(record),columns=['MVT_ID_mvt','FLIGHT_mvt'],
            filters=[('PHASE_mvt','==','DEP'),('ADEP_mvt','==','LIRF')])
        frames.append(pd.DataFrame(dict(mvt_id=f.MVT_ID_mvt,
            airline=rome_signals.airline_code(f.FLIGHT_mvt).to_numpy())))
    joined = pd.concat(frames,ignore_index=True)
    assert joined.mvt_id.is_unique
    return joined.set_index('mvt_id')


def prepare_fold(outer,m,metadata,features,airlines):
    spec = m['folds'][outer]
    vectors = []
    for rec in spec['records']:
        f = pd.read_parquet(verify(rec['prediction']))
        assert list(f) == ['mvt_id','p'] and len(f) == rec['held_rows']
        assert f.mvt_id.is_unique and ids_hash(f.mvt_id) == rec['held_ids_sha256']
        assert set(metadata.loc[f.mvt_id,'month']) == set(rec['predicted_months'])
        assert np.isfinite(f.p).all() and f.p.between(0,1).all()
        vectors.append(f)
    f = pd.concat(vectors,ignore_index=True)
    assert f.mvt_id.is_unique
    f = f.set_index('mvt_id').join(metadata[['month','day','gap','airport']],validate='one_to_one')
    f = f.join(features[['mvt_minus_aobt3_sec']],validate='one_to_one').join(airlines,validate='one_to_one')
    assert set(f.month) == set(spec['months']) and outer not in set(f.month)
    assert f.airport.eq('LIRF').all() and pd.to_datetime(f.day).dt.strftime('%Y-%m').eq(f.month).all()
    counts = dict(oof_rows=len(f),oof_ids_sha256=ids_hash(f.index))
    keep = np.isfinite(f.gap) & f.gap.gt(3600) & f.gap.lt(50400) & f.airline.notna() & f.airline.ne('')
    f = f.loc[keep].copy()
    counts['input_eligible_rows'] = len(f)
    f['p_bin'] = np.minimum(np.floor(10*f.p).astype(int),9)
    f['gap_band'] = np.where(f.gap.le(14400),0,1)
    clock = f.gap-f.mvt_minus_aobt3_sec
    f['clock_band'] = np.where(~np.isfinite(clock),0,np.where(abs(clock).le(60),1,2))
    keys = ['airline','month','p_bin','gap_band','clock_band']
    dates = f.groupby(keys,dropna=False).day.transform('nunique')
    f = f.loc[dates.ge(4)].reset_index().sort_values('mvt_id').reset_index(drop=True)
    f['stratum'] = pd.factorize(pd.MultiIndex.from_frame(f[keys]),sort=True)[0]
    f['group'] = pd.factorize(pd.MultiIndex.from_frame(f[['airline','day']]),sort=True)[0]
    assert (f[['stratum','group']] >= 0).all().all()
    # Only now, after exact input/prediction membership, open this outer's allowed labels.
    labels = pd.concat([pd.read_parquet(verify(spec['labels'][month])) for month in spec['months']],ignore_index=True)
    assert labels.mvt_id.is_unique
    label = labels.set_index('mvt_id').loc[f.mvt_id,'y'].to_numpy(float)
    assert np.isfinite(label).all()
    f['z'] = (abs(f.gap.to_numpy(float)-label) <= 60).astype(np.int8)
    f['u'] = f.gap*(f.z-f.p)
    f['v'] = f.gap**2*f.p*(1-f.p)
    assert np.isfinite(f[['u','v']]).all().all() and f.v.ge(0).all()
    f = f.drop(columns=['airport','mvt_minus_aobt3_sec'])
    group_sizes = f.groupby('group').size()
    counts.update(retained_rows=len(f),retained_strata=f.stratum.nunique(),
        groups=len(group_sizes),groups_with_pairs=int(group_sizes.ge(2).sum()),days=f.day.nunique(),
        dropped_sparse_stratum_rows=counts['input_eligible_rows']-len(f))
    return f,counts


def group_cross(u,group,n_groups):
    return np.bincount(group,weights=u,minlength=n_groups)**2-np.bincount(group,weights=u*u,minlength=n_groups)


def analytic_null(f,n_groups):
    stats = f.groupby('stratum',sort=True).u.agg(['size','sum'])
    square = f.assign(u2=f.u*f.u).groupby('stratum',sort=True).u2.sum()
    stats['mu'] = stats['sum']/stats['size']
    stats['beta'] = np.divide(stats['sum']**2-square,stats['size']*(stats['size']-1),
        out=np.zeros(len(stats)),where=stats['size'].to_numpy()>1)
    slots = f.groupby(['group','stratum'],sort=True).size().rename('n').reset_index()
    slots = slots.join(stats[['mu','beta']],on='stratum',validate='many_to_one')
    mean = slots.n*slots.mu
    within = slots.n*(slots.n-1)*slots.beta
    return np.bincount(slots.group,weights=mean,minlength=n_groups)**2 + np.bincount(
        slots.group,weights=within-mean**2,minlength=n_groups)


def bootstrap_days(days,reps=BOOTSTRAPS):
    rng = np.random.Generator(np.random.PCG64(20261014))
    numerator = np.zeros(reps)
    denominator = np.zeros(reps)
    for month,g in days.groupby('month',sort=True):
        draw = rng.integers(0,len(g),size=(reps,len(g)))
        numerator += g.numerator.to_numpy()[draw].sum(1)
        denominator += g.denominator.to_numpy()[draw].sum(1)
    assert np.isfinite(denominator).all() and (denominator > 0).all()
    return numerator/denominator


def evaluate(f,counts):
    denominator = float(f.v.sum())
    supported = counts['retained_rows'] >= 1000 and counts['groups_with_pairs'] >= 100 and counts['days'] >= 50 and denominator > 0
    if not supported:
        return dict(status='inconclusive_insufficient_support',coverage=counts,denominator=denominator,gate=False),None,None,None
    group = f.group.to_numpy();ng = int(group.max())+1
    group_keys = f[['group','day','month']].drop_duplicates().sort_values('group')
    assert len(group_keys) == ng and np.array_equal(group_keys.group,np.arange(ng))
    day_index = pd.Index(sorted(f.day.unique()))
    group_day = day_index.get_indexer(group_keys.day)
    row_day = day_index.get_indexer(f.day)
    observed = np.bincount(group_day,weights=group_cross(f.u.to_numpy(),group,ng),minlength=len(day_index))
    expected = np.bincount(group_day,weights=analytic_null(f,ng),minlength=len(day_index))
    null = np.empty((PERMUTATIONS,len(day_index)),dtype=float)
    rng = np.random.Generator(np.random.PCG64(20261013))
    strata = [ix for _,ix in f.groupby('stratum',sort=True).indices.items()]
    u = f.u.to_numpy();v = f.v.to_numpy()
    for k in range(PERMUTATIONS):
        permutation = np.arange(len(f))
        for ix in strata:
            permutation[ix] = rng.permutation(ix)
        assert np.array_equal(np.sort(permutation),np.arange(len(f)))
        assert np.isclose(v[permutation].sum(),denominator,rtol=1e-14,atol=1e-6)
        null[k] = np.bincount(group_day,weights=group_cross(u[permutation],group,ng),minlength=len(day_index))
    days = pd.DataFrame(dict(day=day_index,month=day_index.str[:7],observed_cross=observed,
        null_mean=expected,mc_null_mean=null.mean(0),denominator=np.bincount(row_day,weights=v,minlength=len(day_index))))
    days['numerator'] = days.observed_cross-days.null_mean
    estimate = float(days.numerator.sum()/denominator)
    draws = bootstrap_days(days)
    lower,upper = map(float,np.percentile(draws,[2.5,97.5]))
    permutation = pd.DataFrame(dict(iteration=np.arange(PERMUTATIONS),null_cross=null.sum(1),
        excess_variance_ratio=(observed.sum()-null.sum(1))/denominator))
    permutation_p = float((1+np.count_nonzero(null.sum(1)>=observed.sum()))/(PERMUTATIONS+1))
    passed = estimate >= .10 and lower > 0 and permutation_p <= .05
    report = dict(status='material_dependence' if passed else 'closed_fixed_gate_failed',coverage=counts,
        denominator=denominator,observed_cross=float(observed.sum()),null_mean_cross=float(expected.sum()),
        mc_null_mean_cross=float(null.sum(1).mean()),permutation_p=permutation_p,
        excess_variance_ratio=estimate,ci95=[lower,upper],gate=passed,
        bootstrap_conditions='Observed coarse strata and exact empirical permutation means; classifier and nuisance-distribution uncertainty excluded.',
        interpretation='Diagnostic original-family residual dependence, not production copula calibration, causal recorder proof or taxi-CV gain.')
    return report,days,permutation,draws


def run():
    start = time.monotonic();m = guard()
    assert not (OUT/'started.json').exists() and not (OUT/'result.json').exists(), 'Preserve single attempted pilot'
    allocation = read(OUT/'allocation.json')
    assert allocation['authority'] == '/root' and allocation['identity'] == art(MANIFEST)
    assert allocation['state'] == 'released_for_fixed_correlated_mode_training_pilot'
    write_json(OUT/'started.json',dict(identity=art(MANIFEST),allocation=art(OUT/'allocation.json')))
    metadata = pd.read_parquet(verify(m['inputs']['rows.parquet'])).set_index('mvt_id')
    features = pd.read_parquet(verify(m['inputs']['features.parquet']),columns=['mvt_id','mvt_minus_aobt3_sec']).set_index('mvt_id')
    assert metadata.index.is_unique and features.index.is_unique
    airlines = airline_inputs(m)
    results,outputs = {},[]
    for outer in OUTERS:
        folder = OUT/outer;folder.mkdir(exist_ok=True)
        frame,counts = prepare_fold(outer,m,metadata,features,airlines)
        report,days,permutation,draws = evaluate(frame,counts)
        frame.to_parquet(folder/'retained_rows.parquet',index=False)
        paths = [folder/'retained_rows.parquet']
        if days is not None:
            days.to_parquet(folder/'days.parquet',index=False)
            permutation.to_parquet(folder/'permutations.parquet',index=False)
            np.save(folder/'bootstrap.npy',draws,allow_pickle=False)
            paths += [folder/'days.parquet',folder/'permutations.parquet',folder/'bootstrap.npy']
        report.update(identity=art(MANIFEST),outer=outer,inputs=m['folds'][outer],outputs=[art(p) for p in paths])
        write_json(folder/'result.json',report)
        results[outer] = report
        outputs += [art(folder/'result.json'),*report['outputs']]
    value = dict(identity=art(MANIFEST),allocation=art(OUT/'allocation.json'),results=results,
        gate=all(r['gate'] for r in results.values()),outputs=outputs,
        seconds=time.monotonic()-start,peak_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        no_fit=True,no_held_outer_labels=True,no_candidate=True,no_upload=True,independent_audit_pending=True)
    write_json(OUT/'result.json',value)
    print(dict(gate=value['gate'],results={k:{a:v[a] for a in ['status','coverage','gate']} for k,v in results.items()},seconds=value['seconds']),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['freeze','run'])
    with threadpool_limits(1):
        globals()[parser.parse_args().stage]()
