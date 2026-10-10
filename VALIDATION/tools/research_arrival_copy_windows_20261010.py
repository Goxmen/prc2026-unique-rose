"""Frozen four arrival-recording inputs; coverage/association only, no fits.

Uses supplied arrival taxi-in to reconstruct arrival in-block. Feature building
never selects departure targets or hidden departure block timestamps.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
import json
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'VALIDATION'), str(ROOT/'pipeline')]
import rome_signals as signals
import tuning_seasonal as ts
from records import sha256

OUT = ROOT/'tmp/research_20261010/error_mechanisms/arrival_copy_windows'
FEATURES = ['handler_arr_copy_pm24h', 'handler_arr_n_pm24h',
            'handler_arr_copy_prev7d', 'handler_arr_n_prev7d']
WINDOWS = {'pm24h': (-86400, 86400, 'right'),
           'prev7d': (-7*86400, 0, 'left')}
DESIGN = dict(features=FEATURES, windows=WINDOWS, match_seconds=60,
              arrival_time_anchor='landing UTC', query_anchor='departure takeoff UTC',
              grouping='airport x airline (Rome-only screen)', prior_weight=10,
              prior='same airport and same local window',
              missing_airline='all four inputs NaN',
              zero_airline_support='count zero, share airport-window prior',
              context='train: allowed training arrivals; held: allowed training plus own month',
              final_context='all 2025 arrivals plus own 2026 month only',
              association_bins=[-float('inf'), -.05, .05, float('inf')],
              no_fit=True)


def seconds(values):
    return pd.to_datetime(values, utc=True).dt.as_unit('ns').astype('int64').to_numpy()/1e9


def counts(times, values, query_times, lower, upper, right_side):
    order = np.argsort(times, kind='stable')
    t = np.asarray(times)[order]
    prefix = np.r_[0., np.cumsum(np.asarray(values, float)[order])]
    lo = np.searchsorted(t, query_times+lower, side='left')
    hi = np.searchsorted(t, query_times+upper, side=right_side)
    return prefix[hi]-prefix[lo], (hi-lo).astype(float)


def build(raw, arrivals):
    """Rome-only query and arrival frames; no departure labels accepted/read."""
    assert raw.ADEP_mvt.eq('LIRF').all() and arrivals.ADES_mvt.eq('LIRF').all()
    a = arrivals.copy()
    a['airline'] = signals.airline_code(a.FLIGHT_mvt).to_numpy()
    a['t'] = seconds(a.MVT_TIME_UTC_mvt)
    delta = (pd.to_datetime(a.MVT_TIME_UTC_mvt, utc=True)
             + pd.to_timedelta(a.taxi_sec, unit='s')
             - pd.to_datetime(a.SCHED_TIME_UTC_mvt, utc=True)).dt.total_seconds()
    a['copy'] = delta.abs().le(60).astype(float)
    a = a.loc[delta.notna() & a.MVT_TIME_UTC_mvt.notna()]
    q = seconds(raw.MVT_TIME_UTC_mvt)
    airlines = signals.airline_code(raw.FLIGHT_mvt).to_numpy()
    result = {'mvt_id': raw.mvt_id.to_numpy()}
    for name, (lo, hi, side) in WINDOWS.items():
        apt_sum, apt_n = counts(a.t, a['copy'], q, lo, hi, side)
        prior = np.divide(apt_sum, apt_n, out=np.full(len(q), np.nan), where=apt_n>0)
        ss, nn = np.zeros(len(q)), np.zeros(len(q))
        for airline in pd.unique(airlines[pd.notna(airlines)]):
            use = airlines == airline
            events = a.loc[a.airline.eq(airline)]
            ss[use], nn[use] = counts(events.t, events['copy'], q[use], lo, hi, side)
        share = (ss+10*prior)/(nn+10)
        missing = pd.isna(airlines)
        share[missing], nn[missing] = np.nan, np.nan
        result['handler_arr_copy_'+name] = share.astype(np.float32)
        result['handler_arr_n_'+name] = nn.astype(np.float32)
    return pd.DataFrame(result).loc[:, ['mvt_id']+FEATURES]


def load(con, source, phase):
    cols = ['MVT_ID_mvt AS mvt_id', 'MVT_TIME_UTC_mvt', 'FLIGHT_mvt']
    if phase == 'ARR':
        cols += ['ADES_mvt', 'SCHED_TIME_UTC_mvt', 'TAXITIME_SEC_mvt AS taxi_sec']
        airport = 'ADES_mvt'
    else:
        cols += ['ADEP_mvt', 'FLIGHT_ID_mvt']
        airport = 'ADEP_mvt'
    x = con.execute(f"SELECT {','.join(cols)} FROM read_parquet(?) "
                    f"WHERE PHASE_mvt='{phase}' AND {airport}='LIRF'", [str(source)]).df()
    assert x.mvt_id.is_unique
    x['month'] = pd.to_datetime(x.MVT_TIME_UTC_mvt, utc=True).dt.strftime('%Y-%m')
    return x


def coverage(raw, extra):
    np.testing.assert_array_equal(raw.mvt_id, extra.mvt_id)
    result = {}
    for group, mask in [('all', np.ones(len(raw), bool)),
                        ('no_id', raw.FLIGHT_ID_mvt.isna().to_numpy())]:
        f = extra.loc[mask]
        result[group] = dict(n=len(f), features={})
        for name in WINDOWS:
            rate, n = f['handler_arr_copy_'+name], f['handler_arr_n_'+name]
            result[group]['features'][name] = dict(available=int(rate.notna().sum()),
                no_airline_support=int(n.eq(0).sum()), at_least_10=int(n.ge(10).sum()),
                support_median=float(n.median()), rate_mean=float(rate.mean()))
    return result


def describe(frame):
    result = {}
    groups = {'all': np.ones(len(frame), bool), 'no_id': frame.FLIGHT_ID_mvt.isna().to_numpy(),
              'no_id_1_4h': (frame.FLIGHT_ID_mvt.isna() & frame.gap.gt(3600) & frame.gap.lt(14400)).to_numpy()}
    for group, mask in groups.items():
        f = frame.loc[mask].copy()
        target = f.y.sub(f.gap).abs().le(60)
        entry = dict(n=len(f), copy_rate=float(target.mean()), features={})
        for name in ['monthly', *WINDOWS]:
            feature = 'handler_arr_sched_share' if name == 'monthly' else 'handler_arr_copy_'+name
            valid = f[feature].notna()
            truth=target[valid].to_numpy(bool)
            ranks=f.loc[valid,feature].rank(method='average').to_numpy()
            npos,nneg=int(truth.sum()),int((~truth).sum())
            auc = float((ranks[truth].sum()-npos*(npos+1)/2)/(npos*nneg)) if npos and nneg else None
            bins = []
            if name != 'monthly':
                delta = f[feature]-f.handler_arr_sched_share
                for label, use in [('below', delta.lt(-.05)), ('near', delta.between(-.05,.05)), ('above', delta.gt(.05))]:
                    z=f.loc[use]
                    bins.append(dict(bin=label, n=len(z), copies=int(target[use].sum()),
                        copy_share=float(target[use].mean()) if len(z) else None,
                        r20_probability_mean=float(z.p.mean()) if len(z) else None,
                        arrival_share_mean=float(z[feature].mean()) if len(z) else None,
                        monthly_share_mean=float(z.handler_arr_sched_share.mean()) if len(z) else None))
            entry['features'][name] = dict(n_valid=int(valid.sum()), auc=auc, relative_monthly_bins=bins)
        result[group]=entry
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Freeze before loading held departure labels; JSON strings avoid nonstandard infinities.
    design = dict(DESIGN, association_bins=['-infinity', -.05, .05, 'infinity'])
    frozen=OUT/'design.json'
    if frozen.exists() and json.loads(frozen.read_text()) != json.loads(json.dumps(design)):
        raise ValueError('Frozen design changed')
    frozen.write_text(json.dumps(design, indent=2))
    con=duckdb.connect(config={'threads':2});con.execute("SET TimeZone='UTC'")
    dep=load(con,ROOT/'data/training_2025-*.parquet','DEP')
    arr=load(con,ROOT/'data/training_2025-*.parquet','ARR')
    final_dep=load(con,ROOT/'data/final_ranking.parquet','DEP')
    final_arr=load(con,ROOT/'data/final_ranking.parquet','ARR')
    con.close()
    report=dict(design=design, source_sha256=sha256(Path(__file__)), coverage={}, association={})
    for month in ts.SPLITS:
        permitted=ts.FOLDS[month]
        for split, query_months, context_months in [('train', permitted, permitted),
                                                  ('held', [month], permitted+[month])]:
            q=dep.loc[dep.month.isin(query_months)]
            a=arr.loc[arr.month.isin(context_months)]
            extra=build(q,a)
            extra.to_parquet(OUT/f'{month}_{split}.parquet',index=False)
            report['coverage'][month+'_'+split]=coverage(q,extra)
    final=[]
    for month in sorted(final_dep.month.unique()):
        q=final_dep.loc[final_dep.month.eq(month)]
        a=pd.concat([arr,final_arr.loc[final_arr.month.eq(month)]],ignore_index=True)
        extra=build(q,a)
        final.append(extra)
        report['coverage'][month+'_final']=coverage(q,extra)
    pd.concat(final,ignore_index=True).to_parquet(OUT/'final2026.parquet',index=False)
    # Held labels enter only the descriptive association stage, after all inputs are frozen.
    observed=pd.read_parquet(OUT.parent/'diagnostic_rows.parquet', columns=[
        'mvt_id','month','ADEP_mvt','FLIGHT_ID_mvt','gap','y','p','handler_arr_sched_share'])
    observed=observed.loc[observed.ADEP_mvt.eq('LIRF')]
    for month in ts.SPLITS:
        f=observed.loc[observed.month.eq(month)].merge(pd.read_parquet(OUT/f'{month}_held.parquet'),
                                                   on='mvt_id',validate='one_to_one')
        report['association'][month]=describe(f)
    (OUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':
    with threadpool_limits(2):
        main()
