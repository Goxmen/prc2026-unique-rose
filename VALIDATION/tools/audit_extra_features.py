"""Brute-force check of `extra_features.py` on a random sample of departures.

Each sampled departure's inputs are recomputed by plain filtering of the movement frames (no
sorting tricks, merge_asof or cumulative sums) and compared with the cached table.
Run: .venv/bin/python VALIDATION/tools/audit_extra_features.py [--n 300]
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]
import extra_features as ef
import overnight_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=300)
    args = ap.parse_args()
    table = ef.load()
    dep, arr, rank, rank_arr = overnight_run.load_inputs(1., ranking=True)
    cd = ['mvt_id', 'ADEP_mvt', 'STAND_mvt', 'RUNWAY_mvt', 'MVT_TIME_UTC_mvt', 'AOBT_3_flt']
    deps = pd.concat([dep[cd], rank[cd]], ignore_index=True)
    arrs = pd.concat([arr[['ADES_mvt', 'STAND_mvt', 'RUNWAY_mvt', 'MVT_TIME_UTC_mvt', 'taxi_sec']],
                      rank_arr[['ADES_mvt', 'STAND_mvt', 'RUNWAY_mvt', 'MVT_TIME_UTC_mvt', 'taxi_sec']]],
                     ignore_index=True)
    deps['t'] = pd.to_datetime(deps.MVT_TIME_UTC_mvt, utc=True)
    proxy = (deps.t - pd.to_datetime(deps.AOBT_3_flt, utc=True)).dt.total_seconds()
    deps['proxy'] = proxy.where((proxy > 0) & (proxy <= ef.PROXY_MAX))
    arrs['land'] = pd.to_datetime(arrs.MVT_TIME_UTC_mvt, utc=True)
    arrs = arrs[arrs.taxi_sec.notna() & (arrs.taxi_sec >= 0)]
    arrs['inb'] = arrs.land + pd.to_timedelta(arrs.taxi_sec, unit='s')
    sample = deps.sample(args.n, random_state=20261003)
    checks, bad = 0, []

    def cmp(name, row, expected):
        nonlocal checks
        got = table.at[row.mvt_id, name]
        checks += 1
        same = (pd.isna(got) and pd.isna(expected)) or (not pd.isna(got) and not pd.isna(expected)
                                                         and abs(float(got) - float(expected)) < 1e-3)
        if not same:
            bad.append((name, row.mvt_id, got, expected))

    by_apt_d = dict(tuple(deps.groupby('ADEP_mvt')))
    by_apt_a = dict(tuple(arrs.groupby('ADES_mvt')))
    for _, r in sample.iterrows():
        a = by_apt_a[r.ADEP_mvt]
        d = by_apt_d[r.ADEP_mvt]
        st = a[(a.STAND_mvt == r.STAND_mvt) & (a.inb < r.t)].sort_values('inb') if pd.notna(r.STAND_mvt) else a.iloc[:0]
        cmp('sa_last_taxiin_sec', r, st.taxi_sec.iloc[-1] if len(st) else np.nan)
        cmp('sa_prev_taxiin_sec', r, st.taxi_sec.iloc[-2] if len(st) > 1 else np.nan)
        rw = a[(a.RUNWAY_mvt == r.RUNWAY_mvt) & (a.land < r.t)].sort_values('land') if pd.notna(r.RUNWAY_mvt) else a.iloc[:0]
        cmp('ra_last_taxiin_sec', r, rw.taxi_sec.iloc[-1] if len(rw) else np.nan)
        others = d[d.mvt_id != r.mvt_id]
        dt = (others.t - r.t).dt.total_seconds()
        w = others[(dt.abs() <= 1800) & others.proxy.notna()]
        cmp('pc_apt_n_pm30', r, len(w))
        cmp('pc_apt_mean_pm30', r, w.proxy.mean() if len(w) else np.nan)
        nx = others[(dt > 0) & (dt <= 1800) & others.proxy.notna()]
        cmp('pc_apt_mean_next30', r, nx.proxy.mean() if len(nx) else np.nan)
        if pd.notna(r.RUNWAY_mvt):
            same = others[others.RUNWAY_mvt == r.RUNWAY_mvt]
            ds = (same.t - r.t).dt.total_seconds()
            nxt = same[(ds > 0) & same.proxy.notna()].sort_values('t')
            cmp('pc_rwy_next_proxy', r, nxt.proxy.iloc[0] if len(nxt) and (nxt.t.iloc[0] - r.t).total_seconds() <= 1800
                else np.nan)
            seq = d[d.RUNWAY_mvt == r.RUNWAY_mvt].sort_values(['t'], kind='mergesort').t.to_numpy()
            i = int(np.flatnonzero(d[d.RUNWAY_mvt == r.RUNWAY_mvt].sort_values(['t'], kind='mergesort').mvt_id.to_numpy()
                                   == r.mvt_id)[0])
            k = 0
            while i - k - 1 >= 0 and (seq[i - k] - seq[i - k - 1]) / np.timedelta64(1, 's') < ef.BURST_GAP:
                k += 1
            cmp('rc_burst_prev', r, k)
        prev60 = d[(d.t >= r.t - pd.Timedelta(minutes=60)) & (d.t < r.t)]
        used = prev60.RUNWAY_mvt.value_counts()
        cmp('rc_dep_rwys_60', r, int((used >= 2).sum()))
    print(f'{checks} comparisons on {args.n} departures; {len(bad)} mismatches')
    for b in bad[:20]:
        print('  ', b)


if __name__ == '__main__':
    main()
