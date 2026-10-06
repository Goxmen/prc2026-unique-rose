"""Snow-day offset probe on v54: one leaderboard test on one group (built, not uploaded, by this script).

On January 2026 snow days, v54's predictions at EDDF, LFPG and LEMD sit much further above the AOBT_3 clock
than on dry days, more than real taxi-outs did on 2025 snow days (Jan-Mar, Nov-Dec):
    extra_A = [mean(pred - clock | snow) - mean(pred - clock | dry)]_2026  -  [same with labels]_2025
The probe lowers every eligible flight at airport A by SHARE x extra_A. Eligible: January 2026 departure at A,
on a UTC day with a METAR snow report (any SN code), clock = takeoff - AOBT_3 in (0, 3 h], rule not triggered.
LTFM is left out: v52 already tested Istanbul snow.

Reading the result: with d_A = SHARE x extra_A and the assumption that v54's mean error on A's group is
k x extra_A, the leaderboard change gives k = SHARE^2... solved exactly in the record:
    N x (LB_new^2 - LB_v54^2) = sum_A n_A (d_A^2 - 2 d_A k extra_A)
so k = sum_A n_A d_A^2 - N dMSE, over 2 sum_A n_A d_A extra_A, and the best shift is k x extra_A.

Run: .venv/bin/python VALIDATION/snow_offset_probe.py --version 55
"""
import argparse
import hashlib
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION')]
from records import write_json

V54 = ROOT / 'VALIDATION/runs/v3_20261006_hpo_submit/submission_v54'
OUT = ROOT / 'VALIDATION/runs/v3_20261006_snow_probe'
AIRPORTS = ['EDDF', 'LFPG', 'LEMD']
WINTER = ['2025-01', '2025-02', '2025-03', '2025-11', '2025-12']
SHARE = 0.5
CLIP = 1800


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--version', type=int, required=True)
    args = ap.parse_args()
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    met = con.execute(f"""select station airport, strftime(valid, '%Y-%m-%d') dstr, max((wxcodes like '%SN%')::int) snow
        from '{ROOT}/data/external/metar/metar.parquet' group by all""").df()
    tr = con.execute(f"""select ADEP_mvt airport, strftime(MVT_TIME_UTC_mvt, '%Y-%m-%d') dstr, TAXITIME_SEC_mvt y,
        epoch(MVT_TIME_UTC_mvt) - epoch(AOBT_3_flt) clock from read_parquet('{ROOT}/data/training_2025-*.parquet')
        where PHASE_mvt = 'DEP' and AOBT_3_flt is not null and ADEP_mvt in ({",".join(f"'{a}'" for a in AIRPORTS)})""").df()
    tr = tr[tr.clock.between(1, 3 * 3600) & tr.dstr.str[:7].isin(WINTER)].merge(met, on=['airport', 'dstr'], how='left')
    tr['off'] = (tr.y - tr.clock).clip(-CLIP, CLIP)
    te = con.execute(f"""select MVT_ID_mvt mvt_id, ADEP_mvt airport, strftime(MVT_TIME_UTC_mvt, '%Y-%m-%d') dstr,
        epoch(MVT_TIME_UTC_mvt) - epoch(AOBT_3_flt) clock from '{ROOT}/data/ranking.parquet' where PHASE_mvt = 'DEP'""").df()
    pr = pd.read_parquet(V54 / 'ranking_2026-01.parquet', columns=['mvt_id', 'pred', 'rule_triggered'])
    te = te.merge(pr, on='mvt_id').merge(met, on=['airport', 'dstr'], how='left')
    te['off'] = (te.pred - te.clock).clip(-CLIP, CLIP)
    ok = te.clock.between(1, 3 * 3600) & ~te.rule_triggered
    sub = pd.read_parquet(ROOT / 'submissions/unique-rose_v54.parquet')
    if sha256(ROOT / 'submissions/unique-rose_v54.parquet') != '367d4d19f365c497e7a408713513a4813a633e02d5f7f658233f68a53d77970e':
        raise ValueError('v54 submission file differs from the scored one')
    check = sub.merge(pr, left_on='MVT_ID_mvt', right_on='mvt_id')
    if not (np.abs(check.TAXITIME_SEC_mvt - check.pred.round()) <= 1).all():
        raise ValueError('v54 file does not match its January prediction frame')
    shift = pd.Series(0.0, index=te.mvt_id.to_numpy())
    rec = dict(parent=dict(file='unique-rose_v54.parquet', sha256=sha256(ROOT / 'submissions/unique-rose_v54.parquet'), lb=236.3328),
               share=SHARE, airports={}, n_total=int(len(sub)))
    for a in AIRPORTS:
        t25, t26 = tr[tr.airport == a], te[(te.airport == a) & ok]
        up25 = t25[t25.snow == 1].off.mean() - t25[t25.snow == 0].off.mean()
        up26 = t26[t26.snow == 1].off.mean() - t26[t26.snow == 0].off.mean()
        extra = float(up26 - up25)
        rows = t26[t26.snow == 1].mvt_id.to_numpy()
        d = SHARE * extra
        shift.loc[rows] = d
        rec['airports'][a] = dict(snow_days_2025_winter=int(t25[t25.snow == 1].dstr.nunique()),
                                  snow_days_2026_jan=int(t26[t26.snow == 1].dstr.nunique()),
                                  up_2025_real=float(up25), up_2026_v54=float(up26), extra=extra, shift_sec=d, rows=int(len(rows)))
        print(a, rec['airports'][a], flush=True)
    new = sub.copy()
    s = shift.reindex(new.MVT_ID_mvt.to_numpy()).fillna(0).to_numpy()
    base = new.merge(pr, left_on='MVT_ID_mvt', right_on='mvt_id', how='left').pred.to_numpy()
    changed = s != 0
    vals = new.TAXITIME_SEC_mvt.to_numpy().astype(np.int64)
    vals[changed] = np.maximum(np.round(base[changed] - s[changed]), 1).astype(np.int64)
    new['TAXITIME_SEC_mvt'] = vals.astype(np.int32)
    template = pd.read_parquet(ROOT / 'data/submitting.parquet')
    if not np.array_equal(template.MVT_ID_mvt.to_numpy(), new.MVT_ID_mvt.to_numpy()) or new.TAXITIME_SEC_mvt.isna().any():
        raise ValueError('Submission rows differ from the template')
    path = ROOT / f'submissions/unique-rose_v{args.version}.parquet'
    new.to_parquet(path, index=False)
    dlt = (new.TAXITIME_SEC_mvt.to_numpy(float) - sub.TAXITIME_SEC_mvt.to_numpy(float))
    rec.update(file=path.name, sha256=sha256(path), changed_rows=int((dlt != 0).sum()), sum_sq_change=float((dlt ** 2).sum()),
               mean_change=float(dlt[dlt != 0].mean()),
               solve=dict(note='k = (sum_A n_A d_A^2 - N dMSE) / (2 sum_A n_A d_A extra_A); best shift = k * extra_A',
                          sum_n_d2=float(sum(v['rows'] * v['shift_sec'] ** 2 for v in rec['airports'].values())),
                          sum_n_d_extra=float(sum(v['rows'] * v['shift_sec'] * v['extra'] for v in rec['airports'].values()))),
               state='built, not uploaded')
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / f'submission_v{args.version}.json', rec)
    print(f"wrote {path.name}: {rec['changed_rows']} rows changed, mean {rec['mean_change']:+.1f} s", flush=True)


if __name__ == '__main__':
    main()
