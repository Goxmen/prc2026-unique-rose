"""February 2025 held out: how v51's recipe (C4) handles Istanbul in snow.

Question (2026-10-05): January 2026 has 1,895 Istanbul (LTFM) departures in snow; January 2025 has
none, so Jan/Jul CV cannot score them. February 2025 has 2,904 such departures over 12 days.

Split: hold out February 2025, train on the other eleven months (January, March-December), so
the model keeps every other Istanbul snow flight (534, against v51's 3,438). Label history
freezes on 31 December, as the ranking months' does. Built by `tuning_seasonal.prepare_split`.

Recipe: C4 (v51) at one tree seed: 0.6 x the mean of 15 `ple` networks with the 1+2+5 inputs
(`nn_features_jan_jul.py`, seeds 42/142/242 x 5) + 0.4 x the mean of combo LightGBM, combo
per-airport LightGBMs and CatBoost (seed 42), then the Rome mix, corrections and rules.

Stages (run in order, each resumable; heavy jobs one at a time):
  prepare   build the split, Rome classifier, corrections (CPU)
  lgb       combo and combo_airport LightGBM, seed 42 (CPU)
  nets      15 networks (GPU)
  catboost  CatBoost, seed 42 (GPU)
  score     final predictions and the Istanbul snow readout -> result.json
  addon     v52 = v51 + an Istanbul snow add-on learned on February (no refit)

Run: .venv/bin/python VALIDATION/feb_snow_check.py <stage>
No full-year fits. `addon` writes submissions/unique-rose_v52.parquet; uploading is separate.
"""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')     # as nn_improve_jan_jul.py, before torch loads
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]
import tuning_seasonal as ts
from records import RUNS, sha256, write_json

OUT = RUNS / 'v3_20261005_feb_snow_check'
MONTH = '2025-02'
ts.FOLDS[MONTH] = ['2025-01'] + [f'2025-{m:02d}' for m in range(3, 13)]
ts.SPLITS = [MONTH]
TREE_SEED = 42
ARMS = ['combo', 'combo_airport', 'catboost']
W_NN = 0.6


def redirect(*modules):
    for m in modules:
        for name in ('OUT', 'SPLITS_FROM'):
            if hasattr(m, name):
                setattr(m, name, OUT)


def prepare():
    ts.prepare(OUT)


def lgb_arms():
    import ideas_jan_jul as ideas
    redirect(ideas)
    ideas.ARMS = {a: ideas.COMBO for a in ('combo', 'combo_airport')}
    for sub in ('models', 'predictions'):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    ideas.load()
    path = OUT / 'fits.json'
    record = json.loads(path.read_text()) if path.exists() else {}
    for arm in ideas.ARMS:
        if (OUT / 'predictions' / f'{arm}_seed{TREE_SEED}.parquet').exists():
            continue
        _, _, info = ideas.fit_arm(arm, TREE_SEED)
        record.setdefault(arm, {})[str(TREE_SEED)] = info
        record.setdefault('_source', {})[arm] = dict(runner=sha256(Path(ideas.__file__)), columns=ideas.ARMS[arm])
        write_json(path, record)


def nets():
    import nn_features_jan_jul as nf
    import nn_improve_jan_jul as nni
    redirect(nf, nni)
    sys.argv = [sys.argv[0], '--groups', 'arrtaxi', 'peers', 'rwycfg', '--tag', 'combo']
    nf.main()


def catboost():
    import ideas_jan_jul as ideas
    import catboost_jan_jul as cb
    redirect(ideas, cb)
    sys.argv = [sys.argv[0], '--seeds', str(TREE_SEED)]
    cb.main()


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def score():
    d = ts.split_dir(OUT, MONTH)
    held, meta = pd.read_pickle(d / 'heldout.pkl'), ts.read(d / 'meta.json')
    raw, V = held['raw'], held['V']
    ids = raw.mvt_id.to_numpy()

    def rows(path):
        f = pd.read_parquet(path)
        f = f[f.month == MONTH] if 'month' in f else f
        if not np.array_equal(f.mvt_id.to_numpy(), ids):
            raise ValueError(f'{path.name}: rows differ from the held-out month')
        return f.base.to_numpy(float)

    nn = np.mean([rows(OUT / 'networks' / f'nn_combo_{MONTH}_s{s}_k{k}.parquet')
                  for s in (42, 142, 242) for k in range(5)], axis=0)
    gbm_parts = {a: rows(OUT / 'predictions' / f'{a}_seed{TREE_SEED}.parquet') for a in ARMS}
    gbm = np.mean(list(gbm_parts.values()), axis=0)
    final = lambda base: ts.final_prediction(base, held['p'], V, raw, meta['parameters'], meta['corrections'])
    y = raw.taxi_sec.to_numpy(float)
    f = pd.DataFrame(dict(mvt_id=ids, airport=raw.ADEP_mvt.astype(str).to_numpy(), y=y,
                          pred=final(W_NN * nn + (1 - W_NN) * gbm), nn=final(nn), gbm=final(gbm),
                          clock=V.mvt_minus_aobt3_sec.to_numpy(float), snow=V.wx_snow.to_numpy() == 1,
                          snow6h=V.dis_snow_6h_fraction.to_numpy(float) > 0, noid=raw.FLIGHT_ID_mvt.isna().to_numpy(),
                          day=pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True).dt.strftime('%Y-%m-%d').to_numpy()))
    for a, v in gbm_parts.items():
        f[a] = v
    f.to_parquet(OUT / 'predictions' / 'c4_feb.parquet', index=False)
    f['res'] = f.y - f.pred
    f['far'] = ~f.noid & f.clock.notna() & ((f.pred - f.clock).abs() > 600)
    f['above'] = f.pred > f.clock

    def summary(g):
        out = dict(rows=int(len(g)), rmse=rmse(g.res), bias=float(g.res.mean()), clock_rmse=rmse((g.y - g.clock).dropna()),
                   network_rmse=rmse(g.y - g.nn), trees_rmse=rmse(g.y - g.gbm))
        if len(g):
            out.update(truth_near_model=float(((g.y - g.pred).abs() <= 180).mean()),
                       truth_near_clock=float(((g.y - g.clock).abs() <= 180).mean()))
        return out

    ltfm = f[f.airport == 'LTFM']
    readout = dict(
        all=summary(f), by_airport={a: summary(g) for a, g in f.groupby('airport')},
        ltfm=dict(snow=summary(ltfm[ltfm.snow]), no_snow=summary(ltfm[~ltfm.snow]),
                  snow_days={day: summary(g) for day, g in ltfm[ltfm.snow].groupby('day')}),
        ltfm_far_from_clock={f'{"snow" if s else "no snow"}, model {"above" if a else "below"} clock': summary(g)
                             for (s, a), g in ltfm[ltfm.far].groupby(['snow', 'above'])})
    for key, g in (('snow', ltfm[ltfm.far & ltfm.snow]), ('no_snow', ltfm[ltfm.far & ~ltfm.snow])):
        diff, r = g.clock - g.pred, g.y - g.pred
        readout.setdefault('ltfm_clock_weight', {})[key] = float((diff * r).sum() / (diff * diff).sum()) if len(g) else None
    write_json(OUT / 'result.json', dict(month=MONTH, train_months=ts.FOLDS[MONTH], tree_seed=TREE_SEED, networks=15,
                                         weights=dict(network=W_NN, trees=dict.fromkeys(ARMS, (1 - W_NN) / 3)),
                                         runner=sha256(Path(__file__)), readout=readout))
    print(json.dumps(readout['ltfm'], indent=1)[:4000])


def addon():
    """v52 = v51 + an Istanbul snow add-on sized by the peer clock, learned on February's misses.

    Learned on the held-out February predictions: Istanbul departures with snow in the previous
    24 h (`dis_snow_24h_fraction` > 0); miss = b x (peer clock - its normal level, floored at 0),
    where the peer clock is `pc_apt_mean_pm30` (other departures' takeoff - AOBT_3 within 30 min)
    and its normal level is the median over February's Istanbul departures without recent snow.
    Checked on February's two snow spells (learn one, check the other): 1008 -> 896 on 18-25 Feb,
    423 -> 440 on 5-13 Feb. Applied unchanged to v51's predictions; nothing is refitted.
    """
    import extra_features as ef
    d = ts.split_dir(OUT, MONTH)
    V = pd.read_pickle(d / 'heldout.pkl')['V']
    f = pd.read_parquet(OUT / 'predictions' / 'c4_feb.parquet')
    f['recent'] = V.dis_snow_24h_fraction.to_numpy(float) > 0
    peers = ef.load()['pc_apt_mean_pm30']
    f['peer'] = peers.reindex(f.mvt_id).to_numpy()
    ltfm = f[f.airport == 'LTFM']
    normal = float(ltfm.peer[~ltfm.recent].median())
    learn = ltfm[ltfm.recent]
    excess = ((learn.peer.fillna(normal) - normal).clip(lower=0) / 60).to_numpy()
    b = float(excess @ (learn.y - learn.pred).to_numpy() / (excess @ excess))
    data = pd.read_pickle(RUNS / 'v3_20261003_ideas_submit' / 'work' / 'inputs.pkl')
    rows = []
    for month, m in data['months'].items():
        raw, Vr = m['raw'], m['V']
        rows.append(pd.DataFrame(dict(mvt_id=raw.mvt_id.to_numpy(), airport=raw.ADEP_mvt.astype(str).to_numpy(),
                                      recent=Vr.dis_snow_24h_fraction.to_numpy(float) > 0,
                                      day=pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True).dt.strftime('%Y-%m-%d').to_numpy())))
    r = pd.concat(rows, ignore_index=True)
    r['peer'] = peers.reindex(r.mvt_id).to_numpy()
    on = (r.airport == 'LTFM') & r.recent
    r['add'] = np.where(on, b * (r.peer.fillna(normal) - normal).clip(lower=0) / 60, 0.0)
    template = pd.read_parquet(ROOT / 'data' / 'submitting.parquet')
    v51 = pd.read_parquet(ROOT / 'submissions' / 'unique-rose_v51.parquet')
    if not v51.MVT_ID_mvt.equals(template.MVT_ID_mvt):
        raise ValueError('v51 rows differ from the template')
    add = r.set_index('mvt_id')['add'].reindex(v51.MVT_ID_mvt.to_numpy()).to_numpy()
    if np.isnan(add).any():
        raise ValueError('ranking rows without an add-on value')
    out = template.copy()
    out['TAXITIME_SEC_mvt'] = (v51.TAXITIME_SEC_mvt.to_numpy(float) + add).astype(v51.TAXITIME_SEC_mvt.dtype)
    path = ROOT / 'submissions' / 'unique-rose_v52.parquet'
    if path.exists():
        raise ValueError(f'{path.name} already exists')
    out.to_parquet(path, index=False)
    changed = r[on & (r['add'] > 0)]
    write_json(OUT / 'submission_v52.json', dict(
        parent=dict(file='unique-rose_v51.parquet', sha256=sha256(ROOT / 'submissions' / 'unique-rose_v51.parquet')),
        file=path.name, sha256=sha256(path), coefficient_sec_per_min=b, normal_peer_clock_sec=normal,
        learned_rows=int(len(learn)), eligible_rows=int(on.sum()), changed_rows=int(len(changed)),
        add_sec=dict(sum_sq=float((add ** 2).sum()), mean_changed=float(changed['add'].mean()), max=float(add.max())),
        add_by_day={k: dict(rows=int(len(g)), mean_sec=float(g['add'].mean())) for k, g in r[on].groupby('day')},
        runner=sha256(Path(__file__))))
    print(f'b = {b:.1f} s per minute above {normal / 60:.1f} min; eligible {on.sum()}, changed {len(changed)}, '
          f'mean add {changed["add"].mean():.0f} s, max {add.max():.0f} s')
    print(r[on].groupby('day')['add'].agg(['size', 'mean']).round(0).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('stage', choices=['prepare', 'lgb', 'nets', 'catboost', 'score', 'addon'])
    stage = ap.parse_args().stage
    OUT.mkdir(parents=True, exist_ok=True)
    dict(prepare=prepare, lgb=lgb_arms, nets=nets, catboost=catboost, score=score, addon=addon)[stage]()


if __name__ == '__main__':
    main()
