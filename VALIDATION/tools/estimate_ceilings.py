"""Ceiling estimates for the open ideas, on row 43's saved OOF predictions."""
import json, numpy as np, pandas as pd
from pathlib import Path

RUN = Path('VALIDATION/runs/v3_20260920T170600876190Z')
rec = json.load(open(RUN / 'result.json'))
W = rec['ranking_weights']['jan_fraction']
SEEDS = [42, 142, 242]
COL = 'pred37_opdi'          # row 43 / v40 recipe
ALT = 'pred34_opdi'          # a different post-processing stack, same models
A = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']
BL = ['B3', 'B4', 'B7']

data = {}
for fold, info in rec['folds'].items():
    if fold not in A + BL:
        continue
    per = {}
    meta = None
    for s in SEEDS:
        f = pd.read_parquet(RUN / info['predictions'][str(s)]).set_index('mvt_id').sort_index()
        if meta is None:
            meta = f[['mvt_time', 'airport', 'y', 'raw_flight_id_missing', 'rule_triggered']].copy()
            meta['hour'] = pd.to_datetime(meta.mvt_time, utc=True).dt.hour
            meta['day'] = pd.to_datetime(meta.mvt_time, utc=True).dt.dayofyear
        per[s] = (f[COL].to_numpy(float), f[ALT].to_numpy(float))
    data[fold] = dict(meta=meta, month=info['val'], pred=per)
print('loaded', {k: len(v['meta']) for k, v in data.items()})

def score(fn=None):
    """fn(fold, seed, meta, pred) -> modified prediction; None = baseline."""
    out = {}
    for name, folds in (('R_A', A), ('R_B_long', BL)):
        months = {}
        for fold in folds:
            d = data[fold]
            sq = np.stack([((fn(fold, s, d['meta'], d['pred'][s][0]) if fn else d['pred'][s][0])
                            - d['meta'].y.to_numpy(float)) ** 2 for s in SEEDS])
            months.setdefault(d['month'], []).append(sq)
        pooled = [np.mean(v, axis=0) for v in months.values()]
        out[name] = float(np.sqrt(np.concatenate(pooled, axis=1).mean()))
    out['R_combined'] = float(np.sqrt(W * out['R_A'] ** 2 + (1 - W) * out['R_B_long'] ** 2))
    return out

base = score()
print('baseline', {k: round(v, 4) for k, v in base.items()},
      '(ledger 245.9817 / 222.5802 / 233.2338)')

def delta(label, fn):
    s = score(fn)
    d = {k: s[k] - base[k] for k in s}
    print(f'{label:<46} R_A {s["R_A"]:9.4f} ({d["R_A"]:+7.4f})   '
          f'R_B_long {s["R_B_long"]:9.4f} ({d["R_B_long"]:+7.4f})   '
          f'comb {s["R_combined"]:9.4f} ({d["R_combined"]:+7.4f})')
    return d['R_combined']

# ---- 1. rule rows -------------------------------------------------------
def oracle_rule(fold, seed, meta, p):
    p = p.copy(); m = meta.rule_triggered.to_numpy(bool)
    p[m] = meta.y.to_numpy(float)[m]; return p
delta('1a perfect on every rule-triggered row', oracle_rule)

BANDS = [(4, 6), (6, 10), (10, 14), (14, 18), (18, 96)]
def oracle_band_constant(fold, seed, meta, p):
    q = p.copy(); m = meta.rule_triggered.to_numpy(bool); y = meta.y.to_numpy(float)
    for lo, hi in BANDS:                      # rule predicts the schedule gap
        sel = m & (p >= lo * 3600) & (p < hi * 3600)
        if sel.sum(): q[sel] = y[sel].mean()   # best constant per band (in-sample)
    return q
delta('1b best constant per rule delay band', oracle_band_constant)

# ---- 2. worst-row concentration ----------------------------------------
def oracle_big(fold, seed, meta, p):
    q = p.copy(); y = meta.y.to_numpy(float); m = y > 50000
    q[m] = y[m]; return q
delta('2  perfect on every label > 50,000 s', oracle_big)

# ---- 3. group bias corrections -----------------------------------------
def bias(keys, honest, k=50.0):
    def fn(fold, seed, meta, p):
        y = meta.y.to_numpy(float); r = y - p
        key = pd.MultiIndex.from_arrays([meta[c] for c in keys])
        df = pd.DataFrame({'r': r, 'k': key.factorize()[0], 'half': (meta.day.to_numpy() % 2)})
        adj = np.zeros(len(df))
        halves = [0, 1] if honest else [None]
        for h in halves:
            fit = df if h is None else df[df.half != h]
            g = fit.groupby('k').r.agg(['mean', 'size'])
            shrunk = g['mean'] * g['size'] / (g['size'] + k)
            tgt = slice(None) if h is None else (df.half == h).to_numpy()
            adj[tgt] = df.k[tgt].map(shrunk).fillna(0.0).to_numpy()
        return p + adj
    return fn

delta('3a airport bias, oracle (in-sample)', bias(['airport'], False))
delta('3b airport bias, honest day-split', bias(['airport'], True))
delta('3c airport x hour, oracle', bias(['airport', 'hour'], False))
delta('3d airport x hour, honest day-split', bias(['airport', 'hour'], True))
delta('3e airport x hour x no-ID, oracle', bias(['airport', 'hour', 'raw_flight_id_missing'], False))
delta('3f airport x hour x no-ID, honest day-split', bias(['airport', 'hour', 'raw_flight_id_missing'], True))

# ---- 4. seed averaging / blending --------------------------------------
def seed_avg(fold, seed, meta, p):
    return np.mean([data[fold]['pred'][s][0] for s in SEEDS], axis=0)
delta('4a average the three seeds', seed_avg)

for w in (0.25, 0.5):
    delta(f'4b blend {1-w:.2f}*row43 + {w:.2f}*row34-stack',
          lambda fold, seed, meta, p, w=w: (1 - w) * p + w * data[fold]['pred'][seed][1])

# ---- 5. where the squared error sits ------------------------------------
rows = []
for fold in A + BL:
    d = data[fold]; y = d['meta'].y.to_numpy(float)
    for s in SEEDS:
        e = (d['pred'][s][0] - y) ** 2
        rows.append(pd.DataFrame({'se': e, 'y': y, 'rule': d['meta'].rule_triggered.to_numpy(),
                                  'noid': d['meta'].raw_flight_id_missing.to_numpy(),
                                  'ap': d['meta'].airport.to_numpy(), 'role': 'A' if fold in A else 'B'}))
all_rows = pd.concat(rows)
tot = all_rows.se.sum()
print('\nshare of squared error (all folds pooled)')
for label, mask in [('rule-triggered rows', all_rows.rule),
                    ('label > 50,000 s', all_rows.y > 50000),
                    ('label > 2 h, no rule', (all_rows.y > 7200) & ~all_rows.rule),
                    ('no flight ID', all_rows.noid),
                    ('LTFM', all_rows.ap == 'LTFM'),
                    ('LIRF', all_rows.ap == 'LIRF'),
                    ('ordinary: ID present, label < 2 h',
                     ~all_rows.noid & (all_rows.y <= 7200) & ~all_rows.rule)]:
    sub = all_rows[mask]
    print(f'  {label:<34} {len(sub):>9,} rows  {100*sub.se.sum()/tot:6.2f}%  '
          f'RMSE {np.sqrt(sub.se.mean()):9.1f}')
