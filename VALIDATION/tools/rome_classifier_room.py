"""How much Jan/Jul CV is left in the Rome classifier? Saved predictions only, no fits.

Base: v48's recipe (0.6 x mean of five `ple` networks + 0.4 x LightGBM, seeds 42/142/242, from
run v3_20261003_nn_improve_jan_jul). At LIRF a held-out flight counts as substituted when its
label is within 60 s of takeoff minus schedule, the definition the classifier was trained on.
The final prediction is rebuilt with three versions of the probability p:
  current       the classifier's p (held-out months, as scored on Jan/Jul CV)
  recalibrated  p mapped to the substitution rate of its 20-quantile bin in the other month
  oracle        1 for substituted flights, 0 otherwise: the most a classifier can give
Writes VALIDATION/reports/rome_classifier_room.png and .json.
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]
import rome_hybrid
import tuning_seasonal as ts
from records import write_json

SPLITS_FROM = ts.RUNS / 'v3_20261002_adsblol_jan_jul'
V47 = ts.RUNS / 'v3_20261002_nn_v46_jan_jul' / 'predictions' / 'nn_seeds42_142_242.parquet'
NETS = ts.RUNS / 'v3_20261003_nn_improve_jan_jul' / 'networks'
OUT = ROOT / 'VALIDATION' / 'reports' / 'rome_classifier_room'
SEEDS, WEIGHT, BINS = [42, 142, 242], 0.6, 20
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
BLUE, ORANGE = '#2a78d6', '#eb6834'


def auc(score, positive):
    r = pd.Series(score).rank().to_numpy()
    n1, n0 = positive.sum(), (~positive).sum()
    return float((r[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    v47 = pd.read_parquet(V47)
    months = {}
    for month in ts.SPLITS:
        d = ts.split_dir(SPLITS_FROM, month)
        held, meta = pd.read_pickle(d / 'heldout.pkl'), ts.read(d / 'meta.json')
        raw, V = held['raw'], held['V']
        ids = raw.mvt_id.to_numpy()
        saved = v47[v47.month == month].set_index('mvt_id').reindex(ids)
        y = raw.taxi_sec.to_numpy(float)
        gap = V[rome_hybrid.GAP].to_numpy(float)
        rome = (raw.ADEP_mvt.to_numpy() == rome_hybrid.AIRPORT) & np.isfinite(gap)
        bases = {}
        for s in SEEDS:
            nets = [pd.read_parquet(NETS / f'ple_{month}_s{s}_k{k}.parquet') for k in range(5)]
            if any(not np.array_equal(n.mvt_id.to_numpy(), ids) for n in nets):
                raise ValueError(f'{month}: network rows differ from the held-out rows')
            nn_base = np.mean([n.base.to_numpy(float) for n in nets], axis=0)
            bases[s] = WEIGHT * nn_base + (1 - WEIGHT) * saved[f'lgb_base_{s}'].to_numpy(float)
        final = lambda base, p, held=held, raw=raw, meta=meta: ts.final_prediction(
            base, p, held['V'], raw, meta['parameters'], meta['corrections'])
        months[month] = dict(y=y, gap=gap, rome=rome, p=np.asarray(held['p'], float), bases=bases, final=final,
                             sub=rome & (np.abs(gap - y) <= rome_hybrid.MATCH_SEC))
    # Recalibration: each month's p mapped through the other month's binned substitution rate.
    for month, other in zip(ts.SPLITS, ts.SPLITS[::-1]):
        o, m = months[other], months[month]
        po, so = o['p'][o['rome']], o['sub'][o['rome']]
        edges = np.unique(np.quantile(po, np.linspace(0, 1, BINS + 1)))
        rate = np.array([so[(po >= a) & (po <= b)].mean() for a, b in zip(edges[:-1], edges[1:])])
        idx = np.clip(np.searchsorted(edges, m['p'], side='right') - 1, 0, len(rate) - 1)
        m['p_recal'] = np.where(m['rome'], rate[idx], m['p'])
        m['p_oracle'] = np.where(m['rome'], m['sub'].astype(float), m['p'])
    out = dict(base='v48 recipe: 0.6 ple networks + 0.4 LightGBM', months={}, scenarios={})
    se = {k: {} for k in ('current', 'recalibrated', 'oracle')}
    for month, m in months.items():
        r, sub, p = m['rome'], m['sub'], m['p']
        for name, pv in (('current', p), ('recalibrated', m['p_recal']), ('oracle', m['p_oracle'])):
            se[name][month] = np.mean([(m['y'] - m['final'](m['bases'][s], pv)) ** 2 for s in SEEDS], axis=0)
        e = se['current'][month]
        out['months'][month] = dict(
            rows=int(len(r)), rome_rows=int(r.sum()), substituted=int(sub.sum()),
            substitution_rate=float(sub[r].mean()), mean_p=float(p[r].mean()), auc=auc(p[r], sub[r]),
            brier=float(((p[r] - sub[r]) ** 2).mean()),
            rome_sse_share=float(e[r].sum() / e.sum()), rome_row_share=float(r.mean()),
            rome_rmse=float(np.sqrt(e[r].mean())), rome_rmse_substituted=float(np.sqrt(e[sub].mean())),
            rome_rmse_ordinary=float(np.sqrt(e[r & ~sub].mean())), other_rmse=float(np.sqrt(e[~r].mean())))
    for name in se:
        level = {mo: float(np.sqrt(se[name][mo].mean())) for mo in ts.SPLITS}
        rome = {mo: float(np.sqrt(se[name][mo][months[mo]['rome']].mean())) for mo in ts.SPLITS}
        out['scenarios'][name] = dict(jan_jul_cv=float(np.mean(list(level.values()))), months=level, rome_rmse=rome)
    cur = out['scenarios']['current']['jan_jul_cv']
    for name, v in out['scenarios'].items():
        v['change_vs_current'] = v['jan_jul_cv'] - cur
    # Where Rome's squared error sits: by p decile and true class (both months pooled).
    p_all = np.concatenate([m['p'][m['rome']] for m in months.values()])
    s_all = np.concatenate([m['sub'][m['rome']] for m in months.values()])
    e_all = np.concatenate([se['current'][mo][months[mo]['rome']] for mo in ts.SPLITS])
    gap_err = np.concatenate([np.abs(m['gap'] - m['y'])[m['rome']] for m in months.values()])
    edges = np.linspace(0, 1, 11)
    b = np.clip(np.digitize(p_all, edges) - 1, 0, 9)
    table = pd.DataFrame(dict(bin=b, p=p_all, sub=s_all, se=e_all)).groupby('bin').agg(
        n=('sub', 'size'), mean_p=('p', 'mean'), rate=('sub', 'mean'),
        sse_sub=('se', lambda x: x[s_all[x.index]].sum()), sse_ord=('se', lambda x: x[~s_all[x.index]].sum()))
    total = e_all.sum()
    table['share_sub'], table['share_ord'] = table.sse_sub / total, table.sse_ord / total
    out['by_p_decile'] = table.reset_index().to_dict(orient='records')
    out['ordinary_rows_gap_far'] = float((gap_err[~s_all] > 600).mean())
    write_json(OUT.with_suffix('.json'), out)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': GRID,
                         'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                         'axes.titlecolor': INK, 'axes.titlesize': 10, 'axes.titleweight': 'bold'})
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.6), facecolor=SURFACE)
    for a in ax.flat:
        a.set_facecolor(SURFACE)
        a.grid(axis='y', color=GRID, lw=0.6)
        a.set_axisbelow(True)
        for side in ('top', 'right'):
            a.spines[side].set_visible(False)
    # A: p by true class
    a = ax[0, 0]
    hb = np.linspace(0, 1, 41)
    a.hist(p_all[~s_all], hb, color=BLUE, histtype='step', lw=2, label=f'not substituted ({(~s_all).sum():,})')
    a.hist(p_all[s_all], hb, color=ORANGE, histtype='step', lw=2, label=f'substituted ({s_all.sum():,})')
    a.set_yscale('log')
    a.set(xlabel='classifier p', ylabel='Rome flights (log scale)')
    a.set_title('A. p for each true class', loc='left')
    a.legend(frameon=False, labelcolor=INK2)
    # B: reliability
    a = ax[0, 1]
    a.plot([0, 1], [0, 1], color=INK2, lw=1, ls='--')
    a.plot(table.mean_p, table.rate, color=BLUE, lw=2, marker='o', ms=6, mec=SURFACE, mew=2)
    for _, r_ in table.iterrows():
        a.annotate(f'{int(r_.n):,}', (r_.mean_p, r_.rate), textcoords='offset points', xytext=(6, -10),
                   fontsize=7, color=INK2)
    a.set(xlim=(0, 1), ylim=(0, 1), xlabel='mean p in bin', ylabel='share actually substituted')
    a.set_title('B. Calibration (flights per bin labelled)', loc='left')
    # C: SSE share by p bin and class
    a = ax[1, 0]
    x = table.index.to_numpy() / 10 + 0.05
    a.bar(x, table.share_ord * 100, width=0.09, color=BLUE, label='not substituted', edgecolor=SURFACE, lw=1)
    a.bar(x, table.share_sub * 100, width=0.09, bottom=table.share_ord * 100, color=ORANGE,
          label='substituted', edgecolor=SURFACE, lw=1)
    a.set(xlim=(0, 1), xlabel='classifier p (deciles)', ylabel="% of Rome's squared error")
    a.set_title("C. Where Rome's error sits", loc='left')
    a.legend(frameon=False, labelcolor=INK2)
    # D: room, as the change from the current classifier
    a = ax[1, 1]
    names = ['recalibrated', 'oracle']
    vals = [out['scenarios'][n]['change_vs_current'] for n in names]
    yy = np.arange(len(names))[::-1]
    a.barh(yy, vals, height=0.45, color=[BLUE, ORANGE])
    a.axvline(0, color=INK2, lw=1)
    for yi, v in zip(yy, vals):
        a.text(v + (0.4 if v >= 0 else -0.4), yi, f'{v:+.2f} s', va='center', ha='left' if v >= 0 else 'right',
               color=INK)
    a.set_xlim(min(vals) - 6, 4)
    a.set_yticks(yy, ['recalibrated p\n(other month)', 'perfect classifier'])
    a.grid(axis='y', visible=False)
    a.grid(axis='x', color=GRID, lw=0.6)
    a.set_xlabel(f'change in Jan/Jul CV (s) from {cur:.2f} (v48 recipe)')
    a.set_title('D. Room for improvement', loc='left')
    fig.suptitle('Rome substitution classifier on Jan/Jul 2025 held-out flights', x=0.01, ha='left',
                 fontsize=12, fontweight='bold', color=INK)
    fig.tight_layout()
    fig.savefig(OUT.with_suffix('.png'), dpi=150, facecolor=SURFACE)
    print({k: round(v['jan_jul_cv'], 4) for k, v in out['scenarios'].items()})
    print({k: {mo: round(x, 1) for mo, x in v['rome_rmse'].items()} for k, v in out['scenarios'].items()})
    for mo, v in out['months'].items():
        print(mo, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items()})
    print(table.round(4).to_string())
    print('ordinary Rome rows with |gap - y| > 600 s:', round(out['ordinary_rows_gap_far'], 3))


if __name__ == '__main__':
    main()
