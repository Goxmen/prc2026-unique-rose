"""Residual atlas for officially scored recipes, using saved OOF predictions only.

Run from the repository root:
  .venv/bin/python VALIDATION/tools/plot_scored_residuals.py \
      --leaderboard tmp/residual_audit/leaderboard.json

No training, submission, label filtering, or protocol change. All 16 folds and
three seeds are checked. Main figures use A1-A6: one prediction per flight per
seed. Losses, not predictions, are averaged across seeds. Signed scatter plots
use seed 42 explicitly. Replayed recipes are distinguished from original OOF.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import gc
import hashlib
import html
import json
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline')]
from protocol import MAIN_NAMES
from records import clean, write_json
from summarize_artifact_errors import regime_labels
from record_fixes import twin_rows

RUNS = ROOT / 'VALIDATION/runs'
REPLAY = 'v3_20260919_artifact_control'
SEEDS = [42, 142, 242]
MODELS = {
    'v12': ('B0-800', 'v3_20260916T095948645262Z', 'pred', 'original'),
    'v13': ('Rome 6h rule', 'v3_20260916T123230629423Z', 'pred', 'original'),
    'v14': ('Recovery + 6h', 'v3_20260916T123433565511Z', 'pred', 'original'),
    'v15': ('Rome 4h rule', 'v3_20260916T164054429797Z', 'pred', 'original'),
    'v16': ('Recovery + 4h', 'v3_20260916T164247756836Z', 'pred', 'original'),
    'v19': ('Clock features', 'v3_20260916T132038893775Z', 'model_pred', 'reconstructed'),
    'v30': ('Stand clock', REPLAY, 'pred_stand', 'replay'),
    'v33': ('Plan window', REPLAY, 'pred_window', 'replay'),
    'v34': ('No-ID cap', REPLAY, 'pred_noid_cap', 'replay'),
}
CONTEXT = ['mvt_id', 'mvt_time', 'airport', 'y', 'FLIGHT_ID_mvt', 'FLIGHT_mvt',
           'ADEP_mvt', 'ADES_mvt', 'STAND_mvt', 'RUNWAY_mvt', 'AIRCRAFT_TYPE_mvt',
           'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'LOBT_flt', 'AOBT_3_flt',
           'raw_aobt3_missing', 'raw_plan_block_missing', 'gap_sec', 'p', 'ordinary',
           'rule_triggered', 'late_rows', 'mvt_minus_lobt_sec', 'mvt_minus_aobt3_sec',
           'clock_span_sec', 'q_queue_ahead_since_aobt3', 'q_rwy_prev_takeoff_gap_sec',
           'wx_visibility_m', 'wx_wind_kt', 'wx_deicing', 'st_last_arr_sec', 'airline']
COLORS = ['#8595a4', '#cb9549', '#a287b7', '#397ba9', '#5b9d91', '#6561b4',
          '#e07d4e', '#bd6084', '#067d7a']
FAMILIES = ['regime', 'airport', 'month', 'hour', 'target_band', 'schedule_band',
            'plan_clock_band', 'queue_band', 'visibility', 'wind', 'deicing', 'stand_age']


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def ledger():
    rows = []
    for line in (ROOT / 'SUBMISSIONS.md').read_text().splitlines():
        cells = [c.strip() for c in line.split('|')[1:-1]]
        if len(cells) != 8 or not cells[0].isdigit() or not re.fullmatch(r'v\d+', cells[-1]):
            continue
        try:
            score = float(cells[6].replace('*', ''))
        except ValueError:
            continue
        rows.append(dict(version=cells[-1], name=cells[1].replace('*', ''), lb=score,
                         coverage=MODELS[cells[-1]][3] if cells[-1] in MODELS else 'unavailable',
                         cv_combined=None if cells[5] == '—' else float(cells[5].replace('*', '').replace('†', ''))))
    return rows


def with_live_inventory(official, lb):
    """Include scored uploads newer than the checked-in ledger; never infer recipes."""
    known = {r['version']: r for r in official}
    for row in lb.get('history', []):
        if row.get('status') != 'Succeeded' or row.get('score') is None:
            continue
        version = 'v' + str(row['version'])
        if version in known:
            assert abs(float(row['score']) - known[version]['lb']) < 1e-5
        else:
            known[version] = dict(version=version, name='Recipe not documented locally',
                                  lb=float(row['score']), coverage='unavailable; recipe unverified',
                                  cv_combined=None)
    return sorted(known.values(), key=lambda r: int(r['version'][1:]))


def context():
    parts = []
    for i in range(1, 7):
        d = pd.read_parquet(RUNS / REPLAY / f'oof_A{i}_seed42.parquet', columns=CONTEXT)
        d['regime'] = regime_labels(d, twin_rows(d))
        parts.append(d)
    d = pd.concat(parts, ignore_index=True).set_index('mvt_id').sort_index()
    assert d.index.is_unique and not d.index.hasnans
    d['noid'] = d.FLIGHT_ID_mvt.isna()
    d['month'] = d.mvt_time.dt.strftime('%Y-%m')
    d['hour'] = d.mvt_time.dt.hour.map(lambda h: f'{h:02d}')
    d['date'] = d.mvt_time.dt.strftime('%Y-%m-%d')
    cuts = {
        'target_band': ('y', [-np.inf, 0, 600, 1200, 1800, 3600, 7200, 43200, np.inf],
                        ['≤0', '0–10m', '10–20m', '20–30m', '30–60m', '1–2h', '2–12h', '>12h']),
        'schedule_band': ('gap_sec', [-np.inf, 0, 1800, 3600, 14400, 50400, np.inf],
                          ['≤0', '0–30m', '30–60m', '1–4h', '4–14h', '>14h']),
        'plan_clock_band': ('mvt_minus_aobt3_sec', [-np.inf, 0, 900, 1800, 3600, 7200, np.inf],
                            ['≤0', '0–15m', '15–30m', '30–60m', '1–2h', '>2h']),
        'queue_band': ('q_queue_ahead_since_aobt3', [-np.inf, 0, 5, 10, 20, 40, np.inf],
                       ['0', '1–5', '6–10', '11–20', '21–40', '>40']),
        'visibility': ('wx_visibility_m', [-np.inf, 1000, 5000, 9999, np.inf],
                       ['≤1km', '1–5km', '5–10km', '≥10km']),
        'wind': ('wx_wind_kt', [-np.inf, 5, 10, 20, 30, np.inf],
                 ['≤5kt', '5–10kt', '10–20kt', '20–30kt', '>30kt']),
        'stand_age': ('st_last_arr_sec', [-np.inf, 1800, 3600, 7200, 21600, np.inf],
                      ['≤30m', '30–60m', '1–2h', '2–6h', '>6h']),
    }
    for key, (col, edges, labels) in cuts.items():
        d[key] = pd.cut(d[col], edges, labels=labels).cat.add_categories('missing').fillna('missing')
    d['deicing'] = d.wx_deicing.map({0: 'no', 1: 'yes'}).fillna('missing')
    return d


def read_model(version, d, provenance):
    name, run, col, kind = MODELS[version]
    recpath = RUNS / run / 'result.json'
    rec = json.loads(recpath.read_text())
    arrays = {k: np.full((3, len(d)), np.nan) for k in ['err', 'long_sq']}
    fold_stats, long_count = {}, np.zeros(len(d), dtype=int)
    fold_slices = []
    for fold in MAIN_NAMES:
        per_seed = []
        for j, seed in enumerate(SEEDS):
            path = RUNS / run / rec['folds'][fold]['predictions'][str(seed)]
            cols = ['mvt_id', 'mvt_time', 'airport', 'y', 'fold', 'seed', col]
            if version == 'v19':
                cols += ['raw_flight_id_missing', 'rule_gap_sec']
            f = pd.read_parquet(path, columns=cols).set_index('mvt_id').sort_index()
            assert f.index.is_unique and not f.index.hasnans
            pos = d.index.get_indexer(f.index)
            assert (pos >= 0).all() and f.fold.eq(fold).all() and f.seed.eq(seed).all()
            meta = d.iloc[pos]
            for c in ['mvt_time', 'airport', 'y']:
                assert np.array_equal(f[c].to_numpy(), meta[c].to_numpy()), (version, fold, seed, c)
            pred = f[col].to_numpy(float).copy()
            if version == 'v19':
                mask = (f.airport.eq('LIRF') & f.raw_flight_id_missing & f.rule_gap_sec.ge(14400)).to_numpy()
                pred[mask] = f.rule_gap_sec.to_numpy()[mask]
            e = pred - f.y.to_numpy(float)
            assert np.isfinite(e).all()
            if fold.startswith('A'):
                assert np.isnan(arrays['err'][j, pos]).all()
                arrays['err'][j, pos] = e
            if fold in ['B3', 'B4', 'B7']:
                arrays['long_sq'][j, pos] = np.nan_to_num(arrays['long_sq'][j, pos]) + e**2
                if j == 0:
                    long_count[pos] += 1
            per_seed.append(e**2)
            provenance.append(dict(model=version, file=str(path.relative_to(ROOT)), sha256=digest(path)))
        loss = np.mean(per_seed, axis=0)
        fold_stats[fold] = float(np.sqrt(loss.mean()))
        # Retain all 16 folds, with per-slice counts for sensitivity inspection.
        for family in ['airport', 'regime']:
            t = pd.DataFrame({'key': meta[family].to_numpy(), 'sq': loss})
            for key, part in t.groupby('key', observed=True):
                fold_slices.append(dict(model=version, fold=fold, family=family, key=str(key),
                                        n=len(part), rmse=float(np.sqrt(part.sq.mean())), sse=float(part.sq.sum())))
    err = arrays['err']
    assert np.isfinite(err).all()
    long_mask = long_count > 0
    long_sq = arrays['long_sq'][:, long_mask] / long_count[long_mask]
    a = float(np.sqrt((err**2).mean()))
    b = float(np.sqrt(long_sq.mean()))
    w = rec['ranking_weights']['jan_fraction']
    scores = dict(R_A=a, R_B_long=b, R_combined=float(np.sqrt(w*a*a + (1-w)*b*b)))
    # Exact saved run reproduction applies to original OOF and row-34 replay.
    if kind == 'original' or version == 'v34':
        for k, v in scores.items():
            assert np.isclose(v, rec['report']['scores'][k], atol=1e-8, rtol=0), (version, k, v)
    return dict(name=name, kind=kind, run=run, column=col, err=err[0].copy(),
                sq=(err**2).mean(axis=0), bias=err.mean(axis=0), long_sq=long_sq.mean(axis=0),
                long_mask=long_mask, scores=scores, folds=fold_stats, fold_slices=fold_slices,
                seed_scores=np.sqrt((err**2).mean(axis=1)).tolist(),
                seed_noid_share=[float((e[d.noid.to_numpy()]**2).sum() / (e**2).sum()*100) for e in err])


def aggregate(d, m, family, scope='all'):
    mask = np.ones(len(d), bool) if scope == 'all' else d.airport.eq(scope).to_numpy()
    # Operational diagnostics conditioned on ID availability are explicitly labelled.
    f = pd.DataFrame(dict(key=d[family].reset_index(drop=True).loc[mask].reset_index(drop=True), sq=m['sq'][mask],
                          err=m['bias'][mask], pred=(d.y.to_numpy()+m['err'])[mask],
                          abs42=np.abs(m['err'][mask]), day=d.date.to_numpy()[mask]))
    total = f.sq.sum()
    g = f.groupby('key', observed=True, dropna=False).agg(
        n=('sq', 'size'), days=('day', 'nunique'), sse=('sq', 'sum'), bias=('err', 'mean'),
        mae42=('abs42', 'mean'), mean_prediction42=('pred', 'mean'))
    g['rmse'] = np.sqrt(g.sse/g.n)
    g['error_share'] = g.sse/total*100
    g['flight_share'] = g.n/len(f)*100
    g['mse_contribution'] = g.sse/len(f)
    return [dict(key=str(k), **clean(row.to_dict())) for k, row in g.iterrows()]


def metrics(d, m):
    e, sq = m['err'], m['sq']
    sort = np.sort(sq)[::-1]
    n = len(sq)
    ranks = np.unique(np.geomspace(1, n, 220).astype(int))
    cumulative = np.cumsum(sort)
    thresholds = np.unique(np.r_[np.geomspace(1, max(abs(e).max(), 2), 180), 300, 600, 3600])
    absolute = np.sort(np.abs(e))
    # Full tails remain represented; zero errors are included in the denominator.
    survival = (n-np.searchsorted(absolute, thresholds, side='right'))/n*100
    return dict(scores=m['scores'], seed_scores=m['seed_scores'], folds=m['folds'],
                bias=float(m['bias'].mean()), mae42=float(abs(e).mean()),
                abs_quantiles42=dict(zip(['p50','p90','p95','p99','p99.9','max'],
                                         np.quantile(abs(e), [.5,.9,.95,.99,.999,1]).tolist())),
                noid_error_share=float(sq[d.noid.to_numpy()].sum()/sq.sum()*100),
                noid_share_by_seed=m['seed_noid_share'],
                over_error_share42=float((e[e>0]**2).sum()/(e**2).sum()*100),
                under_error_share42=float((e[e<0]**2).sum()/(e**2).sum()*100),
                tail_over_hour_pct42=float(np.mean(abs(e)>3600)*100),
                top_shares={str(k):float(sort[:k].sum()/sq.sum()*100) for k in [10,100,1000,10000]},
                concentration=[[int(k), float(cumulative[k-1]/sq.sum()*100)] for k in ranks],
                survival=[[float(x),float(y)] for x,y in zip(thresholds,survival)])


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False,
                         'axes.spines.right':False, 'axes.titlelocation':'left', 'axes.titlesize':12,
                         'axes.labelcolor':'#405467', 'text.color':'#173145', 'axes.edgecolor':'#ced9df',
                         'grid.color':'#e5ebee', 'grid.alpha':.8, 'figure.facecolor':'#fafcfc',
                         'axes.facecolor':'#fafcfc', 'savefig.facecolor':'#fafcfc', 'legend.frameon':False})


def figures(d, models, summary, out):
    style()
    pages = []
    def save(fig, name, title, note):
        fig.suptitle(title+'\n'+note, x=.025, ha='left', fontsize=15, linespacing=1.6)
        fig.savefig(out/(name+'.png'), dpi=155, bbox_inches='tight')
        pdf.savefig(fig, bbox_inches='tight')
        pages.append((name,title,note))
        plt.close(fig)
    def legend(ax):
        ax.legend(fontsize=8)
        ax.grid(alpha=.4)
    picks = ['v12','v15','v19','v30','v33','v34']
    color = dict(zip(models, COLORS))
    pretty = lambda v: v+('†' if MODELS[v][3]!='original' else '')
    with PdfPages(out/'residual_atlas.pdf') as pdf:
        fig, ax = plt.subplots(1, 2, figsize=(15, 5.8), layout='constrained')
        history = summary['official']
        for available,c in [(False,'#b9c4cb'),(True,'#067d7a')]:
            pts=[r for r in history if (r['version'] in models)==available]
            ax[0].scatter([int(r['version'][1:]) for r in pts],[r['lb'] for r in pts],color=c,
                          label='Residuals available' if available else 'No local residuals')
        for r in history:
            ax[0].annotate(r['version'],(int(r['version'][1:]),r['lb']),xytext=(0,5),
                           textcoords='offset points',fontsize=7,ha='center')
        ax[0].set(xlabel='Submission version',ylabel='Official LB RMSE (s)',title='All officially scored submissions')
        legend(ax[0])
        for metric,marker in [('R_A','o'),('R_B_long','s')]:
            ax[1].plot(range(len(models)),[m['scores'][metric] for m in models.values()],marker=marker,label=metric)
        ax[1].set_xticks(range(len(models)),[pretty(v) for v in models],rotation=30)
        ax[1].set(ylabel='Validation RMSE (s)',title='Matched validation: next month and longer gaps')
        legend(ax[1])
        save(fig,'01_coverage','Which scored recipes can we inspect?',
             '† = documented local replay / reconstruction; LB labels remain hidden. CV and LB have different populations.')

        fig, axes = plt.subplots(2,2,figsize=(15,10),layout='constrained')
        for v in picks:
            m=models[v]; s=summary['models'][v]; e=m['err']
            edges=np.linspace(-900,900,91)
            hist,_=np.histogram(e,edges)
            axes[0,0].plot((edges[1:]+edges[:-1])/2,hist/len(e)*100,color=color[v],label=pretty(v))
            arr=np.array(s['survival']);good=arr[:,1]>0
            axes[0,1].plot(arr[good,0],arr[good,1],color=color[v],label=pretty(v))
            arr=np.array(s['concentration'])
            axes[1,0].plot(arr[:,0]/len(d)*100,arr[:,1],color=color[v],label=pretty(v))
            q=pd.qcut(d.y.to_numpy()+e,20,duplicates='drop')
            f=pd.DataFrame({'q':q,'pred':d.y.to_numpy()+e,'e':e})
            g=f.groupby('q',observed=True).agg(pred=('pred','mean'),bias=('e','mean'))
            axes[1,1].plot(g.pred,g.bias,marker='.',color=color[v],label=pretty(v))
        axes[0,0].axvline(0,color='#80909a',lw=.8)
        axes[0,0].set(title='Signed residuals: central view (seed 42)',xlabel='Prediction − actual (s); positive = too high',ylabel='% of ALL flights per 20-second bin')
        axes[0,1].set(xscale='log',yscale='log',title='Full absolute-error tail (seed 42)',xlabel='Absolute residual (s)',ylabel='% of flights exceeding this error')
        axes[1,0].set(xscale='log',title='Few flights carry much of the squared error',xlabel='% of distinct flights, worst first',ylabel='Cumulative squared-error share (%)')
        axes[1,1].axhline(0,color='#80909a',lw=.8)
        axes[1,1].set(title='Calibration by predicted value (seed 42)',xlabel='Mean prediction in each 5% bin (s)',ylabel='Mean prediction − actual (s)')
        for ax in axes.flat:legend(ax)
        save(fig,'02_residual_shapes','Typical errors and catastrophic tails need different explanations',
             'A1–A6 only; each flight once. Tail plot contains all errors. Concentration averages losses over three seeds.')

        fig,axes=plt.subplots(1,3,figsize=(17,6.4),layout='constrained')
        apt=sorted(d.airport.unique(),key=lambda a:-models['v34']['sq'][d.airport.eq(a)].sum())
        for v in picks:
            table={r['key']:r for r in summary['slices'][v]['all']['airport']}
            axes[0].plot([table[a]['rmse'] for a in apt],range(len(apt)),'.-',color=color[v],label=pretty(v))
            axes[1].plot([table[a]['bias'] for a in apt],range(len(apt)),'.-',color=color[v],label=pretty(v))
        t={r['key']:r for r in summary['slices']['v34']['all']['airport']}
        axes[2].barh(np.arange(len(apt))-.18,[t[a]['error_share'] for a in apt],.35,color='#067d7a',label='Squared error')
        axes[2].barh(np.arange(len(apt))+.18,[t[a]['flight_share'] for a in apt],.35,color='#aebec9',label='Flights')
        for ax in axes:ax.set_yticks(range(len(apt)),apt);ax.invert_yaxis();legend(ax)
        axes[0].set(title='Airport RMSE',xlabel='RMSE (s)')
        axes[1].set(title='Airport bias',xlabel='Mean prediction − actual (s)');axes[1].axvline(0,color='gray',lw=.8)
        axes[2].set(title='v34†: error share vs traffic share',xlabel='Share (%)')
        save(fig,'03_airports','Where the models miss, and where errors cost the most',
             'Next-month flights; three-seed loss/bias averages. Airport differences alone do not identify their causes.')

        fig,axes=plt.subplots(1,2,figsize=(16,6.8),layout='constrained')
        regimes=[r['key'] for r in summary['slices']['v34']['all']['regime']]
        matrix=np.array([[next(r['mse_contribution'] for r in summary['slices'][v]['all']['regime'] if r['key']==s) for v in picks] for s in regimes])
        im=axes[0].imshow(matrix,aspect='auto',norm=LogNorm(vmin=max(matrix[matrix>0].min(),1),vmax=matrix.max()),cmap='YlOrRd')
        axes[0].set_xticks(range(len(picks)),[pretty(v) for v in picks]);axes[0].set_yticks(range(len(regimes)),regimes)
        for i in range(len(regimes)):
            for j in range(len(picks)):
                axes[0].text(j,i,f'{matrix[i,j]:,.0f}',ha='center',va='center',fontsize=8,color='white' if matrix[i,j]>matrix.max()/3 else '#173145')
        fig.colorbar(im,ax=axes[0],label='Contribution to total MSE (s²); columns sum to R_A²',shrink=.6)
        rows=summary['slices']['v34']['all']['regime']
        axes[1].barh(range(len(rows)),[r['error_share'] for r in rows],color='#067d7a')
        axes[1].set_yticks(range(len(rows)),[f"{r['key']}\nn={r['n']:,}; {r['days']} days; RMSE {r['rmse']:,.0f}s" for r in rows],fontsize=9)
        axes[1].invert_yaxis();axes[1].set(xlabel='% of total squared error',title='v34†: observable regimes and sample sizes')
        axes[1].grid(axis='x')
        save(fig,'04_regimes','Separate recording regimes from ordinary operational errors',
             'Disjoint groups defined using observable inputs only. No target-defined route or label filtering.')

        fig,axes=plt.subplots(2,2,figsize=(15,10),layout='constrained')
        m=models['v34'];err=m['err'];pred=d.y.to_numpy()+err
        hb=axes[0,0].hexbin(pred,err,gridsize=70,mincnt=1,bins='log',extent=(0,3600,-1800,1800),cmap='viridis')
        center=(pred>=0)&(pred<=3600)&(abs(err)<=1800)
        fig.colorbar(hb,ax=axes[0,0],label='Number of flights (log colour)')
        axes[0,0].axhline(0,color='#efb150',lw=1)
        axes[0,0].set(xlim=(0,3600),ylim=(-1800,1800),title=f'v34†: central residual cloud ({center.mean():.2%} in view)',xlabel='Prediction (s)',ylabel='Prediction − actual (s)')
        noid=d.noid.to_numpy();rome=d.airport.eq('LIRF').to_numpy();mask=noid&rome
        gap=d.gap_sec.to_numpy()/3600
        axes[0,1].scatter(gap[mask],err[mask]/3600,c=np.where(d.rule_triggered[mask],1,0),cmap='coolwarm',s=18,alpha=.7)
        for x in [1,4,14]:axes[0,1].axvline(x,color='gray',ls=':',lw=1)
        axes[0,1].axhline(0,color='gray',lw=.7)
        axes[0,1].set(title=f'All {mask.sum():,} Rome flights without an ID',xlabel='Takeoff − scheduled time (hours)',ylabel='Prediction − actual (hours)')
        axes[0,1].text(.03,.04,'Red = hard rule; blue = model / late estimate',transform=axes[0,1].transAxes,fontsize=9)
        axes[1,0].scatter(gap[mask],d.y.to_numpy()[mask]/3600,c=np.where(d.rule_triggered[mask],1,0),cmap='coolwarm',s=18,alpha=.7)
        top=max(np.nanmax(gap[mask]),np.max(d.y.to_numpy()[mask]/3600))
        axes[1,0].plot([0,top],[0,top],color='gray',ls='--',label='Label = schedule gap')
        axes[1,0].axhline(24,color='#067d7a',ls=':',label='One day')
        axes[1,0].set(title='Three competing label patterns in Rome',xlabel='Takeoff − scheduled time (hours)',ylabel='Recorded taxi time (hours)');legend(axes[1,0])
        order=np.argsort(-m['sq'])[:20]
        for v in ['v12','v15','v30','v34']:
            axes[1,1].plot(models[v]['err'][order]/3600,range(len(order)),'.',label=pretty(v),color=color[v])
        axes[1,1].set_yticks(range(len(order)),[f'{d.airport.iloc[i]} {d.mvt_time.iloc[i]:%m-%d %H:%M}' for i in order],fontsize=8)
        axes[1,1].invert_yaxis();axes[1,1].axvline(0,color='gray',lw=.8)
        axes[1,1].set(title='Same 20 worst v34† flights, compared across models',xlabel='Prediction − actual (hours)');legend(axes[1,1])
        save(fig,'05_extreme_flights','What do the worst residuals actually look like?',
             'Signed points use seed 42. Worst-flight ranking uses mean squared error across all three seeds.')

        # Conditional operational patterns. Keep ID-linked rows explicit, with airport controls.
        fig,axes=plt.subplots(2,2,figsize=(15,10),layout='constrained')
        operational=d.loc[~d.noid].copy()
        om={k:v[~noid] if isinstance(v,np.ndarray) and len(v)==len(d) else v for k,v in m.items()}
        for ax,family,title in [(axes[0,0],'hour','Hour of takeoff (UTC)'),(axes[0,1],'queue_band','Aircraft ahead since AOBT3'),
                                 (axes[1,0],'visibility','METAR visibility'),(axes[1,1],'plan_clock_band','Takeoff − AOBT3')]:
            allrows=aggregate(operational,om,family)
            labels=[r['key'] for r in allrows]
            for apt in ['LIRF','LFPG','LTFM','EGLL']:
                t={r['key']:r for r in aggregate(operational,om,family,apt)}
                ax.plot(range(len(labels)),[t[x]['rmse'] if x in t and t[x]['n']>=100 else np.nan for x in labels],'.-',label=apt)
            ax.set_xticks(range(len(labels)),labels,rotation=45 if len(labels)>10 else 20,fontsize=8)
            ax.set(title=title,xlabel='Observable condition',ylabel='RMSE (s), ID present only');legend(ax)
        save(fig,'06_operational','Recurring operational errors after separating no-ID records',
             'v34†, next-month, three seeds. Within-airport descriptive curves; cells below 100 flights are omitted from curves only.')

        fig,axes=plt.subplots(2,1,figsize=(15,9),layout='constrained')
        values=np.array([[m['folds'][f] for f in MAIN_NAMES] for m in models.values()])
        im=axes[0].imshow(values,aspect='auto',cmap='YlOrRd')
        axes[0].set_xticks(range(16),MAIN_NAMES);axes[0].set_yticks(range(len(models)),[pretty(v) for v in models])
        for i in range(len(models)):
            for j in range(16):axes[0].text(j,i,f'{values[i,j]:.0f}',ha='center',va='center',fontsize=8)
        fig.colorbar(im,ax=axes[0],label='RMSE (s)',shrink=.7)
        axes[0].set(title='All 16 validation folds, each averaging three-seed loss')
        for v in ['v12','v15','v30','v34']:
            t=pd.DataFrame({'date':d.date,'se':models[v]['sq']}).groupby('date').se.mean()
            axes[1].plot(pd.to_datetime(t.index),np.sqrt(t),label=pretty(v),color=color[v],alpha=.9)
        axes[1].set(title='Daily next-month RMSE exposes bursts',ylabel='RMSE (s)',xlabel='2025 validation date');legend(axes[1])
        save(fig,'07_time_and_folds','Do failures repeat across dates and training origins?',
             'All folds remain visible. November/December repeat in B; the main A charts never count a flight twice.')

        fig,axes=plt.subplots(1,2,figsize=(15,6),layout='constrained')
        for v in ['v15','v19','v30','v33']:
            e=models[v]['err'];latest=models['v34']['err']
            cutoff=np.quantile(models['v34']['sq'],.999)
            ix=np.unique(np.r_[np.linspace(0,len(d)-1,3500,dtype=int),np.flatnonzero(models['v34']['sq']>=cutoff)])
            if v in ['v15','v30']:
                axes[0].scatter(e[ix],latest[ix],s=5,alpha=.2,label=pretty(v)+' vs v34†')
        axes[0].set(xscale='symlog',yscale='symlog',xlabel='Earlier model residual (s)',ylabel='v34† residual (s)',title='Are the errors shared? (seed 42)')
        lo,hi=axes[0].get_xlim();axes[0].plot([lo,hi],[lo,hi],color='gray',ls=':');legend(axes[0])
        for v in ['v30','v33']:
            delta=models['v34']['sq']-models[v]['sq']
            by=pd.DataFrame({'regime':d.regime,'delta':delta}).groupby('regime',observed=True).delta.sum()/len(d)
            axes[1].plot(by.to_numpy(),range(len(by)),'.-',label=f'v34† minus {v}†')
        axes[1].set_yticks(range(len(by)),by.index);axes[1].invert_yaxis();axes[1].axvline(0,color='gray',lw=.8)
        axes[1].set(title='Where the last two corrections helped',xlabel='Change in total MSE contribution (s²); left = better');legend(axes[1])
        save(fig,'08_shared_errors','Persistent mistakes and the scope for combining models',
             'Scatter: deterministic sample plus all worst 0.1% v34 flights. MSE changes: every flight, all three seeds.')
    return pages


def tables_and_findings(d, models, summary, out):
    last=models['v34']; s=summary['models']['v34']; total=last['sq'].sum()
    worst=d[['mvt_time','airport','FLIGHT_mvt','STAND_mvt','RUNWAY_mvt','regime','y','gap_sec',
             'mvt_minus_lobt_sec','mvt_minus_aobt3_sec','p']].copy()
    for v,m in models.items():worst[v+'_residual_seed42']=m['err']
    worst['v34_mean_sq_error']=last['sq']
    worst['v34_error_share_pct']=last['sq']/total*100
    worst.sort_values(['v34_mean_sq_error'],ascending=False,kind='stable').head(250).to_csv(out/'worst_flights.csv')
    all_slices=[]
    for v,scopes in summary['slices'].items():
        for scope,families in scopes.items():
            for family,rows in families.items():
                all_slices.extend(dict(model=v,scope=scope,family=family,**r) for r in rows)
    pd.DataFrame(all_slices).to_csv(out/'slice_metrics.csv',index=False)
    pd.DataFrame([r for m in models.values() for r in m['fold_slices']]).to_csv(out/'fold_slice_metrics.csv',index=False)
    # Correlation and actual fixed 50:50 averages are exploratory, never selected recipes.
    pairs=[]
    top_latest=set(np.argsort(-last['sq'])[:1000])
    for v,m in models.items():
        e=m['err'];le=last['err']
        pair=dict(model=v,correlation_seed42=float(np.corrcoef(e,le)[0,1]),
                  top1000_overlap=len(top_latest & set(np.argsort(-m['sq'])[:1000])),
                  seed42_rmse=float(np.sqrt(np.mean(e**2))),
                  half_blend_seed42_rmse=float(np.sqrt(np.mean(((e+le)/2)**2))))
        pairs.append(pair)
    summary['shared_errors']=pairs
    pd.DataFrame(pairs).to_csv(out/'shared_errors.csv',index=False)
    # Under/over tails and robust recurrence of input-defined groups.
    f=pd.DataFrame({'airport':d.airport,'date':d.date,'se':last['sq'],'e':last['bias'],
                    'noid':d.noid,'stand':d.STAND_mvt.astype(str),'runway':d.RUNWAY_mvt.astype(str),
                    'airline':d.airline.astype(str)})
    groups=[]
    for family in ['stand','runway','airline']:
        g=f.groupby(['airport',family,'noid'],observed=True).agg(n=('se','size'),days=('date','nunique'),
                sse=('se','sum'),bias=('e','mean'))
        g['rmse']=np.sqrt(g.sse/g.n);g['error_share']=g.sse/total*100
        g=g.sort_values('sse',ascending=False)
        for k,r in g.head(50).iterrows():groups.append(dict(family=family,airport=k[0],key=k[1],noid=bool(k[2]),**r.to_dict()))
    summary['hotspots']=groups
    pd.DataFrame(groups).to_csv(out/'hotspots.csv',index=False)
    summary['worst']=clean(worst.sort_values('v34_mean_sq_error',ascending=False,kind='stable').head(30)
                           .reset_index().assign(mvt_time=lambda x:x.mvt_time.astype(str)).to_dict('records'))
    regimes=summary['slices']['v34']['all']['regime']
    airports=sorted(summary['slices']['v34']['all']['airport'],key=lambda r:-r['sse'])
    noid=d.noid.to_numpy()
    summary['latest']={
        'n':len(d),'noid_n':int(noid.sum()),'noid_pct':float(noid.mean()*100),
        'id_rmse':float(np.sqrt(last['sq'][~noid].mean())),
        'noid_rmse':float(np.sqrt(last['sq'][noid].mean())),
        'worst1000_unique_days':int(d.iloc[np.argsort(-last['sq'])[:1000]].date.nunique()),
        'opposite_sign_extremes':int(((models['v30']['err']*last['err']<0)&(abs(last['err'])>3600)).sum()),
    }
    # Paired, calendar-day bootstrap for descriptive airport bias (not feature selection).
    # Every day stays clustered across airports; uncertainty is conditional on these fits.
    daycodes,days=pd.factorize(d.date,sort=True)
    rng=np.random.default_rng(20260919)
    weights=rng.multinomial(len(days),np.full(len(days),1/len(days)),size=2000)
    intervals=[]
    for apt in sorted(d.airport.unique()):
        mask=(d.airport.eq(apt)&~d.noid).to_numpy()
        sums=np.bincount(daycodes[mask],weights=last['bias'][mask],minlength=len(days))
        counts=np.bincount(daycodes[mask],minlength=len(days))
        draws=(weights@sums)/(weights@counts)
        intervals.append(dict(airport=apt,n=int(mask.sum()),bias=float(last['bias'][mask].mean()),
                              ci95=np.quantile(draws,[.025,.975]).tolist()))
    summary['airport_bias_id_present']=intervals
    return regimes,airports


def rome_calibration(out):
    """Small, projected no-ID read: diagnose the existing 1–4h mixture, fit nothing."""
    frames=[]
    for seed in SEEDS:
        parts=[]
        for fold in range(1,7):
            f=pd.read_parquet(RUNS/REPLAY/f'oof_A{fold}_seed{seed}.parquet',
                columns=['mvt_id','mvt_time','y','gap_sec','p','pred_noid_cap'],
                filters=[('airport','==','LIRF'),('raw_flight_id_missing','==',True)])
            parts.append(f[f.gap_sec.gt(3600)&f.gap_sec.lt(14400)])
        f=pd.concat(parts,ignore_index=True)
        assert f.mvt_id.is_unique
        f['seed']=seed
        f['schedule_label']=(f.y-f.gap_sec).abs().le(60)
        f['residual']=f.pred_noid_cap-f.y
        f['sq']=f.residual**2
        f['p_bin']=pd.cut(f.p,[0,.1,.3,.5,.7,.9,1],include_lowest=True)
        frames.append(f)
    data=pd.concat(frames,ignore_index=True)
    rows=[]
    for (seed,band),f in data.groupby(['seed','p_bin'],observed=True):
        rows.append(dict(seed=int(seed),band=str(band),n=len(f),days=f.mvt_time.dt.normalize().nunique(),
                         predicted_probability=float(f.p.mean()),observed_rate=float(f.schedule_label.mean()),
                         rmse=float(np.sqrt(f.sq.mean())),bias=float(f.residual.mean())))
    pd.DataFrame(rows).to_csv(out/'rome_probability_bins.csv',index=False)
    first=frames[0]
    mid=first[first.p.gt(.3)&first.p.le(.7)]
    # Day-clustered uncertainty for a broad band chosen during residual inspection.
    daily=mid.assign(date=mid.mvt_time.dt.normalize(),gap=mid.p-mid.schedule_label.astype(float)).groupby('date').gap.agg(['sum','size'])
    rng=np.random.default_rng(20260919)
    weights=rng.multinomial(len(daily),np.full(len(daily),1/len(daily)),size=2000)
    draws=(weights@daily['sum'].to_numpy())/(weights@daily['size'].to_numpy())
    result=dict(n=len(first),days=int(first.mvt_time.dt.normalize().nunique()),bins=rows,
                mid_n=len(mid),mid_predicted=float(mid.p.mean()),mid_observed=float(mid.schedule_label.mean()),
                mid_gap_ci95=np.quantile(draws,[.025,.975]).tolist(),
                seed_mid_gaps=[float((f.loc[f.p.gt(.3)&f.p.le(.7),'p']-f.loc[f.p.gt(.3)&f.p.le(.7),'schedule_label']).mean()) for f in frames],
                note='Post-hoc exploratory calibration diagnostic on reused next-month folds; no calibrator fitted. Middle band chosen during residual inspection. Counts per seed are distinct flights, never pooled as independent observations.')
    style()
    fig,axes=plt.subplots(1,3,figsize=(16,5.5),layout='constrained')
    for seed,color in zip(SEEDS,['#067d7a','#cb9549','#6561b4']):
        subset=[r for r in rows if r['seed']==seed]
        axes[0].plot([r['predicted_probability'] for r in subset],[r['observed_rate'] for r in subset],'.-',label=f'Seed {seed}',color=color)
        axes[1].plot([r['predicted_probability'] for r in subset],[r['bias'] for r in subset],'.-',label=f'Seed {seed}',color=color)
    axes[0].plot([0,1],[0,1],':',color='gray',label='Calibrated');axes[0].legend(fontsize=9)
    axes[0].set(xlabel='Mean predicted P(schedule-recorded)',ylabel='Observed fraction: |label − gap| ≤ 60 s',title='Middle probabilities overstate substitution')
    for r in rows[:6]:axes[0].annotate(f"n={r['n']}",(r['predicted_probability'],r['observed_rate']),xytext=(3,-14),textcoords='offset points',fontsize=8)
    axes[1].axhline(0,color='gray',lw=.8)
    axes[1].set(xlabel='Mean predicted P(schedule-recorded)',ylabel='Mean prediction − actual (s)',title='The same bands tend to predict too high')
    for match,color,label in [(False,'#c57548','Not a schedule-gap label'),(True,'#067d7a','Schedule-gap label')]:
        f=first[first.schedule_label.eq(match)]
        axes[2].scatter(f.p,f.residual/60,s=14,alpha=.5,color=color,label=label)
    axes[2].axhline(0,color='gray',lw=.8);axes[2].legend(fontsize=8)
    axes[2].set(xlabel='Predicted P(schedule-recorded)',ylabel='Prediction − actual (minutes)',title='Both missed substitutions and false alarms')
    for ax in axes:ax.grid(alpha=.4)
    title='A distinct next experiment: calibrate the existing Rome 1–4h mixture'
    note=f'{len(first):,} unique no-ID flights; A1–A6. Three seeds shown separately. Target-defined colours diagnose errors, not prediction routes.'
    fig.suptitle(title+'\n'+note,x=.025,ha='left',fontsize=15,linespacing=1.6)
    for ext in ['png','pdf']:fig.savefig(out/f'09_rome_calibration.{ext}',dpi=155,bbox_inches='tight')
    plt.close(fig)
    write_json(out/'rome_calibration.json',result)
    return result,('09_rome_calibration',title,note)


def report(summary, pages, out):
    calibration,extra_page=rome_calibration(out)
    summary['rome_calibration']=calibration
    pages=[p for p in pages if p[0]!='09_rome_calibration']+[extra_page]
    write_json(out/'figure_index.json',pages)
    write_json(out/'summary.json',summary)
    s=summary['models']['v34'];n=summary['latest']['n'];latest=summary['latest']
    lines=['# Residuals of officially scored recipes', '',
           f"Generated {summary['generated_at']}. Residual = prediction − actual; positive means overprediction.", '',
           'The ranking labels are hidden. These plots diagnose held-out 2025 predictions of scored recipes; '
           'they do not reveal which 2026 flights caused a leaderboard error.', '',
           f"Main charts use **{n:,} distinct July–December flights**, A1–A6, three seeds. "
           'Squared errors are averaged across seeds, not predictions. Signed distributions and scatter use seed 42. '
           'All 16 folds are shown separately, and R_B_long preserves the original averaging over origins. '
           'No rows, folds or extreme labels are removed. Target-conditioned residual patterns include '
           'regression-to-the-mean effects; they are not by themselves evidence for a deployable correction.', '',
           '**Coverage:** original OOF for v12–v16; v19 reconstructed from saved T1 model predictions with '
           'the registered four-hour rule; v30/v33/v34 use the September 19 replay. '
           'These last four are marked †. Their official scores belong to the original submitted files. '
           f"The v34 replay differs by +0.00697 s combined from the original. All {len(summary['official'])} scored submissions "
           'appear in the inventory; unavailable predictions are not invented.', '',
           f"Official-score source: {summary['leaderboard_source']}.", '',
           '[Open the interactive report](scored_residuals/index.html) · '
           '[Download the eight-page PDF](scored_residuals/residual_atlas.pdf)', '',
           '## What remains in v34', '',
           f"- Next-month RMSE is **{s['scores']['R_A']:.2f} s**; longer-gap is **{s['scores']['R_B_long']:.2f} s**. "
           f"The seed-42 median absolute error is {s['abs_quantiles42']['p50']:.1f} s, "
           f"99th percentile {s['abs_quantiles42']['p99']:.1f} s, and maximum {s['abs_quantiles42']['max']:,.0f} s.",
           f"- The worst **100 flights contribute {s['top_shares']['100']:.1f}%** of squared error; "
           f"the worst 1,000 contribute **{s['top_shares']['1000']:.1f}%**. These are distinct flights, not repeated folds.",
           f"- Missing-ID flights are **{latest['noid_pct']:.3f}%** of flights but **{s['noid_error_share']:.1f}%** "
           f"of squared error (RMSE {latest['noid_rmse']:.0f} s). ID-present flights still carry "
           f"**{100-s['noid_error_share']:.1f}%** of squared error (RMSE {latest['id_rmse']:.1f} s). "
           'Both the extreme records and the broad operational population matter.', '',
           '| Observable regime | Flights | Days | RMSE (s) | Error share | Bias (s) |',
           '|---|---:|---:|---:|---:|---:|']
    for r in summary['slices']['v34']['all']['regime']:
        lines.append(f"| {r['key']} | {int(r['n']):,} | {int(r['days'])} | {r['rmse']:.1f} | {r['error_share']:.2f}% | {r['bias']:+.1f} |")
    c=calibration
    lines += ['', '## A concrete next experiment: the Rome 1–4h mixture', '',
              f"Among **{c['n']} distinct no-ID Rome flights** at 1–4-hour schedule gaps, "
              f"the {c['mid_n']} flights assigned probabilities between 0.3 and 0.7 average "
              f"**{c['mid_predicted']:.1%} predicted schedule substitution versus {c['mid_observed']:.1%} observed**. "
              f"The excess is positive in all three seeds. A descriptive day-bootstrap interval for the seed-42 "
              f"excess is [{c['mid_gap_ci95'][0]*100:.1f}, {c['mid_gap_ci95'][1]*100:.1f}] percentage points. "
              'The broad middle band was chosen during inspection, so this is exploratory evidence.', '',
              'The classifier distinguishes the extremes reasonably, but its middle probabilities often '
              'put too much weight on the schedule gap. Test a **shrunk probability calibration layer**, '
              'trained using inner chronological out-of-fold predictions within each training window. '
              'Judge it by taxi-time squared error as well as probability calibration: the cost of a wrong '
              'probability increases with the distance between the schedule-gap and ordinary predictions. '
              'Keep a separate ordinary/late estimate; do not convert uncertain probabilities into a new hard rule. '
              'No such calibrator was fitted here. This targets the 1–4h mixture, separately from the already '
              'completed day-shift corrections and 4–12h rule experiments.', '',
              '[Probability-calibration figure (PDF)](scored_residuals/09_rome_calibration.pdf) · '
              '[Per-seed bin counts and errors](scored_residuals/rome_probability_bins.csv)', '',
              '## Other modelling implications', '',
              '1. **Treat Rome as a competing-clock problem.** The gap/label scatter separates ordinary taxi, '
              'schedule-gap labels, and roughly one-day labels. The hard rule can badly overpredict an ordinary '
              'flight and underpredict a day-shift record. A route must use observable clock agreement, '
              'record provenance and arrival-handler evidence; the true label cannot define prediction routes. '
              'The September 19 artifact study has already tested day blending and twins. Reuse those results; '
              'these plots are not a new validation of the same corrections.',
              '2. **Inspect broad within-airport operational errors too.** The hour, queue and weather plots '
              'condition on an existing flight ID and compare within airports. Persistent underprediction '
              'at high queue/clock gaps suggests testing airport-specific queue interactions or a separate '
              'congestion component. A feature trend is association, not proof that adding it improves the model; '
              'weather and queue features are already in v34.',
              '3. **Use recurring bias as a small, separately validated correction candidate.** The table below '
              'clusters uncertainty by day. Any airport/hour or stand correction needs shrinkage and must be fitted '
              'inside each training window; fitting offsets to these residuals and reporting the same flights '
              'as validation would leak outcomes.',
              '4. **Check complementarity before an ensemble.** The shared-error table compares the same flights '
              'and a fixed 50:50 blend on seed 42. It is descriptive, with no tuned weight or new candidate '
              'selection. Repeated catastrophic errors will not disappear by averaging highly correlated models.', '',
              'The plots are exploratory on reused development folds. There is no January residual coverage here; '
              'airport, weather and season patterns may transfer differently to the January/July 2026 ranking set. '
              'Tiny groups are hypotheses, not established general improvements.', '',
              '### Airport bias with flight ID present', '',
              'Three-seed mean residual, 2,000 calendar-day bootstrap draws; descriptive, unadjusted intervals. '
              'Intervals condition on the fitted models and do not establish future transfer.', '',
              '| Airport | Flights | Bias (s) | 95% day-bootstrap interval |', '|---|---:|---:|---|']
    for r in summary['airport_bias_id_present']:
        lines.append(f"| {r['airport']} | {r['n']:,} | {r['bias']:+.1f} | [{r['ci95'][0]:+.1f}, {r['ci95'][1]:+.1f}] |")
    lines+=['','## Inventory and reproduction','','| Version | Official LB | Residual coverage | R_A | R_B_long |','|---|---:|---|---:|---:|']
    for r in summary['official']:
        m=summary['models'].get(r['version'])
        score=lambda k: f"{m['scores'][k]:.3f}" if m else '—'
        lines.append(f"| {r['version']} | {r['lb']:.4f} | {r['coverage']} | {score('R_A')} | {score('R_B_long')} |")
    lines+=['','```bash','.venv/bin/python VALIDATION/tools/plot_scored_residuals.py --leaderboard tmp/residual_audit/leaderboard.json','```','',
            'Outputs include file hashes, exact score-reproduction checks, all fold/slice metrics, '
            'the 250 worst distinct flights and self-contained HTML. No training or upload was performed.','']
    (out.parent/'SCORED_RESIDUALS.md').write_text('\n'.join(lines))
    # The HTML embeds aggregates and figures; opening it requires no server/CDN.
    intro=''.join('<p>'+re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html.escape(x))+'</p>' for x in lines[2:9:2])
    images=''.join(f'<section><h2>{html.escape(title)}</h2><p>{html.escape(note)}</p>'
                   f'<img loading="lazy" alt="{html.escape(title)}" src="data:image/png;base64,'
                   +base64.b64encode((out/(name+'.png')).read_bytes()).decode()+'"></section>'
                   for name,title,note in pages)
    template=(Path(__file__).with_name('scored_residuals_template.html')).read_text()
    payload=json.dumps(clean(summary),allow_nan=False,separators=(',',':')).replace('</','<\\/')
    (out/'index.html').write_text(template.replace('__INTRO__',intro).replace('__FIGURES__',images).replace('__DATA__',payload))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--leaderboard',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'VALIDATION/reports/scored_residuals')
    p.add_argument('--report-only',action='store_true',help='Refresh HTML/Markdown from existing aggregates and figure sheets')
    p.add_argument('--from-cache',action='store_true',help='Reuse checked local residual arrays; verify saved source hashes first')
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    official=ledger()
    source='SUBMISSIONS.md snapshot 2026-09-19 10:54 UTC (historical, not a live refresh)'
    if args.leaderboard and args.leaderboard.exists() and args.leaderboard.stat().st_size:
        lb=json.loads(args.leaderboard.read_text())
        source=f"{lb.get('snapshotSource', 'status script')}, {lb['updatedAt']}"
        write_json(out/'leaderboard_snapshot.json',lb)
        official=with_live_inventory(official,lb)
    if args.report_only:
        summary=json.loads((out/'summary.json').read_text())
        summary['official']=official
        summary['leaderboard_source']=source
        pages=json.loads((out/'figure_index.json').read_text())
        write_json(out/'summary.json',summary)
        report(summary,pages,out)
        return
    print('Loading one context row per next-month flight',flush=True)
    d=context();models={};provenance=[]
    cache=ROOT/'tmp/residual_audit/arrays'
    cache.mkdir(parents=True,exist_ok=True)
    for v in MODELS:
        print('Reading and checking',v,flush=True)
        if args.from_cache:
            saved=json.loads((cache/(v+'.json')).read_text())
            for item in saved['provenance']:
                assert digest(ROOT/item['file'])==item['sha256'], item['file']
            provenance.extend(saved['provenance'])
            with np.load(cache/(v+'.npz')) as arrays:
                assert np.array_equal(arrays['mvt_ids'],d.index.to_numpy())
                models[v]=dict(saved['model'],**{k:arrays[k] for k in arrays.files if k!='mvt_ids'})
        else:
            start=len(provenance)
            models[v]=read_model(v,d,provenance)
            m=models[v]
            np.savez(cache/(v+'.npz'),mvt_ids=d.index.to_numpy(),
                     **{k:value for k,value in m.items() if isinstance(value,np.ndarray)})
            write_json(cache/(v+'.json'),dict(model={k:value for k,value in m.items() if not isinstance(value,np.ndarray)},
                                             provenance=provenance[start:]))
        print(v,models[v]['scores'],flush=True);gc.collect()
    summary=dict(generated_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                 leaderboard_source=source,official=official,models={},slices={},
                 methods=dict(residual='prediction minus actual',main_folds='A1-A6',seeds=SEEDS,
                              loss='mean of squared errors, not error of mean prediction',signed='seed 42',
                              all_folds=MAIN_NAMES))
    for v,m in models.items():
        print('Aggregating',v,flush=True)
        summary['models'][v]=dict(name=m['name'],kind=m['kind'],run=m['run'],column=m['column'],**metrics(d,m))
        summary['slices'][v]={scope:{family:aggregate(d,m,family,scope) for family in FAMILIES}
                              for scope in ['all',*sorted(d.airport.unique())]}
        lm=m['long_mask']
        # Longer-gap attribution available separately, with duplicate origins averaged.
        long_model=dict(sq=m['long_sq'],bias=np.zeros(lm.sum()),err=np.zeros(lm.sum()))
        summary['models'][v]['long_regimes']=[{k:value for k,value in r.items() if k not in ['bias','mae42','mean_prediction42']}
                                            for r in aggregate(d.loc[lm],long_model,'regime')]
    write_json(out/'summary.json',summary)
    tables_and_findings(d,models,summary,out)
    # The reconstructed clock recipe is a proxy, never claimed byte-identical.
    summary['models']['v19']['combined_difference_from_recorded']=models['v19']['scores']['R_combined']-273.1284
    write_json(out/'summary.json',summary)
    write_json(out/'provenance.json',dict(files=provenance,script_sha256=digest(Path(__file__)),
               data_context_run=REPLAY,checks='Every fold/seed: finite predictions, unique IDs, exact target/time/airport match; original scores reproduce within 1e-8'))
    print('Rendering eight figure sheets and interactive report',flush=True)
    pages=figures(d,models,summary,out)
    write_json(out/'figure_index.json',pages)
    report(summary,pages,out)
    print(json.dumps(dict(latest=summary['latest'],metrics=summary['models']['v34']),default=str)[:2500],flush=True)
    print('Saved',out/'index.html',flush=True)


if __name__=='__main__':main()
