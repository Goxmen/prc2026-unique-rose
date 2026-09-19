"""Explain recording-artifact errors from completed replay and candidate OOF runs.

Usage, after both runs complete:
  .venv/bin/python VALIDATION/tools/summarize_artifact_errors.py \
      --parent VALIDATION/runs/v3_20260919_artifact_control --run <candidate run>

Uses reporting.load_losses and score_groups for the official protocol weighting.
Candidate files are read sequentially, never during training. No raw-data scan,
model fit, leaderboard access or submission is performed. Ranking targets remain
hidden; ranking information is limited to the existing raw audit's prevalences.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline')]
from protocol import MAIN_NAMES
from records import read_run
import record_fixes
import reporting


GROUP_FOLDS = {'R_A': [f'A{i}' for i in range(1, 7)], 'R_B_long': ['B3', 'B4', 'B7']}
REGIMES = ['Rome next-day hard rule', 'Rome other hard rule', 'Rome no-ID 1–4h',
           'Rome other no-ID', 'Other airport no-ID', 'ID twin', 'Other ID']
LABELS = {'R_A': 'Next month', 'R_B_long': 'Longer gap'}


def regime_labels(frame, twin):
    """Disjoint observable regimes; never use the target to classify a row."""
    noid = frame.FLIGHT_ID_mvt.isna().to_numpy()
    rome = frame.ADEP_mvt.eq('LIRF').to_numpy()
    move = pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True)
    schedule = pd.to_datetime(frame.SCHED_TIME_UTC_mvt, utc=True)
    gap = (move - schedule).dt.total_seconds().to_numpy(float)
    local = lambda s: s.dt.tz_convert('Europe/Rome').dt.tz_localize(None).dt.normalize()
    days = (local(move) - local(schedule)).dt.days.to_numpy(float)
    return pd.Categorical(np.select([
        rome & noid & (gap >= 14 * 3600) & (days == 1),
        rome & noid & (gap >= 4 * 3600),
        rome & noid & (gap >= 3600) & (gap < 4 * 3600),
        rome & noid, ~rome & noid, ~noid & np.asarray(twin, bool),
    ], REGIMES[:-1], default=REGIMES[-1]), categories=REGIMES)


def load_annotations(record, directory):
    """One seed and one A fold per unique month; no repeated-origin metadata."""
    wanted = ['mvt_id', 'ADEP_mvt', 'ADES_mvt', 'FLIGHT_mvt', 'FLIGHT_ID_mvt',
              'STAND_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'LOBT_flt',
              'AOBT_3_flt', 'mask_twins']
    parts = []
    for fold in GROUP_FOLDS['R_A']:
        seed = str(record['config']['seeds'][0])
        path = directory / record['folds'][fold]['predictions'][seed]
        available = set(pq.ParquetFile(path).schema.names)
        frame = pd.read_parquet(path, columns=[c for c in wanted if c in available])
        # Exactly the candidate runner's context: the current validation month.
        twin = frame.mask_twins.to_numpy(bool) if 'mask_twins' in frame else record_fixes.twin_rows(frame)
        out = frame.loc[:, [c for c in ('mvt_id', 'FLIGHT_mvt', 'ADES_mvt', 'STAND_mvt') if c in frame]].copy()
        out['regime'] = regime_labels(frame, twin)
        movement = pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True)
        for name, col in [('schedule_gap_sec', 'SCHED_TIME_UTC_mvt'),
                          ('lobt_gap_sec', 'LOBT_flt'), ('aobt3_gap_sec', 'AOBT_3_flt')]:
            out[name] = (movement - pd.to_datetime(frame[col], utc=True)).dt.total_seconds()
        parts.append(out)
    annotations = pd.concat(parts, ignore_index=True).set_index('mvt_id').sort_index()
    if annotations.index.has_duplicates or annotations.index.isna().any():
        raise ValueError('A folds must provide exactly one metadata row per unique movement')
    return annotations


def main_groups(losses):
    groups = reporting.score_groups(losses)
    if any(name not in groups for name in GROUP_FOLDS):
        raise ValueError('Both complete R_A and R_B_long groups are required')
    return {name: groups[name] for name in GROUP_FOLDS}


def headline(groups, weight):
    values = {name: reporting.rmse(loss) for name, loss in groups.items()}
    values['R_combined'] = float(np.sqrt(weight * values['R_A']**2 + (1-weight) * values['R_B_long']**2))
    return values


def concentration(sq):
    ordered = np.sort(np.asarray(sq, float))[::-1]
    total = float(ordered.sum())
    return {str(k): 100 * float(ordered[:k].sum()) / total if total else 0.0 for k in (20,100,1000)}


def simple_stats(sq, selected):
    values = sq[np.asarray(selected, bool)]
    return dict(n=len(values), rmse=float(np.sqrt(values.mean())) if len(values) else None,
                sse=float(values.sum()))


def diagnostic_summary(loss, annotations):
    sq = loss.sq.mean(axis=0)
    meta = loss.meta
    category = annotations.regime.reindex(meta.index)
    if category.isna().any():
        raise ValueError('Every scored movement needs a recording regime')
    total = float(sq.sum())
    slices = {}
    for regime in REGIMES:
        mask = category.eq(regime).to_numpy()
        row = simple_stats(sq, mask)
        row['sse_pct'] = 100 * row['sse'] / total if total else 0.0
        row['ordinary'] = simple_stats(sq, mask & meta.y.le(7200).to_numpy())
        row['tail'] = simple_stats(sq, mask & meta.y.gt(7200).to_numpy())
        slices[regime] = row
    return dict(n=len(meta), rmse=reporting.rmse(loss), total_sse=total,
                by_seed=dict(zip(map(str,loss.seeds),map(float,np.sqrt(loss.sq.mean(axis=1))))),
                ordinary=simple_stats(sq,meta.y.le(7200)), tail=simple_stats(sq,meta.y.gt(7200)),
                concentration_pct=concentration(sq), regimes=slices)


def details(loss, annotations, indices, baseline=None):
    """Concrete examples; rms_error averages squared errors, not predictions."""
    sq = loss.sq.mean(axis=0)
    base_sq = baseline.sq.mean(axis=0) if baseline is not None else None
    rows = []
    for index in indices:
        movement_id = loss.meta.index[index]
        meta, annotation = loss.meta.iloc[index], annotations.loc[movement_id]
        present = lambda value: str(value) if pd.notna(value) else None
        row = dict(mvt_id=int(movement_id), airport=meta.airport,
                   time_utc=meta.mvt_time.isoformat(), flight=present(annotation.get('FLIGHT_mvt')),
                   destination=present(annotation.get('ADES_mvt')), stand=present(annotation.get('STAND_mvt')),
                   regime=str(annotation.regime), target_sec=float(meta.y),
                   rms_error_sec=float(np.sqrt(sq[index])), mean_squared_error=float(sq[index]))
        for key in ['schedule_gap_sec','lobt_gap_sec','aobt3_gap_sec']:
            value = annotation[key]
            row[key] = float(value) if pd.notna(value) else None
        if base_sq is not None:
            row.update(control_rms_error_sec=float(np.sqrt(base_sq[index])),
                       delta_mean_squared_error=float(sq[index]-base_sq[index]))
        rows.append(row)
    return rows


def stable_indices(values, movement_ids, count, descending=True):
    return np.lexsort((np.asarray(movement_ids), -values if descending else values))[:count]


def changed_examples(candidate, parent, annotations):
    reporting.same_flights(candidate.meta, parent.meta)
    delta = candidate.sq.mean(axis=0) - parent.sq.mean(axis=0)
    changed = np.flatnonzero(np.any(candidate.sq != parent.sq, axis=0))
    good = changed[delta[changed] < 0]
    bad = changed[delta[changed] > 0]
    good = good[stable_indices(delta[good], candidate.meta.index[good], 10, False)]
    bad = bad[stable_indices(delta[bad], candidate.meta.index[bad], 10, True)]
    return dict(unique_loss_changed=len(changed), helped=int((delta<0).sum()), harmed=int((delta>0).sum()),
                delta_sse=float(delta.sum()), fixes=details(candidate,annotations,good,parent),
                harms=details(candidate,annotations,bad,parent))


def influence(parent_groups, candidate_groups, weight):
    """Leave one whole UTC date out only to measure influence, never new scores."""
    daily, full = {}, {}
    combined_contribution = None
    for name in GROUP_FOLDS:
        left, right = parent_groups[name], candidate_groups[name]
        reporting.same_flights(left.meta,right.meta)
        frame = pd.DataFrame(dict(day=left.meta.mvt_time.dt.strftime('%Y-%m-%d').to_numpy(),
             control_sse=left.sq.mean(axis=0),candidate_sse=right.sq.mean(axis=0),n=1))
        day = frame.groupby('day',sort=True).sum()
        day['delta_sse'] = day.candidate_sse-day.control_sse
        daily[name] = day
        full[name] = dict(n=len(frame),control_sse=float(day.control_sse.sum()),
                          candidate_sse=float(day.candidate_sse.sum()))
        share = (weight if name=='R_A' else 1-weight)*day.delta_sse/len(frame)
        combined_contribution = share if combined_contribution is None else combined_contribution.add(share,fill_value=0)
    output = {}
    for score in [*GROUP_FOLDS,'R_combined']:
        impact = combined_contribution if score=='R_combined' else daily[score].delta_sse
        if not np.any(impact.to_numpy()!=0):
            output[score]=dict(day=None,removed_rows_by_group={},full_delta_rmse=0.0,
                without_day_delta_rmse=0.0,interpretation='No UTC day has changed aggregate loss; all rows retained.')
            continue
        selected = sorted(impact.index,key=lambda d:(-abs(float(impact[d])),d))[0]
        included = list(GROUP_FOLDS) if score=='R_combined' else [score]
        before, after, counts = {}, {}, {}
        for name in included:
            stats = full[name]
            removed = daily[name].loc[selected] if selected in daily[name].index else pd.Series(dict(n=0,control_sse=0,candidate_sse=0))
            remaining = stats['n']-int(removed.n)
            if remaining <= 0:
                raise ValueError('Influence diagnostic requires more than one UTC date')
            before[name] = [stats[k]/stats['n'] for k in ('control_sse','candidate_sse')]
            after[name] = [(stats[k]-float(removed[k]))/remaining for k in ('control_sse','candidate_sse')]
            counts[name] = int(removed.n)
        def delta(values):
            mse = (weight*np.asarray(values['R_A'])+(1-weight)*np.asarray(values['R_B_long'])
                   if score=='R_combined' else np.asarray(values[score]))
            return float(np.sqrt(mse[1])-np.sqrt(mse[0]))
        output[score] = dict(day=selected,removed_rows_by_group=counts,
                            full_delta_rmse=delta(before),without_day_delta_rmse=delta(after),
                            interpretation='Influence diagnostic only; official scores retain every row.')
    return output


def selected_prediction_means(record,directory,columns,ids):
    """Projected filtered reads only for reported examples, all seeds/origins."""
    if not ids:
        return {}
    frames = []
    for fold in dict.fromkeys(sum(GROUP_FOLDS.values(),[])):
        for seed,path in record['folds'][fold]['predictions'].items():
            frame = pd.read_parquet(directory/path,columns=['mvt_id',*columns],
                                    filters=[('mvt_id','in',sorted(ids))])
            frame['fold'],frame['seed'] = fold,int(seed)
            frames.append(frame)
    selected = pd.concat(frames,ignore_index=True)
    result = {}
    for group,folds in GROUP_FOLDS.items():
        per_seed = selected.loc[selected.fold.isin(folds)].groupby(['mvt_id','seed'])[columns].mean()
        result[group] = per_seed.groupby('mvt_id').mean()
    return result


def number(value,digits=2):
    return '—' if value is None else f'{value:.{digits}f}'


def table(rows,columns):
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join('---' for _ in columns)+' |',
                      *['| '+' | '.join(str(row.get(c,'—')) for c in columns)+' |' for row in rows]])


def plots(result,output):
    fig,axes=plt.subplots(1,2,figsize=(11.6,4.6),layout='constrained')
    colors=plt.cm.tab10(np.arange(len(REGIMES)))
    bottoms=np.zeros(2)
    for regime,color in zip(REGIMES,colors):
        values=[result['parent']['groups'][g]['regimes'][regime]['sse_pct'] for g in GROUP_FOLDS]
        axes[0].bar(list(LABELS.values()),values,bottom=bottoms,label=regime,color=color)
        bottoms+=values
    axes[0].set(ylabel='Share of squared error (%)',title='Where the control loses',ylim=(0,100))
    axes[0].legend(fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.11),ncol=2,frameon=False)
    x=np.arange(3);width=.36
    for i,(group,label) in enumerate(LABELS.items()):
        values=[result['parent']['groups'][group]['concentration_pct'][str(k)] for k in (20,100,1000)]
        bars=axes[1].bar(x+(i-.5)*width,values,width,label=label,color=('#285f93','#bf673b')[i])
        axes[1].bar_label(bars,fmt='%.1f%%',fontsize=8)
    axes[1].set(xticks=x,xticklabels=['Worst 20','Worst 100','Worst 1,000'],ylabel='Share of squared error (%)',
                title='Error is concentrated',ylim=(0,105))
    axes[1].legend(frameon=False,fontsize=9)
    fig.suptitle('Recording errors · all validation rows retained',fontsize=15)
    for extension in ('png','pdf'):
        fig.savefig(output/f'error_concentration.{extension}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    if result['candidates']:
        fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
        names=list(result['candidates']);y=np.arange(len(names))
        for i,(group,label) in enumerate(LABELS.items()):
            values=[result['candidates'][name]['delta_scores'][group] for name in names]
            ax.barh(y+(i-.5)*.34,values,height=.34,label=label,color=('#285f93','#bf673b')[i])
            for j,name in enumerate(names):
                item=result['candidates'][name].get('comparison',{}).get(group)
                if item:
                    ax.plot(item['ci95'],[y[j]+(i-.5)*.34]*2,color='#303030',lw=1)
        ax.axvline(0,color='#555',lw=.8)
        ax.set(yticks=y,yticklabels=names,xlabel='Candidate − control RMSE (seconds; negative is better)',
               title='Recording corrections on the same folds and seeds\nLines: 95% paired UTC-day bootstrap intervals')
        ax.invert_yaxis();ax.legend(frameon=False)
        for extension in ('png','pdf'):
            fig.savefig(output/f'candidate_changes.{extension}',dpi=180,bbox_inches='tight')
        plt.close(fig)


def markdown(result):
    p=result['parent'];lines=['# Where the recording-artifact model fails','',
      'Ranking targets are hidden. The errors below are measured on 2025 validation flights; ranking counts describe observable patterns only.','',
      'Each flight has equal weight within a score group. Squared errors are averaged across seeds and repeated training origins before scoring. R_A uses July–December; R_B_long uses November–December, averaging its two December origins. All sixteen folds and every target remain in the report.','',
      '![Error concentration](error_concentration.png)','']
    rows=[]
    for name,entry in [('control',p),*result['candidates'].items()]:
        row=dict(Candidate=name,**{g:number(entry['scores'][g]) for g in ('R_A','R_B_long','R_combined')})
        comparison=entry.get('comparison',{}).get('R_combined')
        row['Combined Δ [95% CI]']=(f"{comparison['difference']:+.2f} [{comparison['ci95'][0]:+.2f}, {comparison['ci95'][1]:+.2f}]"
                                   if comparison else '—')
        rows.append(row)
    lines += [table(rows,['Candidate','R_A','R_B_long','R_combined','Combined Δ [95% CI]']),'',
              'These are reused development folds, not untouched confirmation. Combined CV is a provisional summary, not a leaderboard forecast.','']
    if result['candidates']:
        lines += ['![Candidate changes](candidate_changes.png)','']
    rows=[]
    for group in GROUP_FOLDS:
        s=p['groups'][group]
        rows.append(dict(Group=group,Rows=s['n'],**{'≤2h RMSE':number(s['ordinary']['rmse']),'>2h rows':s['tail']['n'],
            'Tail error %':number(100*s['tail']['sse']/s['total_sse'],1),
            'Worst 20 %':number(s['concentration_pct']['20'],1),'Worst 100 %':number(s['concentration_pct']['100'],1),
            'Worst 1000 %':number(s['concentration_pct']['1000'],1)}))
    lines += [table(rows,['Group','Rows','≤2h RMSE','>2h rows','Tail error %','Worst 20 %','Worst 100 %','Worst 1000 %']),'']
    rows=[]
    for regime in REGIMES:
        a,b=(p['groups'][g]['regimes'][regime] for g in GROUP_FOLDS)
        rows.append(dict(Regime=regime,**{'A rows':a['n'],'A error %':number(a['sse_pct'],1),
            'B rows':b['n'],'B error %':number(b['sse_pct'],1)}))
    lines += [table(rows,['Regime','A rows','A error %','B rows','B error %']),'',
      'Regimes are disjoint and use observable clocks and the raw flight-ID status. The next-day regime means exactly one Europe/Rome date crossed and schedule gap ≥14 h; it does not assume the hidden label is a day shift.','']
    for group in GROUP_FOLDS:
        lines += [f'**Largest 20 unique control errors: {group}.** Mean prediction is descriptive; RMS error averages seed/origin squared errors.','']
        rows=[dict(ID=r['mvt_id'],Flight=r['flight'],Airport=r['airport'],Date=r['time_utc'][:10],
                   Target=number(r['target_sec'],0),Prediction=number(r.get('mean_prediction_sec'),0),
                   **{'RMS error':number(r['rms_error_sec'],0),'Schedule gap':number(r['schedule_gap_sec'],0),
                      'Regime':r['regime']}) for r in p['top_errors'][group]]
        lines += [table(rows,['ID','Flight','Airport','Date','Target','Prediction','RMS error','Schedule gap','Regime']),'']
    if result['candidates']:
        lines += ['**Largest fixes and harms.** Counts refer to unique flights whose seed/origin-averaged loss improves or worsens. Examples show control RMS error → candidate RMS error in seconds.','']
        rows=[]
        def example(items):
            if not items:return '—'
            row=items[0]
            return f"{row['flight'] or row['mvt_id']} ({row['airport']}): {row['control_rms_error_sec']:.0f} → {row['rms_error_sec']:.0f}"
        for name,item in result['candidates'].items():
            for group,changes in item['changes'].items():
                rows.append(dict(Candidate=name,Group=group,Helped=changes['helped'],Harmed=changes['harmed'],
                    **{'Largest fix':example(changes['fixes']),'Largest harm':example(changes['harms'])}))
        lines += [table(rows,['Candidate','Group','Helped','Harmed','Largest fix','Largest harm']),'']
        lines += ['**Influence of the most affected UTC day.** This removes that date only for an influence calculation. It does not change validation scores or select easier folds.','']
        rows=[]
        for name,item in result['candidates'].items():
            for group in ('R_A','R_B_long','R_combined'):
                d=item['influence'][group]
                rows.append(dict(Candidate=name,Group=group,Day=d['day'] or '—',**{'Full Δ':number(d['full_delta_rmse']),
                    'Without-day Δ':number(d['without_day_delta_rmse'])}))
        lines += [table(rows,['Candidate','Group','Day','Full Δ','Without-day Δ']),'']
        lines += ['Largest fixes and harms for each candidate, all seed scores, all sixteen folds and regime/target slices are retained in `error_diagnostics.json`.','']
    rows=[]
    combined=result['candidates'].get('combined')
    for fold in MAIN_NAMES:
        r=dict(Fold=fold,Rows=p['folds'][fold]['n'],Control=number(p['folds'][fold]['rmse']))
        if combined:r['Combined candidate']=number(combined['folds'][fold]['rmse'])
        rows.append(r)
    lines += [table(rows,['Fold','Rows','Control']+(['Combined candidate'] if combined else [])),'',
      'The raw training/ranking prevalence audit is linked in [RAW_PATTERN_FINDINGS.md](RAW_PATTERN_FINDINGS.md). No submission was created or uploaded by this report.','']
    return '\n'.join(lines)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--parent',required=True)
    ap.add_argument('--run',help='Completed derived candidate run; omit for control diagnostics only')
    ap.add_argument('--parent-column',default='pred_noid_cap')
    ap.add_argument('--candidates',help='Comma-separated candidate names, default all registered candidates')
    ap.add_argument('--output',type=Path,default=ROOT/'VALIDATION/reports/recording_artifacts_20260919')
    args=ap.parse_args(argv)
    parent,pdir=read_run(args.parent)
    run,rdir=read_run(args.run) if args.run else (None,None)
    for record in [parent]+([run] if run else []):
        if record['status']!='complete' or not set(MAIN_NAMES).issubset(record['folds']) or record['config']['sample']!=1:
            raise ValueError('Only complete, full-sample sixteen-fold runs are supported')
    parent=dict(parent,prediction_column=args.parent_column)
    candidates={k:v for k,v in run['config']['candidates'].items() if k!='control'} if run else {}
    if args.candidates:
        candidates={name:candidates[name] for name in args.candidates.split(',')}
    args.output.mkdir(parents=True,exist_ok=True)
    print('Loading unique regime metadata, then control loss files',flush=True)
    annotations=load_annotations(run or parent,rdir or pdir)
    baseline=reporting.load_losses(parent,pdir)
    base_groups=main_groups(baseline)
    weight=parent['ranking_weights']['jan_fraction']
    def entry(losses,groups):
        return dict(scores=headline(groups,weight),
                    folds={name:dict(n=len(loss.meta),rmse=reporting.rmse(loss),
                        by_seed=dict(zip(map(str,loss.seeds),map(float,np.sqrt(loss.sq.mean(axis=1))))))
                        for name,loss in losses.items()},
                    groups={name:diagnostic_summary(loss,annotations) for name,loss in groups.items()},
                    top_errors={name:details(loss,annotations,stable_indices(loss.sq.mean(axis=0),loss.meta.index,20))
                        for name,loss in groups.items()})
    result=dict(generated_utc=datetime.now(timezone.utc).isoformat(),parent_run=parent['run_id'],
        candidate_run=run['run_id'] if run else None,ranking_targets='hidden',
        weighting='reporting.load_losses + reporting.score_groups; mean squared error over seeds and repeated origins per unique movement',
        ranking_jan_weight=weight,parent=entry(baseline,base_groups),candidates={})
    for name,column in candidates.items():
        print(f'Loading and diagnosing candidate {name}',flush=True)
        candidate_record=dict(run,prediction_column=column)
        losses=reporting.load_losses(candidate_record,rdir)
        reporting.validate_comparison(candidate_record,parent,losses,baseline)
        groups=main_groups(losses)
        candidate=entry(losses,groups)
        candidate['delta_scores']={g:candidate['scores'][g]-result['parent']['scores'][g] for g in candidate['scores']}
        saved=run.get('candidate_scores',{}).get(name)
        if saved:
            if any(not np.isclose(candidate['scores'][g],saved['scores'][g],rtol=0,atol=1e-8)
                   for g in candidate['scores']):
                raise ValueError(f'{name}: scores disagree with the runner report')
            candidate['comparison']=saved['changes']
        sensitivity=run.get('comparisons',{}).get(f'{name}__block3')
        if sensitivity:
            payload=json.loads((rdir/sensitivity['path']).read_text())
            candidate['three_day_comparison']=payload['report']['comparison']['scores']
        candidate['changes']={g:changed_examples(groups[g],base_groups[g],annotations) for g in GROUP_FOLDS}
        candidate['influence']=influence(base_groups,groups,weight)
        result['candidates'][name]=candidate
        del losses,groups
        gc.collect()
    # Read descriptive prediction means only for the compact example roster.
    example_lists=[('control',g,rows) for g,rows in result['parent']['top_errors'].items()]
    for name,candidate in result['candidates'].items():
        example_lists.extend((name,g,rows) for g,rows in candidate['top_errors'].items())
        for g,changes in candidate['changes'].items():
            example_lists.extend((name,g,changes[key]) for key in ('fixes','harms'))
    ids={row['mvt_id'] for _,_,rows in example_lists for row in rows}
    del baseline,base_groups,annotations
    gc.collect()
    means=selected_prediction_means(parent,pdir,[args.parent_column],ids)
    candidate_means=selected_prediction_means(run,rdir,list(candidates.values()),ids) if run and candidates else {}
    for name,group,rows in example_lists:
        for row in rows:
            row['control_mean_prediction_sec']=float(means[group].loc[row['mvt_id'],args.parent_column])
            row['mean_prediction_sec']=(float(candidate_means[group].loc[row['mvt_id'],candidates[name]])
                                        if name!='control' else row['control_mean_prediction_sec'])
    audit_path=args.output/'raw_clock_audit.json'
    if audit_path.exists():
        audit=json.loads(audit_path.read_text())
        result['observable_prevalence']=dict(populations=audit['populations'],patterns=[r for r in audit['patterns']
            if r['population'] in ('train_2025','cv_unique_jul_dec','ranking_jan_jul')])
    (args.output/'error_diagnostics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (args.output/'ERROR_DIAGNOSTICS.md').write_text(markdown(result))
    plots(result,args.output)
    print(f'Wrote {args.output}/ERROR_DIAGNOSTICS.md and PNG/PDF figures',flush=True)


if __name__=='__main__':
    main()
