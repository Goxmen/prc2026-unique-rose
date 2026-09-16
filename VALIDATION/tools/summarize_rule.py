"""Summarize fixed-rule gains and damaging overrides from saved candidate artifacts."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from records import ROOT, read_run, write_json


def changed_summary(frame, names):
    selected = frame.loc[frame.fold.isin(names)].copy()
    # Equal weight per seed and per repeated origin of each flight, as in v3.
    per_seed = selected.groupby(['mvt_id','seed'], sort=True)[
        ['parent_squared_error','rule_squared_error','delta_squared_error']].mean()
    values = per_seed.groupby('mvt_id').mean()
    metadata = selected.drop_duplicates('mvt_id').set_index('mvt_id')
    for key in ['mvt_time','y','pred','rule_gap_sec']:
        values[key] = metadata[key]
    values['parent_prediction_mean'] = selected.groupby('mvt_id').model_pred.mean()
    values['month'] = pd.to_datetime(values.mvt_time, utc=True).dt.strftime('%Y-%m')
    result = {'unique_triggered': len(values),
              'helped': int(values.delta_squared_error.lt(0).sum()),
              'harmed': int(values.delta_squared_error.gt(0).sum()),
              'unchanged': int(values.delta_squared_error.eq(0).sum()),
              'delta_sse': float(values.delta_squared_error.sum()),
              'monthly': {}, 'flights': values.reset_index().to_dict('records')}
    for month, group in values.groupby('month'):
        result['monthly'][month] = {'unique_triggered': len(group),
            'helped': int(group.delta_squared_error.lt(0).sum()),
            'harmed': int(group.delta_squared_error.gt(0).sum()),
            'delta_sse': float(group.delta_squared_error.sum())}
    result['worst_overrides'] = values.loc[values.delta_squared_error.gt(0)].sort_values(
        ['delta_squared_error','mvt_id'], ascending=[False,True]).head(5).reset_index().to_dict('records')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--q0', required=True)
    ap.add_argument('--q1', required=True)
    args = ap.parse_args()
    q0, d0 = read_run(args.q0)
    q1, d1 = read_run(args.q1)
    if any(r['status'] != 'complete' for r in [q0,q1]):
        raise ValueError('Both rule candidates must be complete')
    versus = json.loads((d1/f'comparison_{q0["run_id"]}_block1.json').read_text())['report']['comparison']
    out = {'rule': q0['config']['prediction_rule'], 'candidates': {},
           'q1_vs_q0': versus,
           'weighting': 'Per flight, mean squared error over seeds and repeated training origins.'}
    for label, run, directory in [('Q0',q0,d0),('Q1',q1,d1)]:
        changes = pd.read_csv(directory/run['changed_rows'])
        out['candidates'][label] = {'run_id': run['run_id'], 'parent': run['model_parent_id'],
            'scores': run['report']['scores'], 'vs_parent': run['report']['comparison'],
            'groups': run['report']['groups'], 'changes': {}}
        for group, names in [('R_A',[f'A{i}' for i in range(1,7)]),('R_B_long',['B3','B4','B7'])]:
            out['candidates'][label]['changes'][group] = changed_summary(changes,names)
    for group in ['R_A','R_B_long']:
        a = out['candidates']['Q0']['changes'][group]['flights']
        b = out['candidates']['Q1']['changes'][group]['flights']
        if [(x['mvt_id'],x['pred']) for x in a] != [(x['mvt_id'],x['pred']) for x in b]:
            raise ValueError('Recovery changed rule eligibility or override values')
    out['identical_trigger_ids_and_override_values'] = True
    directory = ROOT/'VALIDATION/reports'
    write_json(directory/'stage2_comparison.json',out)
    fig, axes = plt.subplots(1,2,figsize=(11,4.8),sharey=True)
    fig.subplots_adjust(left=.22,right=.97,top=.76,bottom=.23,wspace=.32)
    fig.suptitle('Stage 2 · the fixed Rome six-hour rule',x=.03,ha='left',fontsize=17,weight='bold')
    fig.text(.03,.85,'Paired RMSE difference and 95% day-bootstrap interval · lower is better',fontsize=11)
    pairs = [('Q0 − B0-800',q0['report']['comparison']),
             ('Q1 − R3',q1['report']['comparison']),('Q1 − Q0',versus)]
    for ax, group, title in zip(axes,['R_A','R_B_long'],['Next month','Longer gap']):
        for y, (label,comparison) in enumerate(pairs):
            item = comparison['scores'][group]
            color = '#087f8c' if item['decision']=='improvement' else '#b34b36' if item['decision']=='regression' else '#596273'
            ax.plot(item['ci95'],[y,y],color=color,linewidth=2)
            ax.plot(item['difference'],y,'o',color=color,markersize=7)
        ax.axvline(0,color='#999999',linewidth=.8,linestyle='--')
        ax.set(title=f'{title} ({group})',xlabel='Candidate − reference RMSE (seconds)',ylim=(2.5,-.5))
        ax.set_yticks(range(3),[x[0] for x in pairs])
        ax.grid(axis='x',alpha=.15)
        ax.spines[['top','right','left']].set_visible(False)
    fig.text(.03,.06,'Saved predictions · no new fits · all validation labels · 2,000 paired UTC-day replicates\n'
             'Fixed raw missing-ID trigger; reused development folds, not official leaderboard scores.',fontsize=9,color='#555555')
    for extension in ['png','pdf']:
        fig.savefig(directory/f'stage2_comparison.{extension}',dpi=170)
    plt.close(fig)
    print(directory/'stage2_comparison.json')


if __name__ == '__main__':
    main()
