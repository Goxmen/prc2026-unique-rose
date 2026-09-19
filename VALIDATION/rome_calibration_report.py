"""Write completed Rome-calibration results and two validation-only ledger rows."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from records import write_json
import rome_calibration as recipe

NAMES={'binary':'Rome-binary-cal', 'three':'Rome-three-cal'}
DESCRIPTIONS={
    'binary': 'Row 34 replay; Rome binary probabilities calibrated on inner chronological predictions; soft mixture replaces the 4 h override; ordinary model reused',
    'three': 'Same soft Rome mixture with a three-class classifier (ordinary/schedule/day), inner probability calibration and a shrunk day component in the existing next-day scope; ordinary model reused',
}
SCORES=('R_A','R_B_long','R_combined')
PROBABILITIES={'binary_raw':2,'binary_cal':2,'three_raw':3,'three_cal_before_day':3,'three_cal':3}


def diagnostic_summary(frame):
    """Average seed/origin losses, not probabilities, with one weight per flight/month."""
    w=frame.weight.to_numpy(float)
    y=frame.y.to_numpy(float)
    labels={k:recipe.outcomes(y,frame.gap_sec.to_numpy(),k) for k in (2,3)}
    noid=frame.raw_flight_id_missing.to_numpy(bool)
    result=dict(probabilities={},cohorts={})
    for stage,k in PROBABILITIES.items():
        p=frame[[f'{stage}_p{i}' for i in range(k)]].to_numpy(float)
        recipe.validate_probabilities(p)
        truth=np.eye(k)[labels[k]]
        nll=-np.log(np.maximum(p[np.arange(len(p)),labels[k]],1e-8))
        brier=np.sum((p-truth)**2,axis=1)
        groups={}
        for name,mask in [('all',np.ones(len(frame),bool)),('id_present',~noid),('id_missing',noid)]:
            total=float(w[mask].sum())
            if not total:continue
            bins={}
            for i in range(k):
                slot=np.minimum((p[:,i]*10).astype(int),9)
                bins[str(i)]=[]
                for b in range(10):
                    selected=mask&(slot==b);weight=float(w[selected].sum())
                    if weight:
                        bins[str(i)].append(dict(low=b/10,high=(b+1)/10,weighted_rows=weight,
                            unique_flights=int(frame.loc[selected,'mvt_id'].nunique()),
                            mean_probability=float(np.average(p[selected,i],weights=w[selected])),
                            observed_fraction=float(np.average(truth[selected,i],weights=w[selected]))))
            groups[name]=dict(weighted_rows=total,unique_flights=int(frame.loc[mask,'mvt_id'].nunique()),
                logloss=float(np.average(nll[mask],weights=w[mask])),
                multiclass_brier=float(np.average(brier[mask],weights=w[mask])),reliability=bins)
        result['probabilities'][stage]=groups
    masks={'all_rome':np.ones(len(frame),bool),'target_at_most_2h':y<=7200,
           'schedule_class':labels[3]==1,'day_class':labels[3]==2,
           'day_scope':frame.day_eligible.to_numpy(bool),
           'day_class_in_scope':(labels[3]==2)&frame.day_eligible.to_numpy(bool)}
    for name,mask in masks.items():
        count=float(w[mask].sum())
        stats=dict(unique_flights=int(frame.loc[mask,'mvt_id'].nunique()),weighted_rows=count,predictions={})
        if count:
            for candidate,column in [('control','pred_noid_cap'),*recipe.COLUMNS.items()]:
                pred=frame[column].to_numpy(float)[mask];err=pred-y[mask];weight=w[mask]
                stats['predictions'][candidate]=dict(rmse=float(np.sqrt(np.average(err**2,weights=weight))),
                    bias=float(np.average(err,weights=weight)),
                    above_2h_fraction=float(np.average(pred>7200,weights=weight)),
                    above_12h_fraction=float(np.average(pred>43200,weights=weight)))
        result['cohorts'][name]=stats
    return result


def build_diagnostics(record,directory):
    columns=['mvt_id','mvt_time','y','gap_sec','raw_flight_id_missing','fold','seed',
             'day_eligible','pred_noid_cap',*recipe.COLUMNS.values()]
    columns += [f'{name}_p{i}' for name,k in PROBABILITIES.items() for i in range(k)]
    result={};day_cases=[];false_high=[]
    for group,folds in [('R_A',[f'A{i}' for i in range(1,7)]),('R_B_long',['B3','B4','B7'])]:
        origins=pd.Series([record['folds'][f]['val'] for f in folds]).value_counts()
        chunks=[]
        for fold in folds:
            info=record['folds'][fold]
            for seed in record['config']['seeds']:
                f=pd.read_parquet(directory/info['predictions'][str(seed)],columns=columns,filters=[('airport','==','LIRF')])
                f['weight']=1/(len(record['config']['seeds'])*origins[info['val']])
                chunks.append(f)
        frame=pd.concat(chunks,ignore_index=True)
        result[group]=diagnostic_summary(frame)
        day=(recipe.outcomes(frame.y,frame.gap_sec,3)==2)|frame.day_eligible.to_numpy(bool)
        ordinary=frame.y.le(7200)
        high=ordinary & frame[['pred_noid_cap',recipe.COLUMNS['binary'],recipe.COLUMNS['three']]].gt(7200).any(axis=1)
        day_cases.append(frame.loc[day].assign(group=group))
        false_high.append(frame.loc[high].assign(group=group))
    write_json(directory/'diagnostics.json',result)
    pd.concat(day_cases,ignore_index=True).to_csv(directory/'day_cases.csv',index=False)
    pd.concat(false_high,ignore_index=True).to_csv(directory/'ordinary_false_high.csv',index=False)
    return result


def update_submissions(path,run_id,results,report_link):
    """Add exactly two completed CV rows; preserve official scores and unrelated edits."""
    path=Path(path)
    original=path.read_text()
    lines=original.splitlines()
    table=[]
    for i,line in enumerate(lines):
        cells=[s.strip() for s in line.split('|')[1:-1]]
        if len(cells)==8 and cells[0].isdigit():table.append((i,int(cells[0])))
    if not table:raise ValueError('Cannot find the eight-column submission table')
    additions=[];next_number=max(n for _,n in table)+1
    for name in ('binary','three'):
        score=results[name]['scores']
        if any(not isinstance(score.get(k),(int,float)) or not math.isfinite(score[k]) or score[k]<0
               for k in SCORES):raise ValueError('Missing complete validation scores')
        existing=[(i,n) for i,n in table if f'**{NAMES[name]}**' in lines[i] and f'({report_link})' in lines[i]]
        if len(existing)>1:raise ValueError('Duplicate experiment ledger rows')
        number=existing[0][1] if existing else next_number
        row=(f'| {number} | **{NAMES[name]}** | {DESCRIPTIONS[name]}; '
             f'[run details]({report_link}) | '+ ' | '.join(f'{score[k]:.4f}' for k in SCORES)+' | — | Not submitted |')
        if existing:lines[existing[0][0]]=row
        else:additions.append(row);next_number+=1
    if additions:lines[table[-1][0]+1:table[-1][0]+1]=additions
    marker=f'<!-- rome-calibration:{run_id} -->'
    if marker not in lines:
        lines+=['',marker,
            f'The two Rome-calibration rows above are local validation experiments from `{run_id}`. '
            'Their probability calibrators use inner chronological predictions inside each training window. '
            'Both remove Rome’s hard four-hour override; the linked report separates that change from calibration. '
            'All 16 folds and three seeds remain. No full-year fit or leaderboard upload was performed.']
    updated='\n'.join(lines)+'\n'
    if path.read_text()!=original:raise ValueError('SUBMISSIONS.md changed during report writing; resume to retry')
    temp=path.with_name(path.name+'.tmp');temp.write_text(updated);temp.replace(path)


def finish(record,directory,root):
    directory,root=Path(directory),Path(root)
    expected={f'A{i}' for i in range(1,7)}|{f'B{i}' for i in range(1,11)}
    if (set(record['folds'])!=expected or record['config']['seeds']!=[42,142,242]
            or record['config']['sample']!=1 or record['config'].get('smoke',False)
            or record['fits']['new_classifiers']!=126
            or any(set(f['predictions'])!={'42','142','242'} for f in record['folds'].values())):
        raise ValueError('Only complete full-data, three-seed validation may update the ledger')
    reports={key:json.loads((directory/info['path']).read_text())['report']
             for key,info in record['comparisons'].items()}
    main={name:reports[f'{name}_vs_control_block1'] for name in ('binary','three')}
    diagnostics=build_diagnostics(record,directory)
    parent=json.loads((root/'VALIDATION/runs'/record['baseline_id']/'result.json').read_text())
    lines=['# Rome binary and three-class calibration', '',
           f"Run `{record['run_id']}`; matched control `{record['baseline_id']}`.", '',
           'Two frozen local-validation candidates; all original evaluation rows, 16 folds and three seeds. '
           'Calibrators use three expanding inner chronological splits within each outer training window. '
           'Ordinary predictions are reused. Neither candidate has an official leaderboard score.', '',
           '| Recipe | R_A | R_B_long | Combined | Combined change vs control, 95% day interval |',
           '|---|---:|---:|---:|---|']
    control=parent['report']['scores']
    lines.append('| Matched row-34 replay | '+' | '.join(f'{control[k]:.4f}' for k in SCORES)+' | — |')
    for name in ('binary','three'):
        r=main[name];delta=r['comparison']['scores']['R_combined']
        lines.append(f'| {NAMES[name]} | '+' | '.join(f"{r['scores'][k]:.4f}" for k in SCORES)+
            f" | {delta['difference']:+.4f} [{delta['ci95'][0]:+.4f}, {delta['ci95'][1]:+.4f}] |")
    artifact=root/'VALIDATION/runs/v3_20260919_artifact_experiments/result.json'
    if artifact.exists():
        old=json.loads(artifact.read_text())['candidate_scores']['combined']['scores']
        lines+=['', 'The separate saved artifact-combined candidate (submission-table row 37) scored '
                + ' / '.join(f'{old[k]:.4f}' for k in SCORES)
                + ' s. This is a descriptive additional reference; the paired control above is row 34.']
    lines+=['','## What changed, and whether it helped','',
            'Both candidates replace the Rome hard override with a soft mixture. The binary classifier is reused '
            'and calibrated; the three-class classifier changes the outcomes and fits new Rome classifiers. '
            'The day class is restricted to no-ID Rome rows at >=14h schedule gap crossing one local date, '
            'then shrunk by training support. It cannot detect day artifacts outside that scope. '
            'The flight-plan window and no-ID cap remain; predictions outside Rome are checked unchanged.', '',
            '| Comparison | R_A change | R_B_long change | Combined change [95% interval] | Pointwise verdict |',
            '|---|---:|---:|---|---|']
    for key in ['binary_vs_control_block1','three_vs_control_block1','binary_vs_binary_uncal_block1',
                'three_vs_binary_block1','three_vs_three_uncal_block1','binary_cal_hard_vs_control_block1',
                'binary_uncal_vs_control_block1','binary_vs_control_block3','three_vs_control_block3']:
        r=reports[key]['comparison']['scores'];c=r['R_combined']
        lines.append(f"| {key} | {r['R_A']['difference']:+.4f} | {r['R_B_long']['difference']:+.4f} | "
                     f"{c['difference']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] | {c['decision']} |")
    lines+=['','Negative changes favour the first recipe. `binary_uncal` is the original binary mixture '
            'without the hard override; `binary_cal_hard` keeps the original override. These isolate override '
            'removal from calibration. `three_uncal` keeps the same day restrictions but omits calibration. '
            'One-day and three-day intervals use 2,000 paired bootstrap draws. All comparisons reuse '
            'development folds; no result is independent confirmation or automatic promotion.', '',
            '## Seed and label-slice checks', '',
            '| Candidate | Readout | Seed 42 | Seed 142 | Seed 242 | Target <=2h RMSE | Target >2h RMSE |',
            '|---|---|---:|---:|---:|---:|---:|']
    for name,r in main.items():
        for group in ('R_A','R_B_long'):
            g=r['groups'][group]
            lines.append(f'| {NAMES[name]} | {group} | '+' | '.join(f"{g['rmse_by_seed'][str(s)]:.4f}" for s in (42,142,242))+
                         f" | {g['ordinary']['rmse']:.4f} | {g['tail']['rmse']:.4f} |")
    lines+=['','## Probability and failure diagnostics','',
            'Losses below average each seed/origin loss with the same flight/month weighting as its score group. '
            'Probabilities are not averaged into an ensemble. Brier is the sum across classes; compare it '
            'within a classifier, since the binary and three-class outcomes differ. '
            'The JSON includes all/present-ID/missing-ID reliability bins for every class.', '',
            '| Group | Probability stage | ID stratum | Log loss | Multiclass Brier |',
            '|---|---|---|---:|---:|']
    for group,d in diagnostics.items():
        for stage,strata in d['probabilities'].items():
            for stratum in ('all','id_missing'):
                if stratum not in strata:continue
                s=strata[stratum]
                lines.append(f"| {group} | {stage} | {stratum} | {s['logloss']:.6f} | {s['multiclass_brier']:.6f} |")
    lines+=['','`three_cal_before_day` isolates calibration; `three_cal` includes the deployment gate and shrinkage.', '',
            '| Group | Cohort | Unique flights | Candidate | RMSE | Mean error | Predicted >2h | Predicted >12h |',
            '|---|---|---:|---|---:|---:|---:|---:|']
    for group,d in diagnostics.items():
        for cohort in ('target_at_most_2h','day_class','day_scope','day_class_in_scope'):
            s=d['cohorts'][cohort]
            for candidate in ('control','binary','three'):
                if candidate not in s['predictions']:continue
                p=s['predictions'][candidate]
                lines.append(f"| {group} | {cohort} | {s['unique_flights']} | {candidate} | {p['rmse']:.4f} | "
                             f"{p['bias']:+.4f} | {100*p['above_2h_fraction']:.4f}% | {100*p['above_12h_fraction']:.4f}% |")
    lines+=['','Positive mean error is overprediction. `day_class` uses schedule precedence; '
            '`day_scope` uses observable fields only. Counts are unique flights within each group; '
            'groups overlap and must not be added. All rare day cases/scope rows are in [day_cases.csv](day_cases.csv); '
            'ordinary flights predicted above two hours by control/either candidate are in '
            '[ordinary_false_high.csv](ordinary_false_high.csv), retaining seed/origin identifiers. '
            'These are diagnostic slices; headline scores keep every row.']
    lines+=['','## Training support and workload','',
            '| Window | Ordinary/other | Schedule | Day | Day examples in deployment scope |','|---|---:|---:|---:|---:|']
    for name,w in record['windows'].items():
        lines.append(f"| {name} | {w['classes'][0]} | {w['classes'][1]} | {w['classes'][2]} | {w['day_count']} |")
    lines+=['',f"Completed {record['fits']['new_classifiers']} new classifiers, reused 18 outer binary classifiers "
            'and all ordinary predictions. Saved inner predictions and calibrators, 48 outer prediction files, '
            'nine paired comparisons and artifact/source hashes. No full-year fit or upload.', '',
            'The design is in [ROME_CALIBRATION.md](../../docs/ROME_CALIBRATION.md). '
            'Class outcomes use training labels only; deployment accepts observable inputs. '
            'The day class is sparse and overlaps across training windows; its counts are not independent evidence.', '',
            '```bash',f'.venv/bin/python VALIDATION/rome_calibration_run.py --output-dir VALIDATION/runs/{record["run_id"]} --resume','```','']
    report_path=directory/'report.md';report_path.write_text('\n'.join(lines))
    relative=str(report_path.relative_to(root))
    update_submissions(root/'SUBMISSIONS.md',record['run_id'],main,relative)
    summary={name:dict(scores={k:r['scores'][k] for k in SCORES},
                      changes={k:r['comparison']['scores'][k] for k in SCORES}) for name,r in main.items()}
    write_json(directory/'scorecard.json',summary)
    return dict(report='report.md',scorecard='scorecard.json',ledger='SUBMISSIONS.md',
                diagnostics='diagnostics.json',day_cases='day_cases.csv',false_high='ordinary_false_high.csv')
