"""Compact, deterministic research status and hypothesis lookup; no training."""
import argparse,json,re,subprocess
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
REGISTRY=ROOT/'VALIDATION/research_registry.json'
STATE=ROOT/'VALIDATION/research_state.json'
RUNS=ROOT/'VALIDATION/runs'


def read(p):return json.loads(p.read_text())


def local_job(path):
    log=path/'fit.log'
    if not log.exists():return dict(stage='pending')
    with log.open('rb') as f:
        f.seek(max(0,log.stat().st_size-8192));tail=f.read().decode(errors='replace')
    matches=re.findall(r'\[(\d+:\d+:\d+)\] (\S+) tabm epoch(\d+)/(\d+): train=([\d.]+); seconds=([\d.]+)',tail)
    record=dict(stage='training_or_inference',completed_records=len(list(path.glob('networks/*.json'))))
    if matches:
        clock,fold,epoch,total,loss,seconds=matches[-1]
        record.update(last_epoch_at=clock,fold=fold,epoch=int(epoch),epochs=int(total),seconds_per_epoch=float(seconds))
    if (path/'audit.json').exists():record['stage']='fits_complete'
    if 'Traceback (most recent call last)' in tail:record.update(stage='inspect_failure',failure_tail=tail[-500:])
    return record


REMOTE="""
import json,re
from pathlib import Path
root=Path('/home/gunes/PRC')
p=root/'VALIDATION/runs/v3_20261012_nn_public_tabm_daily_cohort'
daily=p.exists()
if not daily:p=root/'VALIDATION/runs/v3_20261012_nn_public_tabm_depth3'
done=[m for m in ['2025-01','2025-07'] if (p/f'k0/networks/{m}_tabm.json').exists()]
log=root/('tmp/daily_cohort_fit.log' if daily else 'tmp/tabm_depth3_fit.log')
tail=''
if log.exists():
 with log.open('rb') as f:
  f.seek(max(0,log.stat().st_size-8192));tail=f.read().decode(errors='replace')
matches=re.findall(r'\\[(\\d+:\\d+:\\d+)\\] (\\S+) tabm epoch(\\d+)/(\\d+): train=([\\d.]+); seconds=([\\d.]+)',tail)
r=dict(stage='cv_fits_complete' if (p/'audit.json').exists() else ('daily_cohort_fit_or_inference' if daily else 'depth3_fit_or_inference'),completed_folds=done)
if not log.exists():r['stage']='preflight_pending'
r.update(run=str(p.relative_to(root)),preflight_pass=(p/'preflight.json').exists())
if matches:
 clock,fold,epoch,total,loss,seconds=matches[-1]
 r.update(last_epoch_at=clock,fold=fold,epoch=int(epoch),epochs=int(total),seconds_per_epoch=float(seconds))
if 'Traceback (most recent call last)' in tail:r.update(stage='inspect_failure',failure_tail=tail[-500:])
print(json.dumps(r))
"""


def status(remote=False):
    objective=read(RUNS/'v3_20261011_target215/objective.json')
    value=dict(updated_at=datetime.now(timezone.utc).isoformat(),target=objective['target_rmse'],
        best_version=objective['current_best_version'],best_official_rmse=objective['current_best_rmse'],
        official_checked_at=objective['last_live_check'],slots_remaining=objective['remaining_utc_day_slots'],
        registry=str(REGISTRY.relative_to(ROOT)),hypotheses=len(read(REGISTRY)['entries']),
        desktop=dict(stage='final_catboost_inference',
            old_replay=(RUNS/'v3_20261012_final_public_catboost_inference/old.json').exists(),
            final_replay=(RUNS/'v3_20261012_final_public_catboost_inference/final.json').exists()),
        next=read(REGISTRY)['next'],
        native_goal_status=read(REGISTRY)['current'].get('native_goal_status'),
        active_research=read(REGISTRY).get('active_research',[]))
    if remote:
        r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','glaptop-wsl',
            'cd /home/gunes/PRC && .venv/bin/python -'],input=REMOTE,text=True,capture_output=True,timeout=30,check=True)
        value['laptop']=json.loads(r.stdout)
    elif STATE.exists():
        prior=read(STATE)
        value['laptop']=dict(prior.get('laptop') or {},stale=True)
        value['laptop'].setdefault('last_checked_at',prior['updated_at'])
    result=RUNS/'v3_20261012_nn_public_tabm_duration_stack/combined_result.json'
    if result.exists():
        r=read(result);value['duration_result']={k:r[k] for k in ['cv','paired','gate_pass']}
        value['duration_result']['status']='closed_inconclusive'
    if value['desktop']['final_replay']:value['desktop']['stage']='final_public_models_ready'
    completion=RUNS/'v3_20261012_tabm_pool3_completion/execution.json'
    if completion.exists():
        r=read(completion)
        value['completion']={k:r[k] for k in ['state','active_stage','candidate','failure'] if k in r}
        value['completion']['completed_steps']=len(r['steps'])
    final=RUNS/'v3_20261012_final_public_tabm_inference'
    value['final_tabm_local_receipts']={view:(final/f'{view}.json').exists() for view in ['old','final']}
    legacy=RUNS/'v3_20261012_final_legacy_audit/audit.json'
    if legacy.exists():
        r=read(legacy)
        value['legacy_final']=dict(status=r['status'],versions=[a['version'] for a in r['predictions']],
            excluded_mappings=[a['version'] for a in r['excluded_mappings']])
        value['desktop']['stage']='retained_final_components_audited_missing_basis_pending'
    elif (RUNS/'v3_20261012_final_legacy_adsb_replay/identity.json').exists():
        value['desktop']['stage']='legacy_adsb_final_inference'
    means=RUNS/'v3_20261012_final_legacy_airport_mean_replay/result.json'
    if means.exists():
        r=read(means)
        assert r['status']=='pass' and r['views']['old']['old_integer_reference_exact']
        value['legacy_final']['versions']=sorted(set(value['legacy_final']['versions']+[r['version']]))
        value['legacy_final']['additional_receipts']=[str(means.relative_to(ROOT))]
    depth=RUNS/'v3_20261012_nn_public_tabm_depth3'
    if (depth/'completion/execution.json').exists():
        r=read(depth/'completion/execution.json')
        value['depth3_completion']={k:r[k] for k in ['state','active_stage','steps','failure'] if k in r}
    if (depth/'comparison/independent_audit.json').exists():
        r=read(depth/'comparison/result.json')
        value['depth3_result']={k:r[k] for k in ['cv','paired','gate','scores']}
        if not remote and not (RUNS/'v3_20261012_nn_public_tabm_daily_cohort/identity.json').exists():
            value['laptop']=dict(stage='fixed_depth3_screen_complete',training_receipt=str((depth/'audit.json').relative_to(ROOT)))
    basic=RUNS/'v3_20261012_final_missing_basic_reconstruct'
    for family in ['baseline','recovery']:
        result=basic/family/'result.json';audit=basic/family/'independent_audit.json'
        if result.exists() and audit.exists():
            r=read(result);a=read(audit)
            assert a['status']=='pass'
            versions=[int(v) for v,t in r['views']['old'].items() if t['exact_original_integer_replay']]
            assert len(versions)==len(r['views']['final'])
            value['legacy_final']['versions']=sorted(set(value['legacy_final']['versions']+versions))
            value['legacy_final'].setdefault('additional_receipts',[]).append(str(audit.relative_to(ROOT)))
    allocation=read(REGISTRY).get('machine_allocation',{})
    if 'desktop' in allocation:value['desktop']=allocation['desktop']
    if not remote and 'laptop' in allocation:value['laptop']=dict(allocation['laptop'],stale=True)
    destination=STATE.with_suffix('.json.tmp');destination.write_text(json.dumps(value,indent=2)+'\n');destination.replace(STATE)
    print(json.dumps(value,separators=(',',':')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['status','find'])
    p.add_argument('query',nargs='?',default='');p.add_argument('--remote',action='store_true');a=p.parse_args()
    if a.command=='status':status(a.remote)
    else:
        terms=a.query.lower().split()
        for row in read(REGISTRY)['entries']:
            if all(term in json.dumps(row).lower() for term in terms):
                print(json.dumps({k:row[k] for k in ['id','hypothesis','status','conclusion','metrics'] if k in row}))
        # Index the earlier comparison tables without bringing whole reports
        # into conversational context. These are evidence pointers, not new
        # experiments or automatically inferred decisions.
        hits=[]
        for name in ['RESEARCH_NETWORK_20261010.md','TARGET215_RESEARCH.md']:
            path=ROOT/'VALIDATION/reports'/name
            for lineno,line in enumerate(path.read_text().splitlines(),1):
                if terms and all(term in line.lower() for term in terms):
                    hits.append(dict(report=str(path.relative_to(ROOT)),line=lineno,evidence=line[:500]))
        if hits:print(json.dumps(dict(historical_matches=len(hits),matches=hits[:8],truncated=len(hits)>8)))
