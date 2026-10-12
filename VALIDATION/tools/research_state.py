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
p=Path('/home/gunes/PRC/VALIDATION/runs/v3_20261012_nn_public_tabm_pool3_submit')
done=[k for k in [1,2] if (p/f'k{k}/networks/k0.json').exists()]
log=p/'fit.log'
with log.open('rb') as f:
 f.seek(max(0,log.stat().st_size-8192));tail=f.read().decode(errors='replace')
matches=re.findall(r'\\[(\\d+:\\d+:\\d+)\\] full_year tabm epoch(\\d+)/(\\d+): train=([\\d.]+); seconds=([\\d.]+)',tail)
r=dict(stage='export_complete' if (p/'export.json').exists() else 'fit_or_inference',completed_new_members=done)
if matches:
 clock,epoch,total,loss,seconds=matches[-1]
 r.update(last_epoch_at=clock,epoch=int(epoch),epochs=int(total),seconds_per_epoch=float(seconds))
if 'Traceback (most recent call last)' in tail:r.update(stage='inspect_failure',failure_tail=tail[-500:])
print(json.dumps(r))
"""


def status(remote=False):
    objective=read(RUNS/'v3_20261011_target215/objective.json')
    value=dict(updated_at=datetime.now(timezone.utc).isoformat(),target=objective['target_rmse'],
        best_version=objective['current_best_version'],best_official_rmse=objective['current_best_rmse'],
        official_checked_at=objective['last_live_check'],slots_remaining=objective['remaining_utc_day_slots'],
        registry=str(REGISTRY.relative_to(ROOT)),hypotheses=len(read(REGISTRY)['entries']),
        desktop=local_job(RUNS/'v3_20261012_nn_public_tabm_duration'),
        next=['Return/audit selected production ensemble','Score/audit fixed duration comparison','Submit one audited candidate'])
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
                print(json.dumps({k:row[k] for k in ['id','hypothesis','status','conclusion','metrics']}))
