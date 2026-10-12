"""Collect, score, audit and register the already-running fixed screen once ready.

Starts no fits or uploads. Work is sequential on the desktop; the legacy audit
receipt is a prerequisite so it cannot overlap that host's inference work.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'VALIDATION/runs/v3_20261012_nn_public_tabm_depth3'
OUT = RUN / 'completion'


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'execution.json').exists(), 'Inspect existing completion before starting another'
    tasks = [
        ('return', 'VALIDATION/tools/return_tabm_depth3_20261012.py', [], RUN / 'return_audit.json'),
        ('score', 'VALIDATION/nn_public_tabm_depth3_score.py', ['score'], RUN / 'comparison/result.json'),
        ('audit', 'VALIDATION/tools/audit_tabm_depth3_20261012.py', [], RUN / 'comparison/independent_audit.json'),
    ]
    sources = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for _, name, _, _ in tasks}
    state = dict(state='waiting_for_existing_jobs', started_at=now(), steps=[], source_hashes=sources,
                 no_new_training=True, no_upload=True)
    write(OUT / 'execution.json', state)
    try:
        deadline = time.monotonic() + 3 * 3600
        while True:
            assert time.monotonic() < deadline, 'Existing job readiness timeout; inspect without restarting fits'
            desktop_ready = (ROOT / 'VALIDATION/runs/v3_20261012_final_legacy_audit/audit.json').exists()
            response = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                'glaptop-wsl', 'test -f /home/gunes/PRC/VALIDATION/runs/v3_20261012_nn_public_tabm_depth3/audit.json'],
                capture_output=True, text=True, timeout=20)
            state.update(last_checked_at=now(), desktop_ready=desktop_ready, laptop_ready=response.returncode == 0)
            write(OUT / 'execution.json', state)
            if desktop_ready and response.returncode == 0:
                break
            time.sleep(30)
        for label, script, args, receipt in tasks:
            assert hashlib.sha256((ROOT / script).read_bytes()).hexdigest() == sources[script]
            state.update(state='running', active_stage=label)
            write(OUT / 'execution.json', state)
            if not receipt.exists():
                with (OUT / f'{label}.log').open('w') as log:
                    subprocess.run([str(ROOT / '.venv/bin/python'), script, *args], cwd=ROOT,
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
            state['steps'].append(dict(stage=label, completed_at=now(), receipt=str(receipt.relative_to(ROOT))))
            write(OUT / 'execution.json', state)
        result = json.loads((RUN / 'comparison/result.json').read_text())
        audit = json.loads((RUN / 'comparison/independent_audit.json').read_text())
        assert audit['status'] == 'pass' and audit['gate'] == result['gate']
        registry_path = ROOT / 'VALIDATION/research_registry.json'
        registry = json.loads(registry_path.read_text())
        record = next(a for a in registry['entries'] if a['id'] == 'tabm_three_block_capacity')
        record['status'] = 'screen_passed_confirmation_pending' if result['gate'] else 'closed_fixed_screen'
        record['metrics'] = dict(cv=result['cv'], paired=result['paired'], scores=result['scores'], gate=result['gate'])
        record['conclusion'] = ('The fixed extra-block screen passes both months and paired-day criterion; additional seed confirmation remains necessary before selection.'
                               if result['gate'] else 'The fixed extra-block screen fails its frozen gate. No epoch/architecture tuning, seed expansion, production or candidate follows.')
        record['result'] = str((RUN / 'comparison/result.json').relative_to(ROOT))
        record['audit'] = str((RUN / 'comparison/independent_audit.json').relative_to(ROOT))
        registry['updated_at'] = now()
        write(registry_path, registry)
        report = ROOT / 'VALIDATION/reports/TARGET205_RESEARCH.md'
        with report.open('a') as f:
            f.write('\n\n## Fixed three-block TabM screen complete\n\n')
            f.write(f"Completed at {now()}. Exact v80 CV {result['cv']['control']:.6f} → {result['cv']['candidate']:.6f}; "
                    f"change {result['paired']['change']:+.6f}, paired-day95% interval {result['paired']['ci95']}. "
                    f"Frozen gate: {result['gate']}. " + record['conclusion'] + '\n\n')
            f.write('Both saved models, complete epoch curves, all-row GPU reloads, source hashes, SQL transport and independent bootstrap checks pass. '
                    'The development folds were reused; this is not independent confirmation. All artifacts are retained in '
                    '`../runs/v3_20261012_nn_public_tabm_depth3/`. No upload ran.\n')
        state.update(state='complete', active_stage=None, completed_at=now(), gate=result['gate'],
                     cv=result['cv'], paired=result['paired'])
        write(OUT / 'execution.json', state)
        subprocess.run([str(ROOT / '.venv/bin/python'), 'VALIDATION/tools/research_state.py', 'status'], cwd=ROOT, check=True)
        print(json.dumps(dict(state='complete', gate=result['gate'], cv=result['cv'], paired=result['paired'])), flush=True)
    except Exception as error:
        state.update(state='failed', failed_at=now(), failure=repr(error))
        write(OUT / 'execution.json', state)
        raise


if __name__ == '__main__':
    main()
