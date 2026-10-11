"""Run the two fixed follow-up experiments after the current desktop fit exits.

Finite local training queue, no submissions. Source hashes and a predecessor
result bind this queue to the reviewed jobs. Inactive objective or cancellation
stops the queue and its active child. Heavy jobs never overlap here.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from datetime import datetime, timezone

from records import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def process_start(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[19]
    except FileNotFoundError:
        return None


def run(path):
    path = Path(path).resolve()
    q = read(path)
    assert q['state'] == 'queued'
    assert q['runner_sha256'] == sha256(__file__)
    lock = path.with_suffix('.lock')
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    child = None

    def status(state, **extra):
        q.update(state=state, updated_at=datetime.now(timezone.utc).isoformat(), **extra)
        write_json(path, q)

    def cancelled():
        return ((ROOT / q['cancel_file']).exists()
                or read(ROOT / q['objective'])['state'] != 'active')

    try:
        status('waiting', pid=os.getpid())
        while process_start(q['predecessor_pid']) == q['predecessor_start']:
            if cancelled():
                status('cancelled')
                return
            time.sleep(10)
        assert (ROOT / q['predecessor_result']).exists(), 'Predecessor did not finish scoring'
        for job_index, job in enumerate(q['jobs']):
            if cancelled():
                status('cancelled')
                return
            for source, digest in job['sources'].items():
                assert sha256(ROOT / source) == digest, f'Queued source changed: {source}'
            for stage, argv in enumerate(job['commands']):
                if cancelled():
                    status('cancelled')
                    return
                with (ROOT / job['log']).open('ab', buffering=0) as log:
                    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2',
                               MKL_NUM_THREADS='1', PYTHONPATH=str(ROOT / 'VALIDATION'))
                    child = subprocess.Popen(argv, cwd=ROOT, env=env,
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                    status('running', job=job_index, stage=stage, child_pid=child.pid)
                    while child.poll() is None:
                        if cancelled():
                            child.terminate()
                            try:
                                child.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                child.kill()
                                child.wait()
                            status('cancelled')
                            return
                        time.sleep(5)
                assert child.returncode == 0, f'{job["name"]} stage {stage} failed: {child.returncode}'
                child = None
            assert (ROOT / job['result']).exists()
            job['completed_result_sha256'] = sha256(ROOT / job['result'])
            status('between_jobs', child_pid=None)
        status('complete', child_pid=None)
    except Exception as exc:
        status('error', error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('queue')
    run(parser.parse_args().queue)
