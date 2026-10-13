"""Hash inspection and locked workers; invoked only by the manual runner."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from refit_io import ROOT, PYTHON, environment, now, read, resolve, save, sha

def process_start(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError):
        return None

def alive(record):
    return bool(record.get('pid') and record.get('process_start') and
                process_start(record['pid']) == record['process_start'])

def inspect_files(paths):
    return {p: sha(resolve(p)) if resolve(p).is_file() else None for p in paths}

def busy_processes():
    found = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            argv = path.read_bytes().split(b'\0')
            if 'python' not in Path(os.fsdecode(argv[0])).name:
                continue
            args = [os.fsdecode(v) for v in argv[1:] if v]
            # Do not report ourselves or controller; actual model scripts only.
            if any(v.endswith('.py') and ('VALIDATION/' in v or 'tools/final_selection/' in v or
                   ('tools/v80_refit/' in v and Path(v).name in {'trees.py','neural.py','assemble.py'})) for v in args):
                found.append(dict(pid=int(path.parent.name), script=next(v for v in args if v.endswith('.py'))))
        except (OSError, IndexError, StopIteration):
            continue
    return found

def worker(out, stage, threads):
    folder = resolve(out) / 'execution'
    folder.mkdir(parents=True, exist_ok=True)
    state = folder / (stage + '.json')
    lockpath = ROOT / 'VALIDATION/runs/v80_manual_host.lock'
    with lockpath.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        busy = busy_processes()
        if busy:
            save(state, dict(status='failed', reason='Other heavy Python jobs are active', processes=busy))
            return 1
        record = dict(status='running', stage=stage, pid=os.getpid(), process_start=process_start(os.getpid()),
                      host=os.uname().nodename, started_at=now().isoformat())
        save(state, record)
        child = None
        def stop(signum, frame):
            if child is not None and child.poll() is None:
                child.terminate()
            raise KeyboardInterrupt
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(sig, stop)
        try:
            command = [PYTHON, '-u', f'tools/v80_refit/{stage}.py', 'run', '--out', str(resolve(out)), '--threads', str(threads)]
            child = subprocess.Popen(command, cwd=ROOT, env=environment(threads))
            record['child_pid'] = child.pid
            save(state, record)
            code = child.wait()
            record.update(status='finished' if code == 0 else 'failed', returncode=code)
        except BaseException as exc:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()
            record.update(status='interrupted', error=type(exc).__name__)
        finally:
            record['finished_at'] = now().isoformat()
            save(state, record)
        return 0 if record['status'] == 'finished' else 1

def launch(out, stage, threads):
    folder = resolve(out) / 'execution'
    folder.mkdir(parents=True, exist_ok=True)
    state = folder / (stage + '.json')
    if state.exists() and alive(read(state)):
        return read(state)
    if state.exists() and read(state)['status'] == 'finished':
        return read(state)
    logfile = folder / (stage + '.log')
    with logfile.open('ab') as log:
        p = subprocess.Popen([PYTHON, str(Path(__file__)), 'worker', '--out', out,
                              '--stage', stage, '--threads', str(threads)], cwd=ROOT,
                             env=environment(threads), stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
    for _ in range(50):
        if state.exists() and read(state).get('pid') == p.pid:
            return read(state)
        if p.poll() is not None:
            raise RuntimeError('Worker could not start; see ' + str(logfile))
        time.sleep(.1)
    raise RuntimeError('Worker startup receipt missing; do not start a duplicate')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['inspect','health','launch','worker','status','stop'])
    parser.add_argument('--out', default='VALIDATION/runs/v3_20261013_v80_refit_final')
    parser.add_argument('--stage', choices=['trees','neural'], default='neural')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.command == 'inspect':
        result = inspect_files(json.load(sys.stdin))
    elif args.command == 'health':
        result = dict(host=os.uname().nodename, busy=busy_processes(),
                      available_kb=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))))
    elif args.command == 'launch':
        result = launch(args.out, args.stage, args.threads)
    elif args.command == 'worker':
        raise SystemExit(worker(args.out, args.stage, args.threads))
    else:
        path = resolve(args.out) / 'execution' / (args.stage + '.json')
        result = read(path) if path.exists() else dict(status='absent')
        result['alive'] = alive(result)
        if args.command == 'stop' and result['alive']:
            os.kill(result['pid'], signal.SIGTERM)
            result['stop_requested'] = True
    print(json.dumps(result))

if __name__ == '__main__':
    main()
