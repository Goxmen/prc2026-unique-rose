"""Return completed laptop folds and score after both hosts finish. Never read logs."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "VALIDATION/runs/v3_20261013_final_month_selection"
MONTHS = ("2025-01", "2025-02", "2025-06", "2025-07")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def unlocked(path):
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True


def remote_state(host, relative):
    # Only completion markers and run-source identity, never a log file.
    script = """import json,pathlib,hashlib
r=pathlib.Path.home()/'PRC'/RELATIVE
reduced=r/'reduced_three'
result={'ready':[], 'bindings':{}}
identity=reduced/'identity.json'
if identity.exists(): result['identity_sha256']=hashlib.sha256(identity.read_bytes()).hexdigest()
for m in ('2025-02','2025-06'):
 p=reduced/'folds'/m
 if all((p/n).is_file() for n in ('predictions.json','evaluation.json')):
  result['ready'].append(m)
  result['bindings'][m]=hashlib.sha256((r/'folds'/m/'input_binding.json').read_bytes()).hexdigest()
print(json.dumps(result))
""".replace("RELATIVE", repr(str(relative)))
    command = "cd ~/PRC && .venv/bin/python -c " + shlex.quote(script)
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, command],
                            capture_output=True, text=True, timeout=40)
    if result.returncode:
        raise RuntimeError(f"Laptop status unavailable (ssh exit {result.returncode})")
    return json.loads(result.stdout)


def check_identity(remote, local_identity):
    if remote['ready'] and (not remote.get('identity_sha256') or not local_identity.is_file()):
        raise ValueError('Both reduced identities are required before returning any fold')
    if remote.get('identity_sha256') and local_identity.is_file():
        if hashlib.sha256(local_identity.read_bytes()).hexdigest() != remote['identity_sha256']:
            raise ValueError('Laptop reduced protocol/source identity differs')


def transfer(host, relative, source, destination, output):
    destination.mkdir(parents=True, exist_ok=True)
    command = ["rsync", "-a", "--checksum", "--partial", "--timeout=180",
               "--exclude=inputs.pkl", "--exclude=*.npy", "--exclude=*.log",
               "-e", "ssh -o BatchMode=yes -o ConnectTimeout=15",
               f"{host}:PRC/{relative}/{source}/", str(destination) + "/"]
    with output.open("ab") as stream:
        done = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if done.returncode:
        raise RuntimeError(f"Fold transfer failed (rsync exit {done.returncode}); output saved, not read")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument("--host", default="glaptop-wsl")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    out = args.out.resolve()
    relative = out.relative_to(ROOT)
    dest = out / "reduced_three"
    dest.mkdir(parents=True, exist_ok=True)
    state_path = dest / "collection.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"returned": []}
    state.update(pid=os.getpid(), host=args.host, status="waiting", logs_read=False,
                 original_full_input_pickles="retained on producing host; compact evaluation packets returned")
    with (dest / ".collector.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            state["checked_unix"] = time.time()
            try:
                remote = remote_state(args.host, relative)
                local_identity = dest / "identity.json"
                check_identity(remote, local_identity)
                for month in remote["ready"]:
                    if month in state["returned"]:
                        continue
                    target = out / 'folds' / month
                    in_progress = state.setdefault('in_progress', {})
                    if month not in in_progress:
                        if target.exists() and any(target.iterdir()):
                            raise ValueError(f'Refusing to overwrite an existing local fold: {month}')
                        in_progress[month] = remote['bindings'][month]
                        save(state_path, state)
                    if in_progress[month] != remote['bindings'][month]:
                        raise ValueError(f'Remote input binding changed during collection: {month}')
                    transfer(args.host, relative, f"folds/{month}", out / "folds" / month,
                             dest / "transfer.log")
                    if hashlib.sha256((target / 'input_binding.json').read_bytes()).hexdigest() != in_progress[month]:
                        raise ValueError(f'Returned input binding differs: {month}')
                    transfer(args.host, relative, f"reduced_three/folds/{month}", dest / "folds" / month,
                             dest / "transfer.log")
                    state["returned"].append(month)
                    in_progress.pop(month)
                    save(state_path, state)
                ready = all((dest / "folds" / m / "predictions.json").exists() and
                            (dest / "folds" / m / "evaluation.json").exists() for m in MONTHS)
                if ready and unlocked(out / ".lock"):
                    state["status"] = "reporting"
                    save(state_path, state)
                    with (dest / "report.log").open("ab") as stream:
                        result = subprocess.run([sys.executable, str(ROOT / "tools/final_selection/reduced_three.py"),
                                                 "report", "--out", str(out)], cwd=ROOT,
                                                stdout=stream, stderr=subprocess.STDOUT)
                    state.update(status="complete" if result.returncode == 0 else "report_failed",
                                 report_exit_code=result.returncode, finished_unix=time.time())
                    save(state_path, state)
                    return result.returncode
                state["status"] = "waiting"
                state.pop("transient_error", None)
            except (subprocess.TimeoutExpired, RuntimeError) as error:
                state["transient_error"] = str(error)
            except Exception as error:
                state.update(status="failed", error=f"{type(error).__name__}: {error}")
                save(state_path, state)
                raise
            save(state_path, state)
            time.sleep(max(30, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
