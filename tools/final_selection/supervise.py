"""Save process status without inspecting a worker's log contents."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('A child command is required')
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    if args.receipt.exists():
        raise FileExistsError('Preserve previous launch receipts; use a distinct receipt name')
    state = dict(supervisor_pid=os.getpid(), command=command, cwd=str(Path.cwd()),
                 started_utc=datetime.now(timezone.utc).isoformat(), logs_read=False)

    def save():
        temp = args.receipt.with_suffix('.json.tmp')
        temp.write_text(json.dumps(state, indent=2) + '\n')
        temp.replace(args.receipt)

    save()
    process = subprocess.Popen(command)
    state.update(worker_pid=process.pid, status='running')
    save()
    state.update(returncode=process.wait(), status='finished',
                 finished_utc=datetime.now(timezone.utc).isoformat())
    save()
    return state['returncode']


if __name__ == '__main__':
    raise SystemExit(main())
