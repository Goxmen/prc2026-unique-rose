"""User-launched desktop/laptop validation and compact result collection."""
from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'VALIDATION/runs/v3_20261013_final_month_selection'


def main():
    os.chdir(ROOT)
    python = str(ROOT / '.venv/bin/python')
    runner = 'tools/final_selection/reduced_three.py'
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4',
                       MKL_NUM_THREADS='4', TORCH_DISABLE_NATIVE_JIT='1')
    # Source and data were synchronized and verified at handoff. Recheck both
    # frozen identities before either host begins heavy work.
    sys.path.insert(0, str(ROOT / 'tools/final_selection'))
    import reduced_three
    reduced_identity, _ = reduced_three.freeze(RUN)
    remote_check = "cd ~/PRC && .venv/bin/python -c " + shlex.quote(
        "import sys,json;sys.path.insert(0,'tools/final_selection');"
        "import reduced_three as r;print(json.dumps(r.freeze(r.original.DEFAULT)[0]))")
    checked = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                              'glaptop-wsl', remote_check], capture_output=True, text=True)
    import json
    if checked.returncode or json.loads(checked.stdout) != reduced_identity:
        raise RuntimeError('Laptop unavailable or source identity differs; no jobs started')
    # Local/remote run locks still protect each host from duplicate runners.
    remote_command = ('cd ~/PRC && env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 '
        'MKL_NUM_THREADS=4 TORCH_DISABLE_NATIVE_JIT=1 .venv/bin/python -u '
        + runner + ' run --threads 4 --folds 2025-02 2025-06')
    print('Desktop: January/July. Laptop: February/June. Three finalists; no uploads.', flush=True)
    remote = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=30',
                               'glaptop-wsl', remote_command])
    local = subprocess.Popen([python, '-u', runner, 'run', '--threads', '4',
                              '--folds', '2025-01', '2025-07'], env=environment)
    # The collector is lightweight and starts only under this explicit command.
    collector = subprocess.Popen([python, '-u', 'tools/final_selection/collect_remote.py'], env=environment)
    announced = set()
    while local.poll() is None or remote.poll() is None:
        for name, process in [('desktop', local), ('laptop', remote), ('collector', collector)]:
            code = process.poll()
            if code not in (None, 0) and name not in announced:
                print(f'{name} exited {code}; the other host may still be running. '
                      'Completed work remains saved.', flush=True)
                announced.add(name)
        time.sleep(2)
    local_code, remote_code = local.returncode, remote.returncode
    if local_code or remote_code:
        collector.terminate()
        collector.wait()
        print(f'Run stopped: desktop={local_code}, laptop={remote_code}. '
              'Completed artifacts retained; logs have not been inspected.', flush=True)
        return 1
    code = collector.wait()
    if code:
        print('Collection/report failed; see reduced_three/collection.json.', flush=True)
        return code
    # This is the requested score table, not a training log.
    print((RUN / 'reduced_three/scores.txt').read_text(), end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
