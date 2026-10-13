"""Small, dependency-free IO helpers shared by the manual v80 runner."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'VALIDATION/runs/v3_20261013_v80_refit_final'
PYTHON = str(ROOT / '.venv/bin/python')
DEADLINE = datetime(2026, 10, 14, 10, tzinfo=timezone.utc)

def now():
    return datetime.now(timezone.utc)

def read(path):
    return json.loads(Path(path).read_text())

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temp.replace(path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def relative(path):
    return str(Path(path).absolute().relative_to(ROOT))

def resolve(path):
    result = (ROOT / path).resolve()
    if not result.is_relative_to(ROOT):
        raise ValueError('Artifact outside repository')
    return result

def artifact(path):
    path = resolve(path)
    return dict(path=relative(path), sha256=sha(path), bytes=path.stat().st_size)

def verify(record):
    path = resolve(record['path'])
    if sha(path) != record['sha256']:
        raise ValueError('Artifact changed: ' + record['path'])
    return path

def environment(threads=4):
    return dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS=str(threads),
                MKL_NUM_THREADS=str(threads), TORCH_DISABLE_NATIVE_JIT='1')

def call_json(command, **kwargs):
    return json.loads(subprocess.check_output(command, text=True, **kwargs))

def announce(message):
    print('[' + now().astimezone().strftime('%H:%M:%S') + '] ' + message, flush=True)
