"""Finish the two retained-model inference parts after the main tree exits."""
import hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'VALIDATION/runs/v3_20261012_final_public_tree_inference'
RUNNER=ROOT/'VALIDATION/final_public_tree_inference.py'


def write(value):
    p=OUT/'execution.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(value,indent=2)+'\n');q.replace(p)


def main():
    assert not (OUT/'execution.json').exists()
    identity=json.loads((OUT/'identity.json').read_text())
    wanted=next(a['sha256'] for a in identity['sources'] if a['path']==str(RUNNER.relative_to(ROOT)))
    record=dict(state='waiting_for_main',steps=[],source_sha256=wanted,no_fit=True,no_upload=True)
    write(record);deadline=time.monotonic()+1800
    try:
        while not (OUT/'main/final.json').exists():
            if time.monotonic()>deadline:raise TimeoutError('Main inference receipt absent after30minutes')
            time.sleep(10)
        # The running main process has already finished its expensive inference
        # before atomically recording final.json. No heavy operations follow it.
        for part in ['airport','rome_cat']:
            assert hashlib.sha256(RUNNER.read_bytes()).hexdigest()==wanted
            record.update(state='running',active_part=part);write(record)
            with (OUT/f'{part}.log').open('x') as stream:
                subprocess.run([str(ROOT/'.venv/bin/python'),str(RUNNER),part],cwd=ROOT,
                    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4'),
                    stdout=stream,stderr=subprocess.STDOUT,check=True)
            receipt=json.loads((OUT/part/'final.json').read_text());assert receipt['final_reload_exact']
            record['steps'].append(dict(part=part,status='complete',rows=receipt['rows']));write(record)
        record.update(state='complete',active_part=None);write(record)
    except BaseException as e:
        record.update(state='failed',failure=repr(e));write(record);raise


if __name__=='__main__':main()
