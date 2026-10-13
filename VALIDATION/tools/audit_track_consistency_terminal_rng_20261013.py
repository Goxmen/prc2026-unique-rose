"""Additive postfit RNG-control evidence; no frozen source or metric changes.

Run only after the primary saved-result independent audit. Read six existing CPU
checkpoints and compare terminal CPU/CUDA/dropout and permutation streams across
the three matched arms in each month. No inference, label access or new fits.
"""
import hashlib
import json
import os
from pathlib import Path
import time
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import torch

ROOT=Path(__file__).resolve().parents[2]
TRAIN=ROOT/'VALIDATION/runs/v3_20261013_nn_public_tabm_track_consistency'
OUT=TRAIN/'comparison'
ARMS=['supervised_control','training_consistency','target_consistency']
KEYS=['cpu_rng','cuda_rng','permutation_generator']


def read(path):return json.loads(Path(path).read_text())
def art(path):
    path=Path(path);h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return dict(path=str(path.relative_to(ROOT)),sha256=h.hexdigest())
def verify(record):
    path=ROOT/record['path'];assert art(path)=={k:record[k] for k in ['path','sha256']}
    return path
def digest(value):
    assert isinstance(value,torch.Tensor) and value.device.type=='cpu' and value.dtype==torch.uint8 and value.ndim==1
    return dict(dtype=str(value.dtype),elements=value.numel(),sha256=hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest())


def compare(states):
    assert list(states)==ARMS
    base=states[ARMS[0]];result={}
    for arm in ARMS:
        assert set(states[arm])==set(KEYS)
        result[arm]={}
        for key in KEYS:
            result[arm][key]=digest(states[arm][key])
            assert torch.equal(states[arm][key],base[key]),(arm,key,'terminal RNG diverged')
    return result


def audit():
    started=time.monotonic();torch.set_num_threads(1)
    path=OUT/'terminal_rng_audit.json';assert not path.exists()
    original=read(OUT/'independent_audit.json')
    assert original['status']=='pass' and original['identity']==art(OUT/'identity.json')
    assert original['result']==art(OUT/'result.json') and len(original['members'])==6
    inputs=[];months={}
    for month in ['2025-01','2025-07']:
        states={}
        for arm in ARMS:
            receipt=TRAIN/arm/'networks'/f'{month}_s42.json'
            assert art(receipt) in original['members'];record=read(receipt)
            assert record['month']==month and record['arm']==arm and record['all_held_reloaded_exact']
            model=verify(record['model']);saved=torch.load(model,map_location='cpu',weights_only=False)
            assert saved['month']==month and saved['arm']==arm and saved['identity']==record['identity']
            terminal=saved['terminal_training_state'];assert terminal['updates']==sum(c['updates'] for c in saved['curve'])
            states[arm]={key:terminal[key].clone() for key in KEYS}
            inputs.append(dict(receipt=art(receipt),model=record['model']))
            del saved,terminal
        months[month]=compare(states)
    value=dict(status='pass',source=art(Path(__file__)),primary_audit=art(OUT/'independent_audit.json'),
        comparison_identity=art(OUT/'identity.json'),primary_result=art(OUT/'result.json'),inputs=inputs,months=months,
        all_three_arms_terminal_CPU_CUDA_permutation_streams_exact=True,additive_postfit_control_evidence=True,
        no_frozen_source_or_metrics_changed=True,no_model_inference=True,no_labels_read=True,no_fit=True,no_upload=True,
        seconds=time.monotonic()-started)
    with path.open('x') as stream:stream.write(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(status='pass',audit=art(path))))


if __name__=='__main__':audit()
