"""Small laptop-only exact training and full-width inference execution audit."""
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'VALIDATION')]
import fixed_epoch_fast_train as fast
import callsign_submit as parent
from VALIDATION.tests.test_fixed_epoch_fast_train import check_training


def main():
    torch.set_num_threads(2)
    checks=[]
    for device in ['cpu','cuda']:
        started=time.monotonic()
        data,model,record=check_training(device)
        record['seconds']=time.monotonic()-started
        checks.append(record)
        del data,model
        torch.cuda.empty_cache()
    prep=pd.read_pickle(parent.OUT/'preprocessing/full_year.pkl')
    # A full-width production-schema model, fixed weights, 65,537 examples,
    # and real fitted PLE bins. This deliberately crosses both chunk sizes.
    torch.manual_seed(811)
    rng=np.random.default_rng(812)
    n=65537
    cat=np.column_stack([rng.integers(0,card,n,dtype=np.int64) for card in prep['cards']])
    g=np.column_stack([rng.uniform(float(b[0]),float(b[-1]),n).astype(np.float32) for b in prep['bins']])
    plain=rng.normal(0,.5,(n,len(prep['plain']))).astype(np.float32)
    inputs=tuple(torch.from_numpy(a).to('cuda') for a in [cat,g,plain])
    model=fast.shared.nni.PleNet(prep['cards'],prep['bins'],len(prep['plain'])).to('cuda')
    offset=np.zeros(n,float)
    output={}
    for chunk in [65536,16384]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started=time.monotonic()
        pred=fast.shared.predict(model,inputs,prep['sd'],prep['mu'],offset,chunk=chunk)
        torch.cuda.synchronize()
        output[chunk]=dict(pred=pred,seconds=time.monotonic()-started,peak_bytes=torch.cuda.max_memory_allocated())
    delta=output[65536]['pred']-output[16384]['pred']
    np.testing.assert_allclose(output[65536]['pred'],output[16384]['pred'],rtol=0,atol=.01)
    timing={str(k):{name:value for name,value in rec.items() if name!='pred'} for k,rec in output.items()}
    record=dict(source=parent.artifact(Path(__file__)),trainer=parent.artifact(Path(fast.__file__)),
        test=parent.artifact(ROOT/'VALIDATION/tests/test_fixed_epoch_fast_train.py'),
        preprocessing=parent.artifact(parent.OUT/'preprocessing/full_year.pkl'),
        checks=checks,inference_rows=n,inference_timing=timing,
        inference_max_abs_sec=float(np.abs(delta).max()),inference_mean_abs_sec=float(np.abs(delta).mean()),
        device=torch.cuda.get_device_name(),passed=True,
        caveat='Bounded timing probe does not include4.88GiB persistent production tensors; actual first epoch timed separately')
    parent.write_json(parent.OUT/'portable_worker/fast_execution_parity.json',record)
    print(record,flush=True)


if __name__=='__main__':
    with threadpool_limits(2):
        main()
