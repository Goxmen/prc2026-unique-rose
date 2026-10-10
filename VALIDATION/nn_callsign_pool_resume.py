"""Resume only callsign PLE k1/k2, retaining the original saved preprocessing.

PyTorch tensor storage identifiers make otherwise identical preprocessing pickle
bytes differ between processes. This wrapper changes only the in-process save
guard: every regenerated value must exactly match the saved object, and the old
file remains untouched. The original feature/training helper sources stay byte
identical, so their existing model and prediction provenance remains valid.

The saved group42 k0 and frozen parent epochs21/17 are reused. No inner fit,
feature change, target change, or new stopping decision is permitted.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile

os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'VALIDATION'),str(ROOT/'pipeline'),str(ROOT/'VALIDATION/tools')]
import nn_callsign_clocks_jan_jul as runner
from threadpoolctl import threadpool_limits
from records import sha256,write_json


def equal_preprocessing(saved,current,where='preprocessing'):
    """Reject any type, ordering, value, dtype, shape, or encoder state change."""
    if type(saved) is not type(current):
        raise ValueError(f'{where}: type differs')
    if isinstance(saved,torch.Tensor):
        if (saved.dtype!=current.dtype or saved.device!=current.device
                or saved.shape!=current.shape or not torch.equal(saved,current)):
            raise ValueError(f'{where}: tensor differs')
    elif isinstance(saved,np.ndarray):
        try:
            np.testing.assert_array_equal(saved,current,strict=True)
        except AssertionError as error:
            raise ValueError(f'{where}: array differs') from error
    elif isinstance(saved,dict):
        if list(saved)!=list(current):
            raise ValueError(f'{where}: dictionary keys/order differ')
        for key in saved:
            equal_preprocessing(saved[key],current[key],f'{where}.{key}')
    elif isinstance(saved,(list,tuple)):
        if len(saved)!=len(current):
            raise ValueError(f'{where}: length differs')
        for i,(a,b) in enumerate(zip(saved,current)):
            equal_preprocessing(a,b,f'{where}[{i}]')
    elif isinstance(saved,runner.shared.nnj.Encoder):
        equal_preprocessing(vars(saved),vars(current),where+'.encoder_state')
    elif isinstance(saved,(str,int,float,bool,type(None),np.generic)):
        same=saved==current
        if isinstance(saved,(float,np.floating)) and np.isnan(saved) and np.isnan(current):
            same=True
        if not same:
            raise ValueError(f'{where}: scalar differs')
    else:
        raise TypeError(f'{where}: unsupported preprocessing value {type(saved)}')


def retain_saved_preprocessing(path,current):
    path=Path(path)
    if not path.exists():
        raise ValueError('Continuation requires existing audited preprocessing')
    before=sha256(path)
    with path.open('rb') as stream:
        saved=pickle.load(stream)
    equal_preprocessing(saved,current)
    if sha256(path)!=before:
        raise ValueError('Original preprocessing changed during equality check')


def provenance():
    runner.guard_run()
    report=dict(purpose='Expand existing callsign group42 from k0 to k0/k1/k2',
        only_runtime_change='atomic_pickle validates exact semantic equality and retains original bytes',
        wrapper_sha256=sha256(Path(__file__)),
        callsign_runner_sha256=sha256(Path(runner.__file__)),
        code_identity_sha256=sha256(runner.OUT/'code_identity.json'),
        groups=[42],nets=[0,1,2],pool_size=3,
        original_design_preserved=True,original_single_networks_not_refitted=True,
        epoch_policy='Frozen original N09 parent stop21/17; no inner fit',
        exploratory_reused_development_months=True,months={})
    for month in runner.ts.SPLITS:
        runner.validate_completed_month(month)
        record=runner.ts.read(runner.OUT/'networks'/f'{month}_s42.json')
        if record['stop_epoch']!=runner.STOP_EPOCHS[month] or '0' not in record['networks']:
            raise ValueError('Expected original k0 fit and frozen parent stopping epoch')
        report['months'][month]=dict(stop_epoch=runner.STOP_EPOCHS[month],
            preprocessing_sha256=sha256(runner.OUT/'preprocessing'/f'{month}.pkl'),
            original_k0=record['networks']['0'])
    dest=runner.OUT/'callsign_pool_continuation.json'
    if dest.exists():
        if runner.ts.read(dest)!=report:
            raise ValueError('Continuation implementation or original artifacts changed')
    else:
        write_json(dest,report)
    return report


def self_check():
    # A fresh tensor pickle can differ bytewise; object values must still match.
    path=runner.OUT/'preprocessing'/'2025-01.pkl'
    original=path.read_bytes()
    saved=pickle.loads(original)
    regenerated=pickle.loads(pickle.dumps(saved,protocol=pickle.HIGHEST_PROTOCOL))
    equal_preprocessing(saved,regenerated)
    with tempfile.TemporaryDirectory(prefix='prc_callsign_preprocessing_') as temp:
        clone=Path(temp)/'preprocessing.pkl'
        clone.write_bytes(original)
        retain_saved_preprocessing(clone,regenerated)
        if clone.read_bytes()!=original:
            raise AssertionError('Exact original pickle bytes must remain unchanged')
    for field in ['bins','encoder','mu','columns']:
        bad=copy.deepcopy(saved)
        if field=='bins': bad['bins'][0][0]+=1
        elif field=='encoder':
            name=next(iter(bad['encoder'].q))
            bad['encoder'].q[name][0][0]+=1
        elif field=='mu': bad['mu']+=1
        else: bad['columns'][0]+='_changed'
        try:
            equal_preprocessing(saved,bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Changed {field} was accepted')
    runner.shared.log('Continuation guard accepts identical tensor values and rejects changed bins, encoder, target normalization and columns')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['check','fit','score'])
    args=parser.parse_args()
    torch.set_num_threads(2)
    self_check()
    if args.stage=='check':
        return
    provenance()
    if args.stage=='score':
        with threadpool_limits(limits=1,user_api='blas'):
            runner.score([42],[0,1,2])
        return
    original=runner.shared.atomic_pickle
    runner.shared.atomic_pickle=retain_saved_preprocessing
    try:
        with threadpool_limits(limits=1,user_api='blas'):
            runner.fit([42],[0,1,2],runner.ts.SPLITS)
    finally:
        runner.shared.atomic_pickle=original


if __name__=='__main__':
    main()
