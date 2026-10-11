"""Fixed recurrent track screen using the retained training/scoring runner."""
import argparse
import types

import torch
from threadpoolctl import threadpool_limits

import nn_public_early_jan_jul as parent
import nn_public_recurrent as model_type

ROOT = parent.ROOT
OUT = ROOT / 'VALIDATION/runs/v3_20261011_nn_public_recurrent'
MANIFEST = OUT / 'identity.json'
DESIGN = dict(parent.DESIGN, change=model_type.SETTINGS['change'], model=model_type.SETTINGS,
              shared_runner=parent.art(parent.__file__))

scope = dict(parent.__dict__, __file__=__file__, OUT=OUT, MANIFEST=MANIFEST,
             DESIGN=DESIGN, model_type=model_type)
NAMES = ['paths','completed','control','training_function','guard','fit','audit_models','score','audit_scores']
for name in NAMES:
    original = getattr(parent, name)
    scope[name] = types.FunctionType(original.__code__, scope, original.__name__,
                                    original.__defaults__, original.__closure__)
    assert scope[name].__code__ is original.__code__
globals().update({name: scope[name] for name in NAMES})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['guard','fit','audit_models','score','audit_scores'])
    args = parser.parse_args()
    threads = 2 if args.stage in ['fit','audit_models'] else 1
    torch.set_num_threads(threads)
    with threadpool_limits(threads): globals()[args.stage]()
