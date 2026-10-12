"""Fixed coefficient-one Rome TabM deltas on v79; frozen before classifier fits."""
import argparse
import types
from threadpoolctl import threadpool_limits
import target205_delta_transport as parent

OUT=parent.ROOT/'VALIDATION/runs/v3_20261012_rome_tabm_tracks_delta_transport'
EXPERIMENTS={'rome_tabm_tracks':dict(directory='v3_20261012_rome_tabm_tracks_stack',
    control='control',arms=['tabular','physical'])}
DESIGN={**parent.DESIGN,'experiments':EXPERIMENTS}


def implementation(name):
    original=getattr(parent,name)
    namespace={**original.__globals__,'OUT':OUT,'EXPERIMENTS':EXPERIMENTS,'DESIGN':DESIGN,'__file__':__file__}
    return types.FunctionType(original.__code__,namespace,name,original.__defaults__,original.__closure__)


def freeze():
    implementation('freeze')()
    path=OUT/'design.json';value=parent.read(path)
    value['sources'] += [parent.artifact(parent.__file__),parent.artifact(parent.OUT/'design.json')]
    value['implementation']='unchanged transport bytecode in isolated namespace'
    parent.write_json(path,value)


def score():
    implementation('score')('rome_tabm_tracks')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','score'])
    with threadpool_limits(1):globals()[parser.parse_args().stage]()
