"""Separate, fixed coefficient-one transport; frozen phase/TabM registry untouched."""
import argparse
from pathlib import Path
import types

from threadpoolctl import threadpool_limits
import target205_delta_transport as parent

OUT=parent.ROOT/'VALIDATION/runs/v3_20261012_cohort_delta_transport'
EXPERIMENTS={'cohort':dict(directory='v3_20261012_nn_public_cohort_stack',
    control='selected',arms=['replace_cohort'])}
DESIGN={**parent.DESIGN,'experiments':EXPERIMENTS}


def implementation(name):
    source=getattr(parent,name)
    namespace=dict(source.__globals__)
    namespace.update(OUT=OUT,EXPERIMENTS=EXPERIMENTS,DESIGN=DESIGN,__file__=__file__)
    fn=types.FunctionType(source.__code__,namespace,'cohort_'+name,
        source.__defaults__,source.__closure__)
    assert fn.__code__ is source.__code__ and parent.EXPERIMENTS is not EXPERIMENTS
    return fn


def freeze():
    implementation('freeze')()
    path=OUT/'design.json';value=parent.read(path)
    value['sources'] += [parent.artifact(parent.__file__),parent.artifact(parent.OUT/'design.json')]
    value['implementation']='unchanged frozen transport/freeze/score bytecode in isolated namespace'
    parent.write_json(path,value)


def score():
    source=parent.ROOT/'VALIDATION/runs'/EXPERIMENTS['cohort']['directory']
    assert parent.read(source/'result.json')['scored']==parent.artifact(source/'scored.parquet')
    return implementation('score')('cohort')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['freeze','score'])
    with threadpool_limits(1):globals()[p.parse_args().stage]()
