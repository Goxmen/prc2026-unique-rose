"""Separate predeclared coefficient-one robust-runway delta transport."""
import argparse
import types

from threadpoolctl import threadpool_limits
import target205_delta_transport as parent

OUT=parent.ROOT/'VALIDATION/runs/v3_20261012_other_runway_delta_transport'
EXPERIMENTS={'other_runway':dict(directory='v3_20261012_joint_public_other_runway',
    control='control',arms=['candidate'])}
DESIGN={**parent.DESIGN,'experiments':EXPERIMENTS}


def implementation(name):
    source=getattr(parent,name);namespace=dict(source.__globals__)
    namespace.update(OUT=OUT,EXPERIMENTS=EXPERIMENTS,DESIGN=DESIGN,__file__=__file__)
    fn=types.FunctionType(source.__code__,namespace,'other_runway_'+name,source.__defaults__,source.__closure__)
    assert fn.__code__ is source.__code__ and parent.EXPERIMENTS is not EXPERIMENTS
    return fn


def freeze():
    implementation('freeze')()
    path=OUT/'design.json';value=parent.read(path)
    value['sources'] += [parent.artifact(parent.__file__),parent.artifact(parent.OUT/'design.json')]
    value['implementation']='unchanged frozen transport bytecode in isolated namespace'
    parent.write_json(path,value)


def score():
    source=parent.ROOT/'VALIDATION/runs'/EXPERIMENTS['other_runway']['directory']
    assert parent.read(source/'result.json')['scored']==parent.artifact(source/'scored.parquet')
    return implementation('score')('other_runway')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','score'])
    with threadpool_limits(1):globals()[parser.parse_args().stage]()
