"""Separate predeclared coefficient-one flight-ID clock delta transport."""
import argparse
import types

from threadpoolctl import threadpool_limits
import target205_delta_transport as parent

OUT=parent.ROOT/'VALIDATION/runs/v3_20261012_flight_id_clock_delta_transport'
EXPERIMENTS={'flight_id_clock':dict(directory='v3_20261012_flight_id_clock_lgb',
    control='control',arms=['candidate'])}
DESIGN={**parent.DESIGN,'experiments':EXPERIMENTS}


def implementation(name):
    source=getattr(parent,name);namespace=dict(source.__globals__)
    namespace.update(OUT=OUT,EXPERIMENTS=EXPERIMENTS,DESIGN=DESIGN,__file__=__file__)
    fn=types.FunctionType(source.__code__,namespace,'flight_id_clock_'+name,source.__defaults__,source.__closure__)
    assert fn.__code__ is source.__code__ and parent.EXPERIMENTS is not EXPERIMENTS
    return fn


def freeze():
    implementation('freeze')()
    path=OUT/'design.json';value=parent.read(path)
    value['sources'] += [parent.artifact(parent.__file__),parent.artifact(parent.OUT/'design.json'),
        parent.artifact(parent.ROOT/'VALIDATION/flight_id_clock_lgb.py'),
        parent.artifact(parent.ROOT/'VALIDATION/runs/v3_20261012_flight_id_clock_lgb/identity.json')]
    value['implementation']='unchanged frozen transport bytecode in isolated namespace'
    parent.write_json(path,value)


def score():
    source=parent.ROOT/'VALIDATION/runs'/EXPERIMENTS['flight_id_clock']['directory']
    assert parent.artifact(source/'scored.parquet') in parent.read(source/'result.json')['artifacts']
    return implementation('score')('flight_id_clock')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','score'])
    with threadpool_limits(1):globals()[parser.parse_args().stage]()
