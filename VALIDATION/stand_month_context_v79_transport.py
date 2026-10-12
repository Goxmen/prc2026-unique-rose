"""Fixed coefficient-one stand-context main-tree transport onto v79."""
import argparse
import types
from threadpoolctl import threadpool_limits
import target205_delta_transport as parent

OUT=parent.ROOT/'VALIDATION/runs/v3_20261012_stand_month_context_delta_transport'
EXPERIMENTS={'stand_context':dict(directory='v3_20261012_stand_month_context_lgb',control='control',arms=['candidate'])}
DESIGN={**parent.DESIGN,'experiments':EXPERIMENTS}


def implementation(name):
    fn=getattr(parent,name)
    ns={**fn.__globals__,'OUT':OUT,'EXPERIMENTS':EXPERIMENTS,'DESIGN':DESIGN,'__file__':__file__}
    return types.FunctionType(fn.__code__,ns,name,fn.__defaults__,fn.__closure__)


def freeze():
    implementation('freeze')()
    path=OUT/'design.json';value=parent.read(path)
    value['sources']+=[parent.artifact(parent.__file__),parent.artifact(parent.OUT/'design.json')]
    parent.write_json(path,value)


def score():
    implementation('score')('stand_context')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['freeze','score'])
    with threadpool_limits(1):globals()[p.parse_args().stage]()
