"""Exercise the complete Rome daily-taxi scorer using retained control models.

This is a zero-change arithmetic check before fitting, not a candidate result.
"""
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'VALIDATION'))
import rome_daily_taxi_jan_jul as run
from records import write_json


def main():
    run.guard()
    out = run.OUT/'control_replay'
    assert not out.exists(), 'Preserve completed control replay'
    out.mkdir()

    def retained(month, seed):
        record = run.read(run.cat.OUT/'predictions'/f'{month}_s{seed}.json')
        run.verify(record['prediction']); run.verify(record['model'])
        return record

    namespace = {**run.score.__globals__, 'OUT': out, 'completed': retained,
        'DESIGN': dict(kind='zero-change control replay only', experiment=run.art(run.MANIFEST))}
    fn = types.FunctionType(run.score.__code__, namespace, 'control_replay',
                            run.score.__defaults__, run.score.__closure__)
    fn()
    f = pd.read_parquet(out/'scored.parquet')
    np.testing.assert_array_equal(f.oldcat, f.newcat)
    for parent in ['v76', 'refined', 'quality']:
        np.testing.assert_array_equal(f[parent], f[parent+'_dailycat'])
    result = dict(status='pass', rows=len(f), all_three_controls_exact=True,
        source=run.art(__file__), experiment=run.art(run.MANIFEST),
        replay_result=run.art(out/'result.json'), no_new_fit=True, not_a_candidate=True)
    write_json(out/'audit.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    with threadpool_limits(2):
        main()
