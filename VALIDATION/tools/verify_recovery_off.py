"""Prove current recovery-off matrices equal a frozen control's source, without fitting."""

import argparse
import glob
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import modeling
from harness import load
from common import connect, TRAIN_GLOB, RANKING
from records import ROOT, read_run, sha256, write_json, data_identity


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    run, directory = read_run(args.baseline)
    if run['status'] != 'complete' or run['config'].get('recovery', 'off') != 'off':
        raise ValueError('Expected a completed recovery-off control')
    # The frozen module imports these cached modules. Verify they are unchanged.
    dependencies = ['pipeline/common.py', 'VALIDATION/features_time.py',
                    'VALIDATION/movement_data.py', 'VALIDATION/protocol.py']
    for name in dependencies:
        if sha256(ROOT/name) != run['source']['files'][name]:
            raise ValueError(f'Frozen dependency differs: {name}')
    frozen_path = directory/'source/VALIDATION/modeling.py'
    spec = importlib.util.spec_from_file_location('frozen_control_modeling', frozen_path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    if old.PARAMS != modeling.PARAMS:
        raise ValueError('Model parameters differ')
    if run['data_identity'] != data_identity([*glob.glob(TRAIN_GLOB), RANKING]):
        raise ValueError('Input datasets differ from the frozen control')
    con = connect()
    con.execute('SET threads=2')
    dep, arr = load(con, 1.)
    con.close()
    cfg = run['config']
    result = {'control': run['run_id'],
              'frozen_modeling_sha256': sha256(frozen_path),
              'current_modeling_sha256': sha256(ROOT/'VALIDATION/modeling.py'),
              'dependencies_verified_unchanged': dependencies, 'training': {}, 'validation': {}}
    for model_id, model in run['models'].items():
        td = modeling.period_rows(dep, model['train_start'], model['train_end'])
        ta = modeling.period_rows(arr, model['train_start'], model['train_end'])
        history = model['departure_history']
        a, ya, ca = old.training_matrix(td, ta, cfg, history)
        b, yb, cb = modeling.training_matrix(td, ta, cfg, history)
        pd.testing.assert_frame_equal(a, b, check_exact=True)
        np.testing.assert_array_equal(ya, yb)
        for column in ca:
            pd.testing.assert_index_equal(ca[column], cb[column])
        result['training'][model_id] = dict(rows=len(a), features=a.columns.tolist(),
            values_dtypes_order_categories_targets='exactly_equal')
        del a, b
        for name, fold in run['folds'].items():
            if fold['model_id'] != model_id:
                continue
            vd, va = modeling.month_rows(dep, fold['val']), modeling.month_rows(arr, fold['val'])
            a = old.validation_matrix(td, ta, vd, va, fold['freeze'], ca, cfg, history)
            b = modeling.validation_matrix(td, ta, vd, va, fold['freeze'], cb, cfg, history)
            pd.testing.assert_frame_equal(a, b, check_exact=True)
            result['validation'][name] = dict(rows=len(a), values_dtypes_order_categories='exactly_equal')
            del a, b
        write_json(args.output, result)
        print(model_id, 'training and assigned validation matrices exactly equal', flush=True)
    write_json(args.output, result)
    print(result)


if __name__ == '__main__':
    main()
