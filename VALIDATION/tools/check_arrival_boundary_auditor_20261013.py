"""Tiny synthetic oracle comparisons; never reads a real packet or model."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'VALIDATION'))
import arrival_boundary_transfer_pilot_20261013 as producer


def main():
    spec = importlib.util.spec_from_file_location('independent_arrival_audit', Path(__file__).with_name(
        'audit_arrival_boundary_transfer_20261013.py'))
    audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
    errors = []
    for delta in [1, 1000, 30000, 120000, 120001]:
        for stand in [(45., 7.), (60., 40.)]:
            raw = np.column_stack([np.arange(7)*delta+1700000000000,
                np.array([45000000, 45000001, 45000001, 45000002, 45000003, 45000003, 45000004]),
                np.array([7000000, 7000000, 7000001, 7000001, 7000001, 7000002, 7000002]),
                np.array([-123, -123, 100, -123, -123, -123, -123]), [0, 10, 20, 30, 40, 50, 60]])
            for phase in ['DEP', 'ARR']:
                a, ta = audit.independent_features(raw, *stand, 'LIRF', phase)
                b, tb = producer.point_features(raw, raw[-1, 0], *stand, 'LIRF', phase)
                np.testing.assert_array_equal(ta, tb)
                np.testing.assert_allclose(a, b, rtol=0, atol=2e-11)
                errors.append(float(np.max(np.abs(a-b))))
    fixture = pd.DataFrame(dict(day=np.repeat(np.arange(6), np.arange(1, 7))))
    fixture['control'] = 1.2+np.arange(len(fixture))*.003
    fixture['arrival'] = fixture.control-.2+np.arange(len(fixture))*.0002
    fixture['motion'] = fixture.control-.1-np.arange(len(fixture))*.0001
    c = duckdb.connect(config={'threads': 1}); c.register('losses', fixture)
    daily = c.execute('''SELECT day,count(*) n,sum(control) control_sum,sum(arrival) arrival_sum,
        sum(motion) motion_sum FROM losses GROUP BY day ORDER BY day''').df()
    draws = np.random.default_rng(20261013).integers(0, 6, (2000, 6))
    old = producer.paired_bootstrap(fixture)
    for arm in ['arrival', 'motion']:
        diff = (daily[arm+'_sum']-daily.control_sum).to_numpy(); count = daily.n.to_numpy()
        expected = np.array([np.bincount(draw, minlength=6)@diff/(np.bincount(draw, minlength=6)@count) for draw in draws])
        np.testing.assert_allclose(expected, old[arm], rtol=0, atol=2e-11)
    c.close()
    fixture['direction'] = 'A'; fixture['airport'] = 'LIRF'
    grouped = fixture.groupby(['direction', 'airport'])[['control', 'arrival', 'motion']].agg(['mean', 'size'])
    restored = pd.read_csv(io.StringIO(grouped.to_csv()), header=[0, 1], index_col=[0, 1])
    pd.testing.assert_frame_equal(grouped, restored, check_dtype=False, check_exact=False, atol=2e-11, rtol=0)
    out = ROOT/'VALIDATION/reports/privileged_learning_frontier_20261013/independent_auditor_fixture.json'
    value = dict(status='pass', auditor=audit.art(audit.__file__), producer=audit.art(producer.__file__),
        fixture_source=audit.art(__file__), feature_cases=len(errors), maximum_feature_difference=max(errors),
        millisecond_tangential_and_far_stand_cases=True, SQL_weighted_bootstrap2000=True,
        airport_CSV_roundtrip=True, no_real_data=True, no_fit=True)
    with out.open('x') as f: json.dump(value, f, indent=2); f.write('\n')
    print(json.dumps(value))


if __name__ == '__main__': main()
