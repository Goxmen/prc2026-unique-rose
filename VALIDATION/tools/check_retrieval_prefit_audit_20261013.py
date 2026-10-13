"""Tiny fabricated-data checks for the independently authored retrieval audit."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'VALIDATION'))
import retrieval_frontier_20261013 as producer
import retrieval_label_model as model

spec = importlib.util.spec_from_file_location('independent', Path(__file__).with_name('audit_retrieval_frontier_prefit_20261013.py'))
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)


def main():
    torch.set_num_threads(1)
    raw = pd.DataFrame({c: [None]*4 for c in producer.RAW})
    raw['MVT_ID_mvt'] = [100, 200, 300, 400]; raw['PHASE_mvt'] = 'DEP'
    raw['MVT_TIME_UTC_mvt'] = pd.to_datetime(['2025-04-01T00:00:00Z', '2025-04-02T00:00:00Z',
                                           '2025-03-31T23:59:59Z', '2025-04-15T12:00:00Z'])
    for name in producer.RAW:
        if name.endswith(('_flt', '_mvt')) and name not in ['MVT_TIME_UTC_mvt', 'MVT_ID_mvt', 'PHASE_mvt']:
            raw[name] = ['X', None, 'Y', '']
    for c in audit.CLOCKS.values():
        raw[c] = raw.MVT_TIME_UTC_mvt-pd.to_timedelta([5, 60, 3600, -5], unit='s')
        raw.loc[1, c] = pd.NaT
    values = np.arange(60, dtype=np.float32).reshape(4, 15)
    expected = producer.features(raw, values); actual = audit.make_features(raw, values)
    pd.testing.assert_frame_equal(actual, expected)
    # Explicit timestamp units and missingness are not inferred from pandas storage resolution.
    ms = audit.stamps(raw.MVT_TIME_UTC_mvt)
    us = raw.MVT_TIME_UTC_mvt.astype('datetime64[us, UTC]')
    np.testing.assert_array_equal(ms, audit.stamps(us))
    q = np.array([[999, 86460000, 1, 3, 4, 11, 77]], np.int64)
    d = np.array([[998, 86340000, 1, 8, 9, 11, 0], [997, 86339999, 1, 8, 9, 11, 0],
                  [996, 86460000, 1, 3, 4, 12, 0], [995, 172860000, 2, 3, 4, 11, 0],
                  [994, 259260000, 1, 8, 9, 99, 77]], np.int64)
    qc, dc = np.array([17.]), np.array([19., 19., 17., 17., 20.])
    oracle = model.admissible(torch.tensor(q), torch.tensor(qc), torch.tensor(d), torch.tensor(dc)).numpy()
    np.testing.assert_array_equal(oracle, [[False, True, False, False, False]])
    np.testing.assert_array_equal(audit.donor_counts(q, qc, d, dc), oracle.sum(1))
    combined = np.vstack([d[:, :6], q[:, :6]])
    merged_clock = np.r_[dc, qc]
    a, b = audit.purge(combined, merged_clock, np.arange(5), np.array([5]))
    pa, pb = producer.alias_purge(combined, merged_clock, np.arange(5), np.array([5]))
    np.testing.assert_array_equal(a, pa); np.testing.assert_array_equal(b, pb)
    for cap in [1, 2, 9]:
        np.testing.assert_array_equal(audit.sample(combined, np.arange(5), cap, 'bank42'),
                                      producer.hash_select(combined, np.arange(5), cap, 'bank42'))
    # Check the saved optimizer topology assumed by prescore, on fabricated tensors only.
    torch.manual_seed(42); net = model.Retrieval([3], 2)
    opt = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.0003)
    bank_cat = torch.zeros(20, 1, dtype=torch.int64); bank_num = torch.zeros(20, 2)
    loss = net(bank_cat[:2], bank_num[:2], bank_cat, bank_num, torch.ones(20),
               torch.ones(2, 20, dtype=torch.bool), False).square().sum()
    loss.backward(); opt.step()
    assert len(opt.state_dict()['state']) == len(net.state_dict())
    assert torch.count_nonzero(net.state_dict()['label.weight']) == 0
    report = dict(status='pass', source=audit.art(__file__), auditor=audit.art(audit.__file__),
        producer=audit.art(producer.__file__), model=audit.art(model.__file__),
        checks=['37 fabricated features exact', 'microsecond/millisecond timestamp parity',
                'independent keyed alias purge', 'independent per-airport donor counts',
                'fixed hash selection and canonical ordering', 'checkpoint optimizer topology'],
        real_input_arrays_loaded=False, real_model_fits=False)
    audit.save(ROOT/'VALIDATION/reports/retrieval_frontier_20261013/independent_prefit_fixture.json', report)
    print(json.dumps(dict(status='pass', checks=len(report['checks']))))


if __name__ == '__main__': main()
