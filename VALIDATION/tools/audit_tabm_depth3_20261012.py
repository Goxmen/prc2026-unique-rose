"""Independent deeper-TabM contracts, SQL transport, and paired-day arithmetic."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import duckdb
import numpy as np
import torch
import nn_public_tabm_depth3 as candidate

ROOT, RUN = candidate.ROOT, candidate.OUT
STACK = RUN / 'comparison'
read, art, verify, write = candidate.read, candidate.art, candidate.verify, candidate.write


def main():
    assert not (STACK / 'independent_audit.json').exists()
    identity = read(RUN / 'identity.json')
    result = read(STACK / 'result.json')
    frozen = read(STACK / 'identity.json')
    assert result['identity'] == art(STACK / 'identity.json')
    assert identity['design'] == candidate.DESIGN and frozen['before_fitting']
    assert read(RUN / 'audit.json')['status'] == read(RUN / 'return_audit.json')['status'] == 'pass'
    assert read(STACK / 'component/audit.json')['status'] == 'pass'
    for a in [*identity['files'], *frozen['sources'], result['scored']]:
        verify(a)
    members = []
    for month, epochs in [('2025-01', 21), ('2025-07', 17)]:
        path = RUN / f'k0/networks/{month}_tabm.json'
        r = read(path)
        assert r['identity'] == art(RUN / 'identity.json') and r['design'] == identity['design']
        assert r['seed'] == 42 and r['member'] == 0 and r['month'] == month and r['epochs'] == epochs
        assert r['all_held_reloaded_exact'] and r['inputs'] == identity['folds'][month]
        assert r['preprocessing'] == identity['folds'][month]['preprocessing']
        assert {k: r[k] for k in ['host', 'gpu', 'torch']} == identity['execution']
        for key in ['model', 'prediction', 'preprocessing', 'checkpoint']:
            verify(r[key])
        model = torch.load(verify(r['model']), map_location='cpu', weights_only=False)
        assert model['identity'] == art(RUN / 'identity.json') and model['design'] == identity['design']
        assert model['curve'] == r['curve']
        assert [a['epoch'] for a in model['curve']] == list(range(1, epochs + 1))
        assert all(torch.isfinite(t).all() for t in model['state_dict'].values())
        old = torch.load(verify(identity['reused'][month]['model']), map_location='meta', weights_only=False)
        a, b = model['state_dict'], old['state_dict']
        assert set(b).issubset(a)
        assert set(a) - set(b) == {'backbone.blocks.2.0.' + k for k in ['weight', 'r', 's', 'bias']}
        assert all(a[k].shape == b[k].shape for k in b)
        assert model['design']['architecture']['n_blocks'] == 3
        members.append(art(path))
        del old, model
    rng = np.random.default_rng(0)
    draws = np.zeros(2000)
    with duckdb.connect(config={'threads': 1, 'memory_limit': '256MB'}) as con:
        paths = [('f', STACK / 'scored.parquet'), ('d', STACK / 'component/scored.parquet'),
                 ('o', candidate.parent.OUT.parent / 'v3_20261012_nn_public_tabm_pool3_stack/scored.parquet'),
                 ('c', candidate.parent.OUT.parent / 'v3_20261012_tabm_pool3_other_runway_combo/scored.parquet')]
        for name, path in paths:
            con.read_parquet(str(path)).create_view(name)
            assert con.execute(f'select count(*),count(distinct mvt_id) from {name}').fetchone() == (344419, 344419)
        assert con.execute('''select count(*) from f full join d using(mvt_id) full join o using(mvt_id)
            full join c using(mvt_id) where f.mvt_id is null or d.mvt_id is null or o.mvt_id is null or c.mvt_id is null
            or f.y is distinct from o.y or f.y is distinct from d.y or f.y is distinct from c.y
            or f.month is distinct from c.month or f.day is distinct from c.day
            or f.control is distinct from c.three_tabm_runway
            or f.component_delta is distinct from d.three_tabm-o.three_tabm
            or f.candidate is distinct from round_even(greatest(0,c.three_tabm_runway+d.three_tabm-o.three_tabm),0)''').fetchone()[0] == 0
        for month in sorted(result['scores']):
            scores = con.execute('select sqrt(avg(pow(control-y,2))),sqrt(avg(pow(candidate-y,2))) from f where month=?', [month]).fetchone()
            np.testing.assert_allclose(scores, [result['scores'][month][c] for c in ['control', 'candidate']], atol=1e-8, rtol=0)
            days = np.asarray(con.execute('''select count(*),sum(pow(control-y,2)),sum(pow(candidate-y,2))
                from (select *,row_number() over() pos from f) where month=? group by day order by min(pos)''', [month]).fetchall())
            sample = rng.integers(0, len(days), (2000, len(days)))
            sums = days[sample].sum(1)
            draws += (np.sqrt(sums[:, 2] / sums[:, 0]) - np.sqrt(sums[:, 1] / sums[:, 0])) / 2
    ci = np.percentile(draws, [2.5, 97.5])
    np.testing.assert_allclose(ci, result['paired']['ci95'], atol=1e-8, rtol=0)
    np.testing.assert_allclose(result['cv']['candidate'] - result['cv']['control'], result['paired']['change'], atol=1e-10, rtol=0)
    gate = bool(ci[1] < 0 and all(s['candidate'] < s['control'] for s in result['scores'].values()))
    assert gate == result['gate']
    value = dict(status='pass', identity=art(RUN / 'identity.json'), result=art(STACK / 'result.json'),
                 source=art(Path(__file__)), members=members, rows=344419, independent_SQL_and_bootstrap=True,
                 architecture_changes_exactly_one_extra_block=True, complete_curve_and_all_parameter_finite=True,
                 training_reload_scope='Full held-row GPU reload verified by immutable fit receipts', gate=gate,
                 interpretation='Reused development folds; this is a fixed screen, not independent confirmation.')
    write(STACK / 'independent_audit.json', value)
    print(json.dumps(dict(status='pass', cv=result['cv'], paired=result['paired'], gate=gate)))


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
