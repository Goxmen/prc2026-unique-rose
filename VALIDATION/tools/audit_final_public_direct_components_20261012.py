"""Independent direct-mixture audit; no fit, model inference or submission.

SQL reconstructs all means, probability mixtures and old component identities.
The original correction functions replay direct values independently of the
new builder's final_frame call, including all four final months.
"""
import gc
import hashlib
import json
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'VALIDATION'))
import callsign_submit as original

RUNS = ROOT / 'VALIDATION/runs'
OUT = RUNS / 'v3_20261012_final_public_direct_components'
PINNED = {
    'VALIDATION/final_public_direct_components.py': 'bf945746fd5ce3aef0e58afe6d7b5962716cab9348dd05a76b1478ff2e1e542b',
    'VALIDATION/runs/v3_20261012_final_public_direct_components/identity.json': '26716dd0429a3b910a39d8ae19d9ac52accb14d757f06beb1b429f855cbc51d2',
    'data/ranking.parquet': '9db75e3a56fa09b87aab72e191a53da3fe763d9887ad2b23aea9cde6a6325f86',
    'data/final_ranking.parquet': '4b0692a792846079516a2749e034b0abda55df72c213eabdcda52a4d1f6bc69f',
}
EXPECTED_MONTHS = {
    'old': {'2026-01': 152719, '2026-07': 192122},
    'final': {'2026-01': 152719, '2026-02': 144158, '2026-06': 181791, '2026-07': 192122},
}
OLD_PAIRS = [
    ('control', 'v76', 'control'), ('p', 'v76', 'p'), ('calls_nn', 'v76', 'calls_nn'),
    ('physical_pool', 'v76', 'physical_old'), ('ple_pool', 'v76', 'physical_new'),
    ('physical_neural', 'v76', 'neural_old'), ('ple_neural', 'v76', 'neural_new'),
    ('v76_tree', 'v76', 'tree'), ('v76_direct', 'v76', 'candidate'),
    ('airport_tree', 'airport', 'tree_new'), ('airport_direct', 'airport', 'direct_new'),
    ('cat_tree', 'cat', 'candidate_tree'), ('cat_direct', 'cat', 'candidate_direct'),
    ('main_tree', 'main', 'candidate_tree'), ('main_direct', 'main', 'candidate_direct'),
    ('tabm_direct', 'tabm', 'tabm_direct'),
]
CHECKED = {}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def art(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path))


def verify(record):
    path = (ROOT / record['path']).resolve()
    assert path.is_relative_to(ROOT)
    if record['path'] in CHECKED:
        assert CHECKED[record['path']] == record['sha256']
    else:
        assert sha(path) == record['sha256'], record['path']
        CHECKED[record['path']] = record['sha256']
    return path


def receipts(spec, view):
    """Check output bindings against completed inference receipts, not paths."""
    for family, stem in [('ple', 'public_ple'), ('tabm', 'public_tabm')]:
        directory = RUNS / ('v3_20261012_final_' + stem + '_inference')
        identity = read(directory / 'identity.json')
        receipt = read(directory / (view + '.json'))
        assert receipt['identity'] == art(directory / 'identity.json')
        assert receipt['all_old_predictions_exact']
        assert [m['seed'] for m in identity['members']] == [42, 1042, 2042]
        chosen = [m['old_prediction'] for m in identity['members']] if view == 'old' else receipt['predictions']
        assert spec[family] == chosen
        assert view == 'old' or receipt['all_final_reloads_exact']
    directory = RUNS / 'v3_20261012_final_public_tree_inference'
    identity = read(directory / 'identity.json')
    for family in ['main', 'airport', 'rome_cat']:
        receipt = read(directory / family / (view + '.json'))
        assert receipt['identity'] == art(directory / 'identity.json') and receipt['old_replay_exact']
        assert view == 'old' or receipt['final_reload_exact']
        if family == 'rome_cat':
            chosen = [identity['models']['rome_cat'][str(seed)]['prediction'] for seed in [42, 142, 242]] if view == 'old' else receipt['predictions']
            assert spec['public_rome'] == chosen
        else:
            chosen = identity['old_predictions'][family] if view == 'old' else receipt['predictions'][0]
            assert spec['public_' + family] == chosen
    directory = RUNS / 'v3_20261012_final_daily_main_inference'
    receipt = read(directory / (view + '.json'))
    assert receipt['identity'] == art(directory / 'identity.json')
    assert receipt['all_original_predictions_exact'] and receipt['all_model_reload_predictions_exact']
    assert spec['daily_main'] == receipt['prediction']
    directory = RUNS / 'v3_20261012_final_public_catboost_inference'
    receipt = read(directory / (view + '.json'))
    assert receipt['identity'] == art(directory / 'identity.json') and receipt['old_replay_exact']
    assert view == 'old' or receipt['final_reload_exact']
    chosen = read(RUNS / 'v3_20261011_catboost_full_inputs_submit/predictions/old.json')['prediction'] if view == 'old' else receipt['predictions'][0]
    assert spec['public_cat'] == chosen


def sql_checks(identity, spec, view):
    rows = sum(EXPECTED_MONTHS[view].values())
    with duckdb.connect(config={'threads': 1, 'memory_limit': '1GB', 'preserve_insertion_order': True}) as con:
        con.execute("SET TimeZone='UTC'")
        def bind(name, record):
            con.read_parquet(str(verify(record))).create_view(name)
            assert con.execute(f'select count(*),count(distinct mvt_id) from {name}').fetchone() == (rows, rows)
            assert con.execute(f'''select count(*) from {name} full join f using(mvt_id)
                where {name}.mvt_id is null or f.mvt_id is null''').fetchone()[0] == 0
        con.read_parquet(str(OUT / (view + '.parquet'))).create_view('f')
        assert con.execute('select count(*),count(distinct mvt_id) from f').fetchone() == (rows, rows)
        con.read_parquet(str(verify(spec['template']))).create_view('template')
        con.read_parquet(str(ROOT / 'data' / ('ranking.parquet' if view == 'old' else 'final_ranking.parquet'))).create_view('raw')
        assert con.execute('''select count(*) from f full join template t on f.mvt_id=t.MVT_ID_mvt
            where f.mvt_id is null or t.MVT_ID_mvt is null''').fetchone()[0] == 0
        months = dict(con.execute('''select strftime(r.MVT_TIME_UTC_mvt,'%Y-%m'),count(*)
            from f join raw r on f.mvt_id=r.MVT_ID_mvt where r.PHASE_mvt='DEP' group by 1''').fetchall())
        assert months == EXPECTED_MONTHS[view]
        assert con.execute("select count(*) from raw where PHASE_mvt='DEP' and TAXITIME_SEC_mvt is not null").fetchone()[0] == 0
        columns = [x[0] for x in con.execute('describe f').fetchall() if x[0] != 'mvt_id']
        bad = ' or '.join(f'"{k}" is null or not isfinite("{k}") or "{k}"<0' for k in columns)
        assert con.execute('select count(*) from f where ' + bad).fetchone()[0] == 0
        assert con.execute('''select count(*) from f join raw r on f.mvt_id=r.MVT_ID_mvt
            where f.p>1 or f.r20>1 or f.original_rome_cat>1
            or (r.ADEP_mvt!='LIRF' and (f.p!=0 or f.r20!=0 or f.original_rome_cat!=0))''').fetchone()[0] == 0
        bind('ref', spec['reference']); bind('oldcat', spec['original_rome_cat'])
        joins = ['join ref using(mvt_id)', 'join oldcat using(mvt_id)']
        expected = {'calls_nn': 'cast(ref.calls_nn as double)', 'original_rome_cat': 'cast(oldcat.p as double)',
            'r20': 'least(1.,greatest(0.,2*cast(ref.p as double)-cast(oldcat.p as double)))'}
        for family in ['physical', 'ple', 'tabm', 'public_rome']:
            assert len(spec[family]) == 3
            for k, record in enumerate(spec[family]):
                name = family + str(k); bind(name, record); joins.append('join ' + name + ' using(mvt_id)')
            col = 'p' if family == 'public_rome' else 'base'
            mean = '(' + '+'.join(f'cast({family}{k}.{col} as double)' for k in range(3)) + ')/3.'
            if family == 'public_rome':
                expected['p'] = '.5*(' + mean + '+' + expected['r20'] + ')'
            else:
                expected[family + '_pool'] = mean
                expected[family + '_neural'] = '.5*(cast(ref.calls_nn as double)+' + mean + ')'
        for name in ['public_main', 'public_airport', 'daily_main', 'public_cat']:
            bind(name, spec[name]); joins.append('join ' + name + ' using(mvt_id)')
            expected[name] = f'cast({name}.pred as double)'
        con.read_parquet([str(verify(record)) for record in spec['ordinary']]).create_view('ordinary')
        assert con.execute('select count(*),count(distinct mvt_id) from ordinary').fetchone() == (rows, rows)
        joins.append('join ordinary using(mvt_id)')
        suffix = '_part' if view == 'old' else ''
        ordinary_airport = f'cast(ordinary.airport{suffix} as double)'
        ordinary_cat = f'cast(ordinary.catboost{suffix} as double)'
        for arm, parts in {
            'v76': [expected['public_main'], ordinary_airport, ordinary_cat],
            'airport': [expected['public_main'], expected['public_airport'], ordinary_cat],
            'cat': [expected['public_main'], expected['public_airport'], expected['public_cat']],
            'main': [expected['daily_main'], expected['public_airport'], expected['public_cat']],
        }.items():
            expected[arm + '_tree'] = '(' + '+'.join(parts) + ')/3.'
        clause = ' or '.join(f'f."{name}" is distinct from ({expr})' for name, expr in expected.items())
        n, bad = con.execute('select count(*),count(*) filter(where ' + clause + ') from f ' + ' '.join(joins)).fetchone()
        assert n == rows and bad == 0
        if view == 'old':
            for name, record in identity['controls'].items(): bind('control_' + name, record)
            for output, source, column in OLD_PAIRS:
                assert con.execute(f'''select count(*) from f join control_{source} c using(mvt_id)
                    where f."{output}" is distinct from c."{column}"''').fetchone()[0] == 0
    return dict(rows=rows, months=months, SQL_mixture_values_exact=rows*len(expected),
        SQL_old_control_values_exact=rows*len(OLD_PAIRS) if view == 'old' else 0)


def corrections(spec, view):
    data = pd.read_pickle(verify(spec['inputs']))
    ids = np.asarray(data['rank_ids'])
    assert list(data['months']) == list(EXPECTED_MONTHS[view])
    actual = pd.read_parquet(OUT / (view + '.parquet'))
    template_ids = pd.read_parquet(verify(spec['template']), columns=['MVT_ID_mvt']).MVT_ID_mvt.to_numpy()
    np.testing.assert_array_equal(actual.mvt_id, template_ids)
    actual = actual.set_index('mvt_id').loc[ids]
    ref = pd.read_parquet(verify(spec['reference'])).set_index('mvt_id').loc[ids]
    arms = [('control', 'physical_neural', None)] + [(name + '_direct', 'ple_neural', name + '_tree') for name in ['v76', 'airport', 'cat', 'main']] + [('tabm_direct', 'tabm_neural', 'main_tree')]
    offset = 0; checked = 0
    for month, item in data['months'].items():
        n = len(item['raw']); assert n == EXPECTED_MONTHS[view][month]
        sl = slice(offset, offset+n)
        np.testing.assert_array_equal(item['raw'].mvt_id, ids[sl])
        for column, neural, tree in arms:
            base = original.W_NN * actual[neural].to_numpy()[sl] + (1-original.W_NN) * (ref.gbm_base.to_numpy()[sl] if tree is None else actual[tree].to_numpy()[sl])
            probability = ref.p.to_numpy()[sl] if tree is None else actual.p.to_numpy()[sl]
            # Direct invocation of the two retained correction steps, avoiding
            # builder.build, parent.final_frame and nis.final wrappers.
            values = original.hs.nis.signals.combine(base, probability, item['V'], item['raw'],
                late_mean=data['parameters']['late_mean'], dayshift_weight=None)
            frame, _ = original.hs.nis.v43s.v42.parent.apply_corrections(values['final'], item['raw'], item['V'], data['fitted'], values['p'])
            np.testing.assert_array_equal(frame.mvt_id, ids[sl])
            np.testing.assert_array_equal(frame.pred, actual[column].to_numpy()[sl], err_msg=view + ':' + month + ':' + column)
            checked += n
        offset += n
    assert offset == len(ids)
    del data, actual, ref
    gc.collect()
    return checked


def main():
    assert not (OUT / 'independent_audit.json').exists()
    for path, digest in PINNED.items(): verify(dict(path=path, sha256=digest))
    identity = read(OUT / 'identity.json')
    assert identity['views'] == ['old', 'final'] and identity['no_fit'] and identity['no_affine_substitution']
    assert identity['source']['path'] == 'VALIDATION/final_public_direct_components.py'
    verify(identity['source'])
    for record in identity['sources']: verify(record)
    sources = {r['path']: r['sha256'] for r in identity['sources']}
    assert sources[str(Path(original.__file__).relative_to(ROOT))] == sha(original.__file__)
    checks = {}
    for view in ['old', 'final']:
        spec = identity['specs'][view]; receipt = read(OUT / (view + '.json'))
        assert receipt['identity'] == art(OUT / 'identity.json')
        assert receipt['view'] == view and receipt['all_original_direct_values_exact']
        assert receipt['template'] == spec['template'] and receipt['no_fit'] and receipt['no_submission']
        assert receipt['old_values_checked'] == (344841*len(OLD_PAIRS) if view == 'old' else 0)
        assert receipt['prediction'] == art(OUT / (view + '.parquet'))
        for item in spec.values():
            for record in item if isinstance(item, list) else [item]:
                assert sources[record['path']] == record['sha256']; verify(record)
        receipts(spec, view)
        checks[view] = sql_checks(identity, spec, view)
        checks[view]['all_original_correction_values_exact'] = corrections(spec, view)
        checks[view]['prediction'] = receipt['prediction']
    result = dict(status='pass', source=art(Path(__file__)), identity=art(OUT / 'identity.json'),
        views=checks, source_hashes_checked=len(CHECKED), template_order_exact=True,
        all_original_receipt_bindings_exact=True, independent_SQL_mixtures_and_old_controls=True,
        independent_original_correction_reconstruction=True,
        caveat=identity['caveat'], no_fit_or_model_inference=True, no_submission=True,
        interpretation='Direct components only; no affine assembly, source posterior port or complete final submission claimed.')
    (OUT / 'independent_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status='pass', rows={k:v['rows'] for k,v in checks.items()}, source_hashes_checked=len(CHECKED))))


if __name__ == '__main__':
    with threadpool_limits(1): main()
