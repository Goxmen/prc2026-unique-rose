"""Fixed exact-v80 screen of runway-conditioned source measurement covariance."""
import argparse
import ast
import inspect
from pathlib import Path
import textwrap
import types

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import public_runway_covariance_pilot as pilot
import joint_public_other_runway_cv as old
import tuning_seasonal as validation

ROOT = pilot.ROOT
OUT = ROOT / 'VALIDATION/runs/v3_20261012_public_runway_covariance_cv'
CURRENT = ROOT / 'VALIDATION/runs/v3_20261012_tabm_pool3_other_runway_combo'
art, read, write, verify = old.art, old.read, old.write, old.verify
DESIGN = dict(parent='exactv80', change=pilot.DESIGN['change'], covariance_pseudocells=100,
    calibration='Same allowed training cells and unchanged full-window Huber bias/global covariance; new shrinkage only by runway',
    source='Selected robust all-same-date-runway posterior; update every taxi measurement block consistently',
    transfer='Constrain old native runway component plus0.5*(new_full_posterior-old_full_posterior); add its integer difference once to v80 and clip/round',
    gate='Both Jan/Jul RMSEs improve and paired-day95% upper<0 against exactv80',
    retained='All taxi models, priors, source gates, delay noise, common15-second term, calendars and other v80 components',
    no_weight_search=True, no_extra_fits=True, no_production=True, no_upload=True,
    caveat='Source calibration pilot is not independent confirmation; Jan/Jul development is reused')


def api(enabled):
    def covariance(parameters, runway):
        if enabled:
            return np.asarray(parameters.get('runway_covariances', {}).get(runway, parameters['covariance_per_row']))
        return np.asarray(parameters['covariance_per_row'])

    original = ast.parse("np.asarray(taxi_parameters['covariance_per_row'])", mode='eval').body

    def change(fn, **extra):
        text = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(text)

        class Replace(ast.NodeTransformer):
            count = 0

            def visit_Call(self, node):
                if ast.dump(node) == ast.dump(original):
                    self.count += 1
                    return ast.copy_location(ast.parse('source_covariance(taxi_parameters,key[1])', mode='eval').body, node)
                return self.generic_visit(node)

        transform = Replace()
        tree = transform.visit(tree)
        assert transform.count == 1
        restored = ast.unparse(tree).replace('source_covariance(taxi_parameters, key[1])', ast.unparse(original))
        assert ast.dump(ast.parse(restored)) == ast.dump(ast.parse(text))
        namespace = dict(fn.__globals__, source_covariance=covariance, **extra)
        exec(compile(ast.fix_missing_locations(tree), '<runway-covariance-only>', 'exec'), namespace)
        return namespace['apply']

    base = types.SimpleNamespace(**vars(old.model.base))
    base.apply = change(old.model.base.apply)
    return change(old.model.apply, base=base)


def freeze():
    assert not (OUT / 'identity.json').exists()
    result = read(pilot.OUT / 'result.json')
    assert result['gate'] and result['source_density_scipy_exact']
    for a in read(pilot.OUT / 'identity.json')['sources']:
        verify(a)
    parameters = read(old.parent.OUT / 'parameters.json')
    cells = pd.read_parquet(pilot.parent.OUT / 'calibration.parquet')
    for month, p in parameters.items():
        c = cells[cells.fold.eq(month)]
        beta, common, covariances, counts = pilot.fit(c)
        np.testing.assert_array_equal(beta, p['taxi']['coefficients'])
        np.testing.assert_array_equal(common, p['taxi']['covariance_per_row'])
        assert p['taxi']['allowed'] == pilot.parent.daily.base.FOLDS[month]
        p['taxi']['runway_covariances'] = {k: v.tolist() for k, v in covariances.items()}
        p['taxi']['runway_covariance_cells'] = counts
    api(False)
    api(True)
    # Source code remains immutable and the sole numerical intervention is one
    # lookup in each of the two inherited covariance assembly functions.
    paths = [Path(__file__), Path(pilot.__file__), Path(old.__file__), Path(old.model.__file__),
        Path(old.model.base.__file__), pilot.OUT / 'identity.json', pilot.OUT / 'result.json',
        pilot.OUT / 'scored_source_cells.parquet', pilot.OUT / 'fits.json',
        pilot.parent.OUT / 'calibration.parquet', old.OUT / 'predictions.parquet',
        old.OUT / 'scored.parquet', old.OUT / 'result.json', old.OUT / 'audit.json',
        CURRENT / 'scored.parquet', CURRENT / 'result.json']
    prior = read(old.parent.OUT / 'identity.json')
    paths += [verify(a) for a in prior['inputs']]
    for a in read(old.OUT / 'identity.json')['correction_sources']:
        paths.append(verify(a))
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / 'parameters.json', parameters)
    paths.append(OUT / 'parameters.json')
    write(OUT / 'identity.json', dict(design=DESIGN, sources=[art(p) for p in paths],
        original_global_calibration_exact=True, inherited_AST_restoration_exact=True, before_inference=True))
    print('Frozen covariance-only exact-v80 comparison', flush=True)


def guard():
    identity = read(OUT / 'identity.json')
    assert identity['design'] == DESIGN
    for a in identity['sources']:
        verify(a)
    return identity


def infer():
    guard()
    assert not (OUT / 'inference.json').exists()
    rows = pd.read_parquet(old.parent.OUT / 'inputs.parquet', columns=old.model.pilot.INPUTS)
    taxi = pd.read_parquet(old.parent.OUT / 'taxi_source.parquet')
    delay = pd.read_parquet(old.parent.OUT / 'delay_source.parquet')
    expected = pd.read_parquet(old.OUT / 'predictions.parquet').set_index('mvt_id')
    parameters = read(OUT / 'parameters.json')
    frames, checks = [], []
    for month, p in parameters.items():
        inputs = rows[rows.month.eq(month)].reset_index(drop=True)
        control, _ = api(False)(inputs, taxi, delay, p['taxi'], p['delay'], p['support'])
        result, _ = api(True)(inputs, taxi, delay, p['taxi'], p['delay'], p['support'])
        cached = expected.loc[control.mvt_id]
        np.testing.assert_array_equal(control.touched, cached.touched)
        np.testing.assert_array_equal(result.touched, cached.touched)
        np.testing.assert_allclose(control.new_joint, cached.new_joint, atol=1e-8, rtol=0)
        other, _ = api(True)(inputs.sample(frac=1, random_state=19).assign(y=1e12, true_day='1900-01-01'),
                            taxi, delay, p['taxi'], p['delay'], p['support'])
        np.testing.assert_allclose(result.new_joint, other.set_index('mvt_id').loc[result.mvt_id, 'new_joint'], atol=1e-8, rtol=0)
        f = result[['mvt_id', 'touched', 'extra_cells', 'new_joint']].copy()
        f['old_joint'] = cached.new_joint.to_numpy()
        f['posterior_delta'] = f.new_joint - f.old_joint
        f.loc[~f.touched, 'posterior_delta'] = 0.
        frames.append(f.assign(month=month))
        checks.append(dict(month=month, rows=len(f), touched=int(f.touched.sum()),
            old_replay_max=float(np.max(abs(control.new_joint.to_numpy() - cached.new_joint.to_numpy()))),
            row_permutation_and_poisoned_extraneous_labels_exact_within_1e8=True))
        print(checks[-1], flush=True)
    pd.concat(frames, ignore_index=True).to_parquet(OUT / 'predictions.parquet', index=False)
    write(OUT / 'inference.json', dict(identity=art(OUT / 'identity.json'), checks=checks,
        predictions=art(OUT / 'predictions.parquet'), no_challenge_labels_loaded=True))


def score():
    guard()
    assert not (OUT / 'result.json').exists()
    receipt = read(OUT / 'inference.json')
    assert receipt['identity'] == art(OUT / 'identity.json')
    verify(receipt['predictions'])
    import daily_main_refinements_cv as selected
    control = pd.read_parquet(CURRENT / 'scored.parquet')
    native = pd.read_parquet(old.OUT / 'scored.parquet').set_index('mvt_id').loc[control.mvt_id]
    inferred = pd.read_parquet(OUT / 'predictions.parquet').set_index('mvt_id').reindex(control.mvt_id)
    f = control[['mvt_id', 'month', 'day', 'y']].copy()
    for c in ['month', 'day', 'y']:
        np.testing.assert_array_equal(native[c], f[c])
    f['control'] = control.three_tabm_runway
    f['posterior_delta'] = inferred.posterior_delta.fillna(0).to_numpy()
    f['native_control'] = native.candidate.to_numpy()
    raw = pd.read_parquet(selected.daily.base.OUT / 'rows2025.parquet').set_index('mvt_id').loc[f.mvt_id].reset_index()
    matches = pd.read_parquet(selected.daily.constraints.exclusion.OUT / 'matches2025.parquet')
    constrain = selected.daily.constrain
    ids = f.mvt_id.to_numpy()
    np.testing.assert_array_equal(constrain(raw, ids, f.month, f.native_control.to_numpy(), matches), f.native_control)
    f['native_candidate'] = constrain(raw, ids, f.month, f.native_control.to_numpy() + .5 * f.posterior_delta.to_numpy(), matches)
    f['candidate'] = np.rint(np.maximum(0, f.control + f.native_candidate - f.native_control))
    np.testing.assert_array_equal(f.loc[f.posterior_delta.eq(0), 'candidate'], f.loc[f.posterior_delta.eq(0), 'control'])
    scores = {m: {c: float(np.sqrt(np.mean((g[c] - g.y) ** 2))) for c in ['control', 'candidate']} for m, g in f.groupby('month')}
    paired = validation.paired_bootstrap(f.day, f.month, (f.control - f.y) ** 2, (f.candidate - f.y) ** 2)
    cv = {c: float(np.mean([r[c] for r in scores.values()])) for c in ['control', 'candidate']}
    np.testing.assert_allclose(cv['control'], 289.7627782102906, atol=1e-8, rtol=0)
    f.to_parquet(OUT / 'scored.parquet', index=False)
    with duckdb.connect(config={'threads': 1}) as con:
        for name, path in [('f', OUT / 'scored.parquet'), ('o', CURRENT / 'scored.parquet'),
                           ('r', old.OUT / 'scored.parquet'), ('p', OUT / 'predictions.parquet')]:
            con.read_parquet(str(path)).create_view(name)
        n, u, bad = con.execute('''select count(*),count(distinct f.mvt_id),sum(case when
            f.y is distinct from o.y or f.month is distinct from o.month or f.day is distinct from o.day
            or f.control is distinct from o.three_tabm_runway or f.native_control is distinct from r.candidate
            or f.posterior_delta is distinct from coalesce(p.posterior_delta,0)
            or f.candidate is distinct from round_even(greatest(0,f.control+f.native_candidate-f.native_control),0)
            then 1 else 0 end) from f join o using(mvt_id) join r using(mvt_id) left join p using(mvt_id)''').fetchone()
        assert n == u == 344419 and bad == 0
        for m, value in con.execute('select month,sqrt(avg(pow(candidate-y,2))) from f group by month').fetchall():
            np.testing.assert_allclose(value, scores[m]['candidate'], atol=1e-8, rtol=0)
    result = dict(identity=art(OUT / 'identity.json'), inference=art(OUT / 'inference.json'),
        scored=art(OUT / 'scored.parquet'), scores=scores, cv=cv, paired=paired,
        gate=paired['ci95'][1] < 0 and all(s['candidate'] < s['control'] for s in scores.values()),
        exact_v80_control=True, all_row_SQL_ids_truth_calendar_transport_scores=True,
        changed=int(f.candidate.ne(f.control).sum()), no_production_or_upload=True)
    write(OUT / 'result.json', result)
    print(dict(cv=cv, paired=paired, gate=result['gate'], changed=result['changed']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'infer', 'score'])
    with threadpool_limits(1):
        globals()[parser.parse_args().stage]()
