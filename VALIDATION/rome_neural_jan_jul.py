"""Fixed neural Rome-classifier screen on the selected combined public stack.

Use the original Rome population and public inputs, excluding the five
departure-label history fields so the inner stopping split has no such history
from its validation month. Fit encoders/bins separately for inner and full
training. One seed and one fixed one-third neural probability share; no upload.
"""
import argparse
import gc
import hashlib
from pathlib import Path
import pickle

import duckdb
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import rome_neural_model as model
import rome_public_delay_cat as cat
import rome_public_peer_features as peer
import public_stack_cv as joint
from records import write_json

ROOT = cat.ROOT
OUT = ROOT / 'VALIDATION/runs/v3_20261011_rome_neural'
art, read, verify = joint.art, joint.read, joint.verify
DESIGN = dict(parent='selected combined public main/neural stack with public Cat3/R20',
    change='fixed one-third PLE neural recording-classifier probability',
    model=model.SETTINGS, rows='original R20 Rome population and binary target',
    input_columns=75, removed=peer.DROP_HISTORY, new_inputs=[],
    removing_history='exclude departure-label history to keep the inner stopping split isolated',
    probability='(2/3)*current publicCat3/R20 mean + (1/3)*neural',
    primary='taxi RMSE after full correction stack and v74 transfer',
    no_blend_or_architecture_search=True, no_cost_weighting=True,
    seed=42, machine='desktop RTX3080; inner and full fits on same machine',
    evaluation='reused Jan/Jul development; new single member is an exploratory screen')


def guard():
    assert read(joint.OUT / 'audit.json')['status'] == 'pass'
    assert read(cat.r20.features.OUT / 'audit.json')['status'] == 'pass'
    paths = [Path(__file__), Path(model.__file__), Path(model.original.__file__),
        Path(cat.__file__), Path(cat.r20.__file__), Path(peer.__file__), Path(joint.__file__),
        joint.OUT / 'result.json', joint.OUT / 'audit.json', joint.OUT / 'scored.parquet',
        cat.OUT / 'scored.parquet', cat.r20.features.base.OUT / 'rows2025.parquet',
        cat.r20.features.OUT / 'audit.json']
    paths += [cat.r20.OUT / 'inputs' / f'{m}.pkl' for m in cat.parent.ROUNDS]
    value = dict(design=DESIGN, sources=[art(p) for p in paths], cpu_checks=model.checks('cpu'))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'identity.json'
    if path.exists():
        assert read(path) == value
    else:
        assert not list(OUT.glob('**/*.pt')) and not list(OUT.glob('**/*.parquet'))
        write_json(path, value)
    return value


def inputs(month):
    original, old = cat.parent.validated_parent(month)
    d = pd.read_pickle(cat.r20.OUT / 'inputs' / f'{month}.pkl')
    for key in ['target', 'train_ids', 'ids', 'use']:
        np.testing.assert_array_equal(d[key], old[key])
    for part in ['A', 'B']:
        pd.testing.assert_frame_equal(d[part][original['columns']], old[part][original['columns']])
        assert len(d[part].columns) == 80
    raw = pd.read_parquet(cat.r20.features.base.OUT / 'rows2025.parquet').set_index('mvt_id')
    train = raw.loc[d['train_ids']]
    assert train.airport.eq('LIRF').all()
    assert set(train.month) == set(cat.r20.features.base.FOLDS[month])
    assert month not in set(train.month) and not np.intersect1d(d['train_ids'], d['ids']).size
    np.testing.assert_array_equal(d['target'], ((train.gap-train.y).abs() <= 60).astype(int))
    assert raw.loc[d['ids'][d['use']], 'airport'].eq('LIRF').all()
    for part in ['A', 'B']:
        d[part] = d[part].drop(columns=peer.DROP_HISTORY)
        assert len(d[part].columns) == 75
        assert set(c for c in d[part] if isinstance(d[part][c].dtype, pd.CategoricalDtype)) == set(original['categoricals'])
    d['months'] = train.month.to_numpy()
    return d


def paths(month, phase):
    stem = OUT / f'{month}_{phase}'
    return stem.with_suffix('.pt'), stem.with_suffix('.pkl'), stem.with_suffix('.json')


def trained(month, phase):
    mp, pp, rp = paths(month, phase)
    if not rp.exists():
        assert not mp.exists() and not pp.exists(), 'Unrecorded model/preprocessing; inspect before resuming'
        return None
    rec = read(rp)
    assert rec['identity'] == art(OUT / 'identity.json') and rec['phase'] == phase and rec['month'] == month
    verify(rec['model'])
    verify(rec['preprocessing'])
    return rec


def probability_hash(values):
    return hashlib.sha256(np.asarray(values, dtype='<f8').tobytes()).hexdigest()


def retain(month, phase, network, state, curve, ids, epochs, months, inference=None):
    mp, pp, rp = paths(month, phase)
    assert not mp.exists() and not pp.exists() and not rp.exists()
    before_save = None if inference is None else model.probabilities(network, model.encode(state, inference, 'cuda'))
    with pp.with_suffix('.pkl.tmp').open('wb') as f:
        pickle.dump(state, f, protocol=5)
    pp.with_suffix('.pkl.tmp').replace(pp)
    torch.save(dict(state_dict=network.state_dict(), design=DESIGN), mp.with_suffix('.pt.tmp'))
    mp.with_suffix('.pt.tmp').replace(mp)
    ip = OUT / f'{month}_{phase}_ids.parquet'
    pd.DataFrame(dict(mvt_id=ids)).to_parquet(ip, index=False)
    rec = dict(identity=art(OUT / 'identity.json'), month=month, phase=phase,
        model=art(mp), preprocessing=art(pp), training_ids=art(ip), rows=len(ids),
        epochs=epochs, seed=42, months=sorted(set(months)), curve=curve,
        device=torch.cuda.get_device_name(), columns=state['columns'])
    if before_save is not None:
        restored, restored_state = load_model(rec, 'cuda')
        after_reload = model.probabilities(restored, model.encode(restored_state, inference, 'cuda'))
        np.testing.assert_array_equal(before_save, after_reload)
        rec['held_reload'] = dict(rows=len(before_save), exact=True,
            probability_sha256=probability_hash(before_save), comparison='in-memory trained model versus disk checkpoint')
        del restored, restored_state
    write_json(rp, rec)
    return rec


def load_model(record, device):
    with verify(record['preprocessing']).open('rb') as handle:
        state = pickle.load(handle)
    checkpoint = torch.load(verify(record['model']), map_location=device, weights_only=False)
    assert checkpoint['design'] == DESIGN
    network = model.Classifier(state).to(device)
    network.load_state_dict(checkpoint['state_dict'], strict=True)
    return network, state


def fit():
    guard()
    assert torch.cuda.get_device_name() == 'NVIDIA GeForce RTX 3080'
    write_json(OUT / 'gpu_preflight.json', model.checks('cuda'))
    for month in cat.parent.ROUNDS:
        d = inputs(month)
        last = max(d['months'])
        hold = d['months'] == last
        inner = trained(month, 'inner')
        if inner is None:
            state = model.preprocessing(d['A'].iloc[~hold])
            network, curve = model.fit(state, d['A'].iloc[~hold], d['target'][~hold],
                model.SETTINGS['max_epochs'], monitor=(d['A'].iloc[hold], d['target'][hold]))
            inner = retain(month, 'inner', network, state, curve, d['train_ids'][~hold],
                model.SETTINGS['max_epochs'], d['months'][~hold])
            del network, state
            gc.collect()
            torch.cuda.empty_cache()
        assert last not in inner['months']
        selected = min(inner['curve'], key=lambda point: point['inner_logloss'])['epoch']
        full = trained(month, 'full')
        if full is None:
            state = model.preprocessing(d['A'])
            network, curve = model.fit(state, d['A'], d['target'], selected)
            full = retain(month, 'full', network, state, curve, d['train_ids'], selected, d['months'], inference=d['B'])
            del network, state
            gc.collect()
            torch.cuda.empty_cache()
        assert full['epochs'] == selected
        path = OUT / f'{month}_predictions.parquet'
        rp = path.with_suffix('.json')
        if rp.exists():
            rec = read(rp)
            assert rec['full_model'] == art(paths(month, 'full')[2]) and rec['prediction'] == art(path)
            continue
        assert not path.exists(), 'Unrecorded predictions'
        network, state = load_model(full, 'cuda')
        pred = model.probabilities(network, model.encode(state, d['B'], 'cuda'))
        assert full['held_reload']['exact'] and full['held_reload']['rows'] == len(pred)
        assert full['held_reload']['probability_sha256'] == probability_hash(pred)
        p = np.zeros(len(d['ids']))
        p[d['use']] = pred
        pd.DataFrame(dict(mvt_id=d['ids'], p=p)).to_parquet(path, index=False)
        write_json(rp, dict(full_model=art(paths(month, 'full')[2]), prediction=art(path),
            rows=len(p), rome_rows=len(pred), all_rome_reload_exact=True, selected_epoch=selected,
            selection_month=last, inner_model=art(paths(month, 'inner')[2])))
        del network, state, d
        gc.collect()
        torch.cuda.empty_cache()


def score():
    guard()
    assert not (OUT / 'result.json').exists(), 'Preserve completed score'
    identity = joint.neural.guard()
    replay = cat.r20.physical.replay
    sc, r20, _, trees = replay.baseline.baseline()
    reference = pd.read_parquet(joint.OUT / 'scored.parquet').set_index('mvt_id')
    rome = pd.read_parquet(cat.OUT / 'scored.parquet').set_index('mvt_id')
    raw = pd.read_parquet(cat.r20.features.base.OUT / 'rows2025.parquet').set_index('mvt_id')
    matches = pd.read_parquet(cat.r20.constraints.exclusion.OUT / 'matches2025.parquet')
    frames = []
    for month in cat.parent.ROUNDS:
        d = sc.d[month]
        ids = d['ids']
        rec = read(OUT / f'{month}_predictions.json')
        assert rec['all_rome_reload_exact'] and rec['full_model'] == art(paths(month, 'full')[2])
        neural = pd.read_parquet(verify(rec['prediction']))
        np.testing.assert_array_equal(neural.mvt_id, ids)
        calls = replay.values(replay.combo.sequence.CALLS_PRED, ids, 'candidate_nn')
        bank = [replay.values(verify(joint.neural.completed(month, k, identity)['prediction']), ids, 'base') for k in range(3)]
        main = replay.values(joint.main.OUT / 'predictions' / f'{month}_s42.parquet', ids, 'pred')
        ordinary = .71*.5*(calls+np.mean(bank, axis=0))+.29*np.mean([main, trees['airport'][42][month], trees['catboost'][42][month]], axis=0)
        p_old = .5*rome.loc[ids, 'newcat'].to_numpy()+.5*r20[month]
        prior = rome.loc[ids]
        permitted = matches[matches.donor_month.isin(cat.r20.features.base.FOLDS[month]+[month])]
        values = {}
        for name, p in [('control', p_old), ('candidate', (2/3)*p_old+(1/3)*neural.p.to_numpy()), ('pure', neural.p.to_numpy())]:
            final = sc.final(month, ordinary, p)
            transferred = cat.r20.constraints.matched_bound(raw.loc[ids].reset_index(),
                prior.v74.to_numpy()+final-prior.control.to_numpy())
            transferred, _ = cat.r20.constraints.exclusion.apply(ids, np.maximum(transferred, 0), permitted)
            values[name], values[name+'_transfer'] = final, transferred
        np.testing.assert_array_equal(values['control'], reference.loc[ids, 'combined'])
        np.testing.assert_array_equal(values['control_transfer'], reference.loc[ids, 'combined_transfer'])
        non_rome = ~raw.loc[ids, 'airport'].eq('LIRF').to_numpy()
        np.testing.assert_array_equal(values['candidate_transfer'][non_rome], values['control_transfer'][non_rome])
        frames.append(pd.DataFrame(dict(mvt_id=ids, month=month, day=d['day'], y=d['y'], **values)))
    f = pd.concat(frames, ignore_index=True)
    assert len(f) == 344419 and f.mvt_id.is_unique
    columns = list(values)
    assert np.isfinite(f[columns]).all().all() and f[columns].ge(0).all().all()
    scores = {m: {c: float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in columns} for m, g in f.groupby('month')}
    pairs = [('control', 'candidate'), ('control_transfer', 'candidate_transfer')]
    paired = {f'{b}_vs_{a}': cat.parent.shared.ts.paired_bootstrap(f.day, f.month, (f[a]-f.y)**2, (f[b]-f.y)**2) for a, b in pairs}
    f.to_parquet(OUT / 'scored.parquet', index=False)
    result = dict(design=DESIGN, scores=scores, cv={c: float(np.mean([s[c] for s in scores.values()])) for c in columns},
        paired=paired, original_joint_controls_exact=True, non_rome_unchanged=True,
        identity=art(OUT / 'identity.json'), predictions=art(OUT / 'scored.parquet'),
        interpretation='Fixed one-third blend is primary; pure model diagnostic only; no production selected')
    write_json(OUT / 'result.json', result)
    print(dict(cv=result['cv'], paired=paired), flush=True)


def audit():
    guard()
    assert not (OUT / 'audit.json').exists(), 'Preserve completed audit'
    result = read(OUT / 'result.json')
    assert result['identity'] == art(OUT / 'identity.json') and result['predictions'] == art(OUT / 'scored.parquet')
    for month in cat.parent.ROUNDS:
        d = inputs(month)
        last = max(d['months'])
        for phase, mask in [('inner', d['months'] != last), ('full', np.ones(len(d['target']), bool))]:
            rec = trained(month, phase)
            saved_ids = pd.read_parquet(verify(rec['training_ids']))
            np.testing.assert_array_equal(saved_ids.mvt_id, d['train_ids'][mask])
            with verify(rec['preprocessing']).open('rb') as handle:
                state = pickle.load(handle)
            assert state['columns'] == list(d['A']) and not set(peer.DROP_HISTORY)&set(state['columns'])
            # Independent category vocabulary calculation from actual fitting IDs.
            for name, mapping in state['encoder'].maps.items():
                counts = d['A'].iloc[mask][name].astype(object).value_counts()
                keep = sorted(str(v) for v, n in counts.items() if n >= model.ENCODING['min_cat_rows'])
                assert mapping == {v: k+1 for k, v in enumerate(keep)}
        inner, full = trained(month, 'inner'), trained(month, 'full')
        assert full['epochs'] == min(inner['curve'], key=lambda point: point['inner_logloss'])['epoch']
        rec = read(OUT / f'{month}_predictions.json')
        assert rec['all_rome_reload_exact'] and rec['prediction'] == art(OUT / f'{month}_predictions.parquet')
        pred = pd.read_parquet(OUT / f'{month}_predictions.parquet')
        np.testing.assert_array_equal(pred.mvt_id, d['ids'])
        assert full['held_reload']['exact'] and full['held_reload']['rows'] == int(d['use'].sum())
        assert full['held_reload']['probability_sha256'] == probability_hash(pred.p.to_numpy()[d['use']])
    with duckdb.connect(config={'threads': 1}) as con:
        for name, path in [('candidate', OUT / 'scored.parquet'), ('truth', cat.r20.features.base.OUT / 'rows2025.parquet'),
                           ('control', joint.OUT / 'scored.parquet')]:
            con.read_parquet(str(path)).create_view(name)
        n, unique, wrong = con.execute('''select count(*),count(distinct c.mvt_id),
            sum(case when c.y is distinct from t.y or c.day is distinct from t.day or c.month is distinct from t.month
            or c.control is distinct from p.combined or c.control_transfer is distinct from p.combined_transfer
            then 1 else 0 end) from candidate c join truth t using(mvt_id) join control p using(mvt_id)''').fetchone()
        assert n == unique == 344419 and wrong == 0
        for name in result['cv']:
            values = con.execute(f'select month,sqrt(avg(pow({name}-y,2))) as value from candidate group by month').fetchall()
            for month, value in values:
                np.testing.assert_allclose(value, result['scores'][month][name], atol=1e-8, rtol=0)
            np.testing.assert_allclose(np.mean([v for m, v in values]), result['cv'][name], atol=1e-8, rtol=0)
    value = dict(status='pass', rows=n, training_membership_exact=True, inner_month_excluded=True,
        encoder_vocabularies_train_only=True, independent_sql_truth_controls_scores=True,
        result=art(OUT / 'result.json'), source=art(__file__), identity=art(OUT / 'identity.json'),
        scope='All saved classifier probabilities reloaded in fit; independent fitting-ID/vocabulary/stopping and SQL score checks')
    write_json(OUT / 'audit.json', value)
    print(value, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['guard', 'fit', 'score', 'audit'])
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(2):
        globals()[args.stage]()
