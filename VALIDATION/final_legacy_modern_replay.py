"""Port selected archived blends using retained old/final predictions, without fits.

Each mapping must reproduce the entire scored old integer vector before its
four-month output is retained. No absent affine component is substituted.
"""
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

import callsign_submit as parent
import sparse_callsign_submit as sparse
import nn_track_physical_submit as physical
from records import sha256, write_json

ROOT = parent.ROOT
OUT = ROOT / 'VALIDATION/runs/v3_20261012_final_legacy_modern_replay'
VERSIONS = [54, 57, 58, 64, 66, 68, 70]


def art(p):
    return dict(path=str(Path(p).relative_to(ROOT)), sha256=sha256(p))


def verify(a):
    p = ROOT / a['path']
    assert sha256(p) == a['sha256'], p
    return p


def finish(data, neural, tree, probability, weight):
    frames, offset = [], 0
    for month, item in data['months'].items():
        end = offset + len(item['raw'])
        sl = slice(offset, end)
        base = weight * neural[sl] + (1 - weight) * tree[sl]
        frame, _ = parent.hs.nis.final(base, probability[sl], item['V'], item['raw'],
                                      data['parameters'], data['fitted'])
        frame['month'] = month
        frames.append(frame[['mvt_id', 'month', 'pred']])
        offset = end
    result = pd.concat(frames, ignore_index=True)
    np.testing.assert_array_equal(result.mvt_id, data['rank_ids'])
    return result


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    cache_path = ROOT / 'data/derived/scored_submission_cache_20261011/manifest.json'
    cache = json.loads(cache_path.read_text())
    sources = [Path(__file__), Path(parent.__file__), Path(sparse.__file__), Path(physical.__file__),
               Path(parent.hs.nis.__file__), cache_path]
    # The immutable October11 cache predates these three component submissions.
    for version in [66, 68, 70]:
        name = 'calls_only_candidate' if version == 68 else 'physical_candidate'
        path = ROOT / f'VALIDATION/runs/v3_20261011_{name}/submission_v{version}/manifest.json'
        m = json.loads(path.read_text())
        assert m['state'] == 'scored' and m['remote_sha256'] == m['sha256']
        cache[str(version)] = dict(file=m['file'], sha256=m['sha256'])
        sources.append(path)
    physical_records = []
    for k in range(3):
        path = physical.paths(k)[2]
        r = json.loads(path.read_text())
        physical_records.append(r)
        sources += [path, physical.paths(k)[1]]
        for a in r['predictions'].values():
            sources.append(verify(a))
    for view in ['old', 'final']:
        path = sparse.OUT / f'candidates/{view}/manifest.json'
        m = json.loads(path.read_text())
        sources += [path, verify(m['parts']), verify(m['output']), verify(m['template'])]
    for v in VERSIONS:
        p = ROOT / cache[str(v)]['file']
        assert sha256(p) == cache[str(v)]['sha256']
        sources.append(p)
    identity = dict(sources=[art(p) for p in sources], versions=VERSIONS, no_fit=True, no_upload=True,
                    gate='Every old integer prediction exact before final assembly',
                    formulas={'54': 'HPO neural weight0.6', '57': 'HPO neural weight0.45',
                              '58': 'HPO neural weight0.71', '64': 'Retained sparse/callsign blend',
                              '66': 'v64 physical member0 replaces sparse pool',
                              '68': 'v64 callsign-only neural component',
                              '70': 'v64 physical members0/1/2 replace sparse pool'})
    if (OUT / 'identity.json').exists():
        assert json.loads((OUT / 'identity.json').read_text()) == identity
    else:
        write_json(OUT / 'identity.json', identity)
    for view in ['old', 'final']:
        if (OUT / f'{view}.json').exists():
            continue
        data = parent.ranking_data(view)
        ids = data['rank_ids']
        manifest = json.loads((sparse.OUT / f'candidates/{view}/manifest.json').read_text())
        reference = pd.read_parquet(verify(manifest['parts']))
        np.testing.assert_array_equal(reference.mvt_id, ids)
        control, masks = parent.final_frame(data, reference.nn_base.to_numpy(),
                                           reference.gbm_base.to_numpy(), reference.p.to_numpy())
        pd.testing.assert_frame_equal(control, reference[control.columns], check_exact=True)
        assert masks == manifest['correction_masks']
        template = pq.read_table(verify(manifest['template']))
        parent.assert_parent_output(control, template, pq.read_table(verify(manifest['output'])), reference)
        neural, tree, probability, hpo_sources = parent.parent_parts(view, data)
        members = []
        for r in physical_records:
            frame = pd.read_parquet(verify(r['predictions'][view])).set_index('mvt_id')
            assert frame.index.is_unique and set(frame.index) == set(ids)
            members.append(frame.loc[ids, 'base'].to_numpy(float))
        results = []
        for version in VERSIONS:
            if view == 'final' and not json.loads((OUT / f'old_v{version}.json').read_text())['all_original_predictions_exact']:
                continue
            if version in [54, 57, 58]:
                frame = finish(data, neural, tree, probability, {54: .6, 57: .45, 58: .71}[version])
            elif version == 64:
                frame = control[['mvt_id', 'month', 'pred']].copy()
            else:
                calls = reference.calls_nn.to_numpy(float)
                new_neural = calls if version == 68 else .5 * (calls + (members[0] if version == 66 else np.mean(members, axis=0)))
                frame = finish(data, new_neural, reference.gbm_base.to_numpy(), reference.p.to_numpy(), .71)
            table = sparse.serialized(template, ids, frame.pred.to_numpy())
            # Always store in template order, including final's different month order.
            target_ids = template['MVT_ID_mvt'].to_numpy()
            frame = frame.set_index('mvt_id').loc[target_ids].reset_index()
            frame['submitted'] = table['TAXITIME_SEC_mvt'].to_numpy()
            assert frame.mvt_id.is_unique and np.isfinite(frame.submitted).all()
            path = OUT / f'{view}_v{version}.parquet'
            frame.to_parquet(path, index=False)
            result = dict(version=version, rows=len(frame), identity=art(OUT / 'identity.json'),
                          inputs=art(parent.fs.WORK / f'inputs_{view}.pkl'),
                          original=art(ROOT / cache[str(version)]['file']),
                          hpo_parts=hpo_sources, exact_sparse_parent_replay=True,
                          prediction=art(path), template=manifest['template'])
            if view == 'old':
                expected = pd.read_parquet(ROOT / cache[str(version)]['file']).set_index('MVT_ID_mvt').TAXITIME_SEC_mvt.loc[target_ids]
                delta = frame.submitted.to_numpy() - expected.to_numpy()
                result.update(all_original_predictions_exact=bool((delta == 0).all()),
                              changed_rows=int(np.count_nonzero(delta)),
                              rms_difference=float(np.sqrt(np.mean(delta ** 2))),
                              max_abs_difference=float(np.max(np.abs(delta))))
            else:
                result['old_control_exact'] = True
            write_json(OUT / f'{view}_v{version}.json', result)
            print({k: v for k, v in result.items() if not isinstance(v, (dict, list))}, flush=True)
            results.append(result)
        write_json(OUT / f'{view}.json', dict(view=view, results=results, no_fit=True, no_upload=True))
        del data, reference, members
        gc.collect()


if __name__ == '__main__':
    with threadpool_limits(2):
        run()
