"""Independent SQL identity/serialization and original-file audit of legacy ports."""
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'VALIDATION/runs'
OUT = RUNS / 'v3_20261012_final_legacy_audit'
PREFIXES = ['v3_20261012_final_legacy_' + name + '_replay' for name in ['v38', 'overnight', 'modern', 'adsb']]


def read(p):
    return json.loads(p.read_text())


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def art(p):
    return dict(path=str(p.relative_to(ROOT)), sha256=sha(p))


def main():
    assert not (OUT / 'audit.json').exists()
    cache = read(ROOT / 'data/derived/scored_submission_cache_20261011/manifest.json')
    verified, rows, excluded = set(), [], []

    def verify(a):
        p = ROOT / a['path']
        if (str(p), a['sha256']) not in verified:
            assert sha(p) == a['sha256'], p
            verified.add((str(p), a['sha256']))
        return p

    with duckdb.connect(config={'threads': 1, 'memory_limit': '1GB'}) as con:
        con.execute("SET TimeZone='UTC'")
        con.read_parquet(str(ROOT / 'data/final_submitting.parquet')).create_view('template')
        con.read_parquet(str(ROOT / 'data/final_ranking.parquet')).create_view('raw')
        for name in PREFIXES:
            directory = RUNS / name
            identity_path = directory / 'identity.json'
            identity = read(identity_path)
            for a in identity['sources']:
                verify(a)
            old, final = read(directory / 'old.json'), read(directory / 'final.json')
            if 'results' not in old:
                old, final = [dict(old, version=38)], [dict(final, version=38)]
            else:
                old, final = old['results'], final['results']
            old = {r['version']: r for r in old}
            permitted = {v for v, r in old.items() if r['all_original_predictions_exact']}
            assert permitted == {r['version'] for r in final}
            excluded += [dict(version=v, changed_rows=r['changed_rows'], rms_difference=r['rms_difference'])
                         for v, r in old.items() if v not in permitted]
            for r in final:
                version = r['version']
                o = old[version]
                assert o['identity'] == r['identity'] == art(identity_path)
                assert o['rows'] == 344841 and r['rows'] == 670790
                for entry in [o, r]:
                    verify(entry['inputs'])
                    verify(entry['prediction'])
                if str(version) in cache:
                    original = ROOT / cache[str(version)]['file']
                    assert sha(original) == cache[str(version)]['sha256']
                else:
                    original = verify(o['original'])
                con.read_parquet(str(original)).create_view('original', replace=True)
                con.read_parquet(str(verify(o['prediction']))).create_view('old', replace=True)
                assert con.execute('select count(*),count(distinct mvt_id) from old').fetchone() == (344841, 344841)
                assert con.execute('''select count(*) from old full join original
                    on old.mvt_id=original.MVT_ID_mvt where old.mvt_id is null or original.MVT_ID_mvt is null
                    or old.submitted is distinct from original.TAXITIME_SEC_mvt''').fetchone()[0] == 0
                con.read_parquet(str(verify(r['prediction']))).create_view('p', replace=True)
                assert con.execute('select count(*),count(distinct mvt_id) from p').fetchone() == (670790, 670790)
                col = 'pred_combined' if version == 38 else 'pred'
                assert con.execute(f'''select count(*) from p full join template t on p.mvt_id=t.MVT_ID_mvt
                    where p.mvt_id is null or t.MVT_ID_mvt is null or p.{col} is null or not isfinite(p.{col})
                    or p.submitted is distinct from round_even(greatest(0,p.{col}),0)''').fetchone()[0] == 0
                months = dict(con.execute('''select strftime(r.MVT_TIME_UTC_mvt,'%Y-%m'),count(*)
                    from p join raw r on p.mvt_id=r.MVT_ID_mvt where r.PHASE_mvt='DEP' group by 1''').fetchall())
                assert months == {'2026-01': 152719, '2026-02': 144158, '2026-06': 181791, '2026-07': 192122}
                rows.append(dict(version=version, final_prediction=r['prediction'], old_exact_in_SQL=True,
                                 final_rows=670790, months=months))
    # The base-network final arrays predate this work and came from October8
    # refits. Their tiny floating-point difference must not become a claim of
    # bitwise original-weight reproduction merely because blend replay passes.
    folder = RUNS / 'v3_20261008_final_submit/work'
    networks, neural_sources = [], []
    ids = views = None
    for group in [42, 142, 242]:
        for member in range(5):
            path = folder / f'nn/n{group}_{member}.parquet'
            frame = pd.read_parquet(path)
            if ids is None:
                ids, views = frame.mvt_id.to_numpy(), frame['set'].to_numpy()
            else:
                np.testing.assert_array_equal(frame.mvt_id, ids)
                np.testing.assert_array_equal(frame['set'], views)
            networks.append(frame.pred.to_numpy(float))
            neural_sources.append(art(path))
    mean = np.mean(networks, axis=0)
    final_nn = pd.read_parquet(folder / 'parts/nn.parquet')
    np.testing.assert_array_equal(final_nn.mvt_id, ids[views == 'final'])
    np.testing.assert_array_equal(final_nn.pred, mean[views == 'final'].astype(np.float32))
    old_path = RUNS / 'v3_20261006_hpo_submit/work/parts/nn.parquet'
    old_nn = pd.read_parquet(old_path)
    np.testing.assert_array_equal(old_nn.mvt_id, ids[views == 'old'])
    difference = mean[views == 'old'] - old_nn.drop(columns='mvt_id').to_numpy(float).mean(axis=1)
    metadata = read(folder / 'parts/nn.json')
    rms = float(np.sqrt(np.mean(difference ** 2)))
    np.testing.assert_allclose(rms, metadata['against_v54_old_inputs']['rms_diff'], atol=1e-12, rtol=0)
    lineage = dict(prior_refit_date='2026-10-08', members=15, final_mean_recomputed_exact=True,
                   old_input_float_rms_difference=rms, old_original_weights_bitwise_reproduced=False,
                   sources=neural_sources + [art(old_path), art(folder / 'parts/nn.json'), art(folder / 'parts/nn.parquet')],
                   scope='v54/v57/v58 final neural parts reuse these archived refits; this audit performs no fitting.')
    result = dict(status='pass', sources_verified=len(verified), predictions=rows, excluded_mappings=excluded,
                  archived_base_network_provenance=lineage,
                  source=art(Path(__file__)), all_SQL_id_joins_and_rounding_exact=True,
                  interpretation='Only listed archived components ported; missing affine basis remains unresolved.',
                  no_fit=True, no_upload=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status='pass', versions=[r['version'] for r in rows], excluded=excluded,
                          values_checked=len(rows) * (344841 + 670790))))


if __name__ == '__main__':
    main()
