"""Independently verify cross-fit membership and every peer-budget input."""
import gc
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'VALIDATION'))
import duckdb
import numpy as np
import pandas as pd
import rome_public_peers_jan_jul as run
from records import write_json


def main():
    run.guard()
    checks = []
    for month, allowed in run.r20.features.base.FOLDS.items():
        prepared = run.OUT / 'inputs' / f'{month}.pkl'
        rec = run.cat.parent.shared.ts.read(prepared.with_suffix('.json'))
        assert rec['identity'] == run.artifact(run.OUT / 'identity.json')
        assert rec['prepared'] == run.artifact(prepared)
        assert rec['feature_table'] == run.artifact(prepared.with_name(f'{month}_features.parquet'))
        data = pd.read_pickle(prepared)
        with duckdb.connect(config={'threads': 1}) as con:
            metadata = con.execute('select mvt_id,month from read_parquet(?) where airport=?',
                [str(run.r20.features.base.OUT / 'rows2025.parquet'), 'LIRF']).fetchdf().set_index('mvt_id')
        train_months = metadata.loc[data['train_ids'], 'month'].to_numpy()
        probabilities = []
        for k, audit in enumerate(rec['inner_fits']):
            excluded = sorted(allowed)[k::3] if k < 3 else [month]
            use = ~np.isin(train_months, excluded)
            fit_ids = data['train_ids'][use]
            held_ids = data['train_ids'][~use] if k < 3 else data['ids'][data['use']]
            assert run.sha256_bytes(fit_ids) == audit['fit_ids_sha256']
            assert run.sha256_bytes(held_ids) == audit['held_ids_sha256']
            assert not np.intersect1d(fit_ids, held_ids).size
            assert set(metadata.loc[fit_ids, 'month']) <= set(allowed)
            assert set(metadata.loc[fit_ids, 'month']).isdisjoint(metadata.loc[held_ids, 'month'])
            assert not set(audit['columns']).intersection(run.peer.DROP_HISTORY+run.r20.features.FEATURES)
            assert len(audit['columns']) == 63 and audit['all_held_reload_exact']
            p = pd.read_parquet(ROOT / audit['prediction']['path'])
            assert audit['prediction'] == run.artifact(ROOT / audit['prediction']['path'])
            np.testing.assert_array_equal(p.mvt_id, held_ids)
            probabilities.append(p)
        probability = pd.concat(probabilities, ignore_index=True)
        assert probability.mvt_id.is_unique
        del data, probabilities, metadata
        gc.collect()
        with duckdb.connect(config={'threads': 1}) as con:
            con.register('probability', probability)
            con.register('wanted_month', pd.DataFrame(dict(month=allowed+[month])))
            actual = con.execute('''with raw as (
                select r.mvt_id,r.airport,r.day,r.gap,
                    cast(f.fltrule is distinct from 'V' as double) eligible,
                    coalesce(p.p,case when not r.has_id and r.gap>=14400 then 1.0 else .5 end) cp,
                    cast(p.p is null and f.fltrule is distinct from 'V'
                        and not(not r.has_id and r.gap>=14400) as double) missing
                from read_parquet(?) r join wanted_month w using(month)
                join read_parquet(?) f using(mvt_id) left join probability p using(mvt_id)
                where r.airport='LIRF'),
              contrib as (select *, greatest(0,gap-900)*eligible c900,
                    greatest(0,gap-1800)*eligible c1800 from raw),
              expected as (select *, (1-cp)*c900 e900, (1-cp)*c1800 e1800,
                    cp*(1-cp)*c900*c900 v900, cp*(1-cp)*c1800*c1800 v1800 from contrib),
              aggregate as (select airport,day,sum(eligible) n,sum(missing) missing_n,
                    sum(e900) mean900,sum(e1800) mean1800,sum(v900) var900,sum(v1800) var1800
                    from expected group by airport,day),
              source as (select APT_ICAO airport,cast(FLT_DATE as varchar) AS "day",60*DLY_ALL_PRE_2 total
                    from read_csv_auto(?))
              select r.mvt_id,r.cp peer_copy_probability,a.missing_n peer_missing_count,
                    (s.total-a.mean900)/greatest(a.n,1) peer_resid900_mean,
                    s.total-a.mean900+r.e900 peer_own900_left,
                    (s.total-a.mean900+r.e900)/greatest(r.c900,600) peer_own900_fraction,
                    (s.total-a.mean1800)/greatest(a.n,1) peer_resid1800_mean,
                    s.total-a.mean1800+r.e1800 peer_own1800_left,
                    (s.total-a.mean1800+r.e1800)/greatest(r.c1800,600) peer_own1800_fraction,
                    sqrt(greatest(a.var900,0)) peer_uncertainty900,
                    sqrt(greatest(a.var1800,0)) peer_uncertainty1800
              from expected r join aggregate a using(airport,day) left join source s using(airport,day)
              order by r.mvt_id''', [str(run.r20.features.base.OUT / 'rows2025.parquet'),
                    str(run.r20.features.OUT / 'inputs2025.parquet'),
                    str(run.r20.features.base.SOURCE / 'all_pre_departure_delays_2025.csv')]).fetchdf()
        saved = pd.read_parquet(ROOT / rec['feature_table']['path']).sort_values('mvt_id').reset_index(drop=True)
        np.testing.assert_array_equal(actual.mvt_id, saved.mvt_id)
        np.testing.assert_allclose(actual[run.peer.FEATURES], saved[run.peer.FEATURES], rtol=0, atol=1e-7)
        checks.append(dict(month=month, rows=len(saved), values=len(saved)*len(run.peer.FEATURES),
            all_inner_training_months_disjoint=True, history_columns_absent=True,
            maximum_absolute_difference=float(np.max(np.abs(actual[run.peer.FEATURES].to_numpy()-saved[run.peer.FEATURES].to_numpy())))))
    result = dict(status='pass', source=run.artifact(__file__), identity=run.artifact(run.OUT / 'identity.json'),
        checks=checks, independent_sql_arithmetic=True, independently_reconstructed_inner_ids=True)
    write_json(run.OUT / 'feature_audit.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
