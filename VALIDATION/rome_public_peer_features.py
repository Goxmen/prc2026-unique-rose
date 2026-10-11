"""Public daily budgets after cross-fitted peer recording-mode estimates.

The feature builder reads no departure labels. Probabilities must be supplied
by the month-excluded inner classifiers. Fixed 900/1800-second ordinary
references match the selected public-feature recipe. All airport-day sums
exclude explicit VFR and include unknown flight rules.
"""
import numpy as np
import pandas as pd

FEATURES = ['peer_copy_probability', 'peer_missing_count',
    'peer_resid900_mean', 'peer_own900_left', 'peer_own900_fraction',
    'peer_resid1800_mean', 'peer_own1800_left', 'peer_own1800_fraction',
    'peer_uncertainty900', 'peer_uncertainty1800']
DROP_HISTORY = ['op_taxi_30d', 'op_taxi_30d_n', 'rwy_taxi_30d',
                'stand_taxi_90d', 'aphour_taxi_30d']


def build(rows, probabilities, source):
    """Return one row per input ID, preserving its order and complete day peers."""
    required = ['mvt_id', 'airport', 'day', 'gap', 'has_id', 'fltrule']
    f = rows[required].copy()
    assert f.mvt_id.is_unique and f.airport.eq('LIRF').all()
    assert probabilities.index.is_unique
    observed = probabilities.reindex(f.mvt_id).to_numpy(float)
    assert np.isfinite(observed[~np.isnan(observed)]).all()
    assert ((observed[~np.isnan(observed)] >= 0) & (observed[~np.isnan(observed)] <= 1)).all()
    missing = np.isnan(observed)
    hard = ~f.has_id.to_numpy(bool) & f.gap.ge(14400).to_numpy()
    p = np.where(missing, np.where(hard, 1., .5), observed)
    eligible = ~f.fltrule.eq('V').to_numpy()
    f['n'] = eligible.astype(float)
    f['missing'] = (missing & eligible & ~hard).astype(float)
    for ref in [900, 1800]:
        c = np.maximum(f.gap.to_numpy(float)-ref, 0) * eligible
        f[f'c{ref}'] = c
        f[f'e{ref}'] = (1-p)*c
        f[f'v{ref}'] = p*(1-p)*c*c
    sums = ['n', 'missing', 'e900', 'e1800', 'v900', 'v1800']
    g = f.groupby(['airport', 'day'], as_index=False)[sums].sum()
    g = g.merge(source[['airport', 'day', 'total']], on=['airport', 'day'],
                how='left', validate='one_to_one')
    g = f[['mvt_id', 'airport', 'day', 'c900', 'c1800', 'e900', 'e1800']].merge(
        g, on=['airport', 'day'], how='left', suffixes=('_own', ''), validate='many_to_one')
    np.testing.assert_array_equal(g.mvt_id, f.mvt_id)
    out = pd.DataFrame(dict(mvt_id=f.mvt_id.to_numpy(),
        peer_copy_probability=p, peer_missing_count=g['missing'].to_numpy()))
    for ref in [900, 1800]:
        residual = g.total-g[f'e{ref}']
        left = residual+g[f'e{ref}_own']
        out[f'peer_resid{ref}_mean'] = residual/np.maximum(g.n, 1)
        out[f'peer_own{ref}_left'] = left
        out[f'peer_own{ref}_fraction'] = left/np.maximum(g[f'c{ref}'], 600)
        out[f'peer_uncertainty{ref}'] = np.sqrt(np.maximum(g[f'v{ref}'], 0))
    return out[['mvt_id']+FEATURES]


def checks():
    rows = pd.DataFrame(dict(mvt_id=[1, 2, 3, 4], airport='LIRF', day='2025-01-01',
        gap=[5000., 3000., 8000., 15000.], has_id=[True, True, True, False],
        fltrule=['I', None, 'V', 'I']))
    p = pd.Series([.2, .8, .3], index=[1, 2, 3])
    source = pd.DataFrame(dict(airport=['LIRF'], day=['2025-01-01'], total=[3000.]))
    actual = build(rows, p, source).set_index('mvt_id')
    expected = .8*4100+.2*2100
    np.testing.assert_allclose(actual.peer_resid900_mean, (3000-expected)/3)
    np.testing.assert_allclose(actual.loc[1, 'peer_own900_left'], 3000-.2*2100)
    np.testing.assert_allclose(actual.peer_uncertainty900, np.sqrt(.2*.8*(4100**2+2100**2)))
    assert actual.loc[4, 'peer_copy_probability'] == 1 and (actual.peer_missing_count == 0).all()
    poisoned = rows.assign(y=1e12, TAXITIME_SEC_mvt=-1e10, BLOCK_TIME_UTC_mvt='2099-01-01')
    pd.testing.assert_frame_equal(build(poisoned, p, source), build(rows, p, source))
    shuffled = build(rows.sample(frac=1, random_state=42), p, source).set_index('mvt_id').sort_index()
    pd.testing.assert_frame_equal(shuffled, actual.sort_index())
    return dict(hand_arithmetic=True, explicit_vfr_excluded=True, unknown_rule_included=True,
        hard_rule_fallback=True, label_isolation=True, row_order_invariant=True)


if __name__ == '__main__':
    print(checks())
