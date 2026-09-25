"""Stand position inputs (idea 34, step 2): where the stand is and how far it is
from the start of the departure runway.

Positions come from `data/external/stand_maps/stand_coords.parquet` and runway
ends from `runway_ends.parquet`, built by `tools/build_stand_coords.py` from the
X-Plane Gateway and OpenStreetMap. Only static map geometry and the supplied
`ADEP_mvt`/`STAND_mvt`/`RUNWAY_mvt` fields are read; no labels.

* ``sc_x_m`` / ``sc_y_m``  metres east / north of the airport's runway-end centre;
* ``sc_rwy_end_m``         straight-line metres from the stand to the end of the
                           runway named in `RUNWAY_mvt`, where the take-off roll
                           starts (NaN when the stand or runway has no position).
"""
from pathlib import Path

import numpy as np
import pandas as pd

import stand_geometry as sg

FEATURES = ['sc_x_m', 'sc_y_m', 'sc_rwy_end_m']
COORDS = sg.OUT_DIR / 'stand_coords.parquet'
ENDS = sg.OUT_DIR / 'runway_ends.parquet'
SETTINGS = dict(features=FEATURES, origin='mean of the airport X-Plane runway ends',
                distance='flat-earth metres (stand_geometry.metres)',
                runway_point='X-Plane runway end for RUNWAY_mvt (paved end, not displaced threshold)',
                sources='stand_coords.parquet / runway_ends.parquet (tools/build_stand_coords.py)')


def load(coords=COORDS, ends=ENDS):
    """Stand positions keyed by airport:stand and runway ends keyed by airport:runway."""
    c = pd.read_parquet(coords).dropna(subset=['stand', 'lat'])
    e = pd.read_parquet(ends)
    if c.duplicated(['airport', 'stand']).any() or e.duplicated(['airport', 'runway']).any():
        raise ValueError('Duplicate stand or runway positions')
    origin = e.groupby('airport')[['lat', 'lon']].mean()
    return c, e, origin


def add_features(X, raw, tables=None):
    """Return X with FEATURES added, row-aligned with raw."""
    c, e, origin = tables or load()
    if len(X) != len(raw):
        raise ValueError('Inputs and raw movements are not row-aligned')
    ap = raw['ADEP_mvt'].astype('string').to_numpy()
    stand = pd.Series(ap + ':' + raw['STAND_mvt'].astype('string').to_numpy())
    rwy = pd.Series(ap + ':' + raw['RUNWAY_mvt'].astype('string').to_numpy())
    pos = c.assign(k=c.airport + ':' + c.stand).set_index('k')[['lat', 'lon']]
    end = e.assign(k=e.airport + ':' + e.runway).set_index('k')[['lat', 'lon']]
    slat, slon = (stand.map(pos[col]).to_numpy(float) for col in ('lat', 'lon'))
    rlat, rlon = (rwy.map(end[col]).to_numpy(float) for col in ('lat', 'lon'))
    olat, olon = (pd.Series(ap).map(origin[col]).to_numpy(float) for col in ('lat', 'lon'))
    out = X.copy()
    mlat = np.radians(olat)
    out['sc_x_m'] = (slon - olon) * sg.M_PER_DEG_LAT * np.cos(mlat)
    out['sc_y_m'] = (slat - olat) * sg.M_PER_DEG_LAT
    out['sc_rwy_end_m'] = sg.metres(slat, slon, rlat, rlon)
    return out


def coverage(X):
    return {col: dict(known=int(X[col].notna().sum()), rows=len(X),
                      minimum=float(X[col].min()), maximum=float(X[col].max())) for col in FEATURES}
