"""A coherent no-plan query view of v58's original 156 columns.

All own plan columns and their clock/history descendants are removed. Other
flights retain their observable plans, as at deployment. Existing own physical
witnesses are reused only when removing CALLSIGN_flt leaves the original lookup
key unchanged. Unverified keys and known no-matches both become all-NaN physical
families, so the masked representation cannot identify augmentation provenance
from a zero-versus-NaN match flag. Genuine matched zero durations are retained.

The metadata export reads movement inputs and labels separately: labels are
used only to check existing training membership and monitor fixed-epoch fits.
No hidden departure block timestamp is read. No model fitting occurs here.
"""
import argparse
import json
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'VALIDATION'),str(ROOT/'pipeline')]
from records import sha256,write_json

DEFAULT=ROOT/'data/derived/missing_plan_view_20261010'
DROP_GROUPS={
    'plan_categories':['market_segment','flight_type','wake_cat','operator'],
    'plan_gaps':['mvt_minus_lobt_sec','mvt_minus_iobt_sec','mvt_minus_eobt1_sec','mvt_minus_aobt3_sec'],
    'operator_history':['op_taxi_30d','op_taxi_30d_n'],
    'obt_ref_counts':['dep_actual_30m_before','arr_actual_30m_before','dep_sched_60m_around'],
    'own_queue':['q_queue_ahead_since_aobt3'],
    'own_clock_comparison':['pc_own_minus_apt'],
    'clocks':['clock_aobt3_minus_schedule_sec','clock_aobt3_minus_eobt1_sec',
        'clock_eobt1_minus_iobt_sec','clock_iobt_minus_lobt_sec','clock_schedule_available',
        'clock_lobt_available','clock_iobt_available','clock_eobt1_available','clock_aobt3_available',
        'clock_raw_flight_id_missing','clock_raw_plan_block_missing','clock_raw_aobt3_missing',
        'clock_available_count','clock_span_sec','clock_std_sec','clock_median_sec',
        'clock_baseline_sec','clock_baseline_source'],
    'match_flags':['op_matched','ab_matched'],
}
DROP=[c for cols in DROP_GROUPS.values() for c in cols]
OP_OWN=['op_fs_ground','op_fs_sec','op_own_stand_sec','op_first_stand_sec',
        'op_first_apron_sec','op_first_twy_sec','op_rwy_sec','op_deice_sec','op_ground_sec']
AB_OWN=['ab_seg_sec','ab_move_sec','ab_still_end_sec','ab_seg_n','ab_first_gs',
        'ab_last_gnd_sec','ab_prev_gap_sec','ab_stand_min_m','ab_stand_n','ab_stand_exit_sec',
        'ab_exit_gap_sec','ab_seg_stand_m','ab_seg_rwy_m']
SETTINGS=dict(drop_groups=DROP_GROUPS,feature_count=121,
    query_only_mask=True,recovery='off',physical_reuse='unchanged fallback key AND original matched flag==1',
    unavailable_physical='all NaN for both changed/unverified keys and original no-match/missing flags',
    output_gate='original raw AOBT3 missing AND movement airport != LIRF',
    raw_removed='all16 PLAN_FIELDS and FLIGHT_ID; none of their own direct/derived descendants retained',
    label_use='membership/target check and fixed-epoch monitoring only; no label-based feature construction')


def key_flags(frame):
    """Exactly reproduce each existing matcher's string handling, for testing."""
    cs=frame.CALLSIGN_flt.astype('string')
    flt=frame.FLIGHT_mvt.astype('string')
    op_cs=cs.str.strip().replace('',pd.NA)
    op_flt=flt.str.strip().replace('',pd.NA)
    original=op_cs.fillna(op_flt)
    op=(original.eq(op_flt)|(original.isna()&op_flt.isna())).fillna(False)
    normalize=lambda x:x.fillna('').str.replace(' ','',regex=False).str.upper()
    ab=normalize(cs.fillna(flt)).eq(normalize(flt))
    return pd.DataFrame(dict(op_key_ok=op.to_numpy(bool),ab_key_ok=ab.to_numpy(bool)),index=frame.index)


def transform(X,ids,metadata):
    if len(X.columns)!=156 or len(set(X.columns))!=156 or len(DROP)!=35:
        raise ValueError('Expected original156 unique columns and frozen35 removals')
    if not set(DROP+OP_OWN+AB_OWN)<=set(X.columns):
        raise ValueError('Original input contract changed')
    if not metadata.index.is_unique or len(ids)!=len(X) or not np.isin(ids,metadata.index).all():
        raise ValueError('Missing or ambiguous movement metadata')
    meta=metadata.loc[np.asarray(ids)]
    if meta[['op_key_ok','ab_key_ok']].isna().any().any():
        raise ValueError('Unknown lookup-key provenance')
    keep=[c for c in X if c not in DROP]
    out=X.loc[:,keep].copy()
    for key,flag,family in [('op_key_ok','op_matched',OP_OWN),('ab_key_ok','ab_matched',AB_OWN)]:
        valid=meta[key].to_numpy(bool)&X[flag].eq(1).to_numpy()
        out.loc[~valid,family]=np.nan
    if len(out.columns)!=121 or not out.index.equals(X.index):
        raise ValueError('No-plan view changed alignment')
    return out


def gate(metadata,ids):
    q=metadata.loc[np.asarray(ids)]
    return q.raw_aobt3_missing.to_numpy(bool)&q.airport.ne('LIRF').to_numpy()


def build():
    DEFAULT.mkdir(parents=True,exist_ok=True)
    dest=DEFAULT/'metadata.parquet'
    manifest=DEFAULT/'manifest.json'
    sources={str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/'data').glob('training_2025-*.parquet'))}
    if len(sources)!=12:
        raise ValueError('Expected twelve source training months')
    provenance=dict(settings=SETTINGS,source_sha256=sha256(Path(__file__)),raw_sources=sources)
    if dest.exists():
        old=json.loads(manifest.read_text())
        if any(old[k]!=v for k,v in provenance.items()) or old['metadata_sha256']!=sha256(dest):
            raise ValueError('Existing metadata identity changed')
        return
    with duckdb.connect(config={'threads':2}) as con:
        con.execute("SET TimeZone='UTC'")
        table=con.execute('''SELECT MVT_ID_mvt mvt_id,
          strftime(MVT_TIME_UTC_mvt,'%Y-%m') AS month,ADEP_mvt airport,
          AOBT_3_flt IS NULL raw_aobt3_missing,
          (coalesce(nullif(trim(CALLSIGN_flt),''),nullif(trim(FLIGHT_mvt),''))
            IS NOT DISTINCT FROM nullif(trim(FLIGHT_mvt),'')) op_key_ok,
          upper(replace(coalesce(CALLSIGN_flt,FLIGHT_mvt,''),' ',''))=
            upper(replace(coalesce(FLIGHT_mvt,''),' ','')) ab_key_ok,
          TAXITIME_SEC_mvt taxi_sec
          FROM read_parquet(?) WHERE PHASE_mvt='DEP' ORDER BY MVT_ID_mvt''',
          [str(ROOT/'data/training_2025-*.parquet')]).df()
    if table.mvt_id.isna().any() or table.mvt_id.duplicated().any():
        raise ValueError('Ambiguous movement metadata')
    for c in ['month','airport']:
        table[c]=table[c].astype('category')
    table.to_parquet(dest,index=False)
    write_json(manifest,dict(**provenance,metadata_sha256=sha256(dest),rows=len(table),
        columns=list(table),months=sorted(table.month.unique()),
        raw_missing_rows=int(table.raw_aobt3_missing.sum())))


def load():
    manifest=json.loads((DEFAULT/'manifest.json').read_text())
    if manifest['settings']!=SETTINGS or manifest['source_sha256']!=sha256(Path(__file__)):
        raise ValueError('No-plan view source/settings changed')
    path=DEFAULT/'metadata.parquet'
    if sha256(path)!=manifest['metadata_sha256']:
        raise ValueError('Metadata transfer hash mismatch')
    return pd.read_parquet(path).set_index('mvt_id'),manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['build','check'])
    a=p.parse_args()
    if a.stage=='build':build()
    else:print(load()[1])
