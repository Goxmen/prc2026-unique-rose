"""Load observable movement fields without enabling flight-plan recovery.

The extra fields support later fold-scoped joins; modeling.observable still
selects only the existing RAW_NAMES. Hidden block timestamps are never loaded.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from common import RAW_COLS

RECOVERY_NAMES = [
    "FLIGHT_ID_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "ADEP_flt", "ADES_flt",
    "ADES_FILED_flt", "FLIGHT_RULE_flt", "AIRCRAFT_TYPE_flt", "ARVT_1_flt", "ARVT_3_flt",
]
RAW_NAMES = [name.strip() for name in RAW_COLS.split(",")]


def load_phase(con, source, phase, sample=1.0):
    if phase not in ("DEP", "ARR") or not 0 < sample <= 1:
        raise ValueError("expected DEP/ARR and a sampling fraction in (0, 1]")
    columns = ["MVT_ID_mvt AS mvt_id", "TAXITIME_SEC_mvt AS taxi_sec"]
    columns += RAW_NAMES + RECOVERY_NAMES
    frac = f"USING SAMPLE {sample * 100:.4f} PERCENT (bernoulli, 7)" if sample < 1 else ""
    frame = con.sql(
        f"SELECT {', '.join(columns)} FROM read_parquet(?) "
        f"WHERE PHASE_mvt = '{phase}' {frac}", params=[source],
    ).df()
    frame["raw_flight_id_missing"] = frame.FLIGHT_ID_mvt.isna()
    frame["raw_plan_block_missing"] = frame.ADEP_flt.isna()
    frame["raw_aobt3_missing"] = frame.AOBT_3_flt.isna()
    return frame
