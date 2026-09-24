"""Resolve each departure movement to an ICAO 24-bit address.

The address is what names an ADSB.lol history file:
    globe_history/YYYY/MM/DD/traces/<last 2 hex>/trace_full_<icao24>.json

Join key is (callsign, departure airport, date). Two details matter:

* OPDI's ``flt_id`` is space-padded, so it needs ``trim()``. An equality join
  without it matches zero rows.
* ``dof`` is the date of flight, not the takeoff date. They disagree around
  midnight UTC, where the unmatched share otherwise reaches 15-21%. We allow a
  one-day window and keep the candidate whose ``first_seen`` is closest to the
  recorded off-block.

Verified on 2025 against three fields the join does not use: aircraft type
agrees 98.3% (LSZH) / 97.5% (LIRF) where both sides know it, destination 96.8% /
96.0% where OPDI has one, and ``first_seen`` lands a median 33 s (LSZH) / 1052 s
(LIRF) after off-block -- the Rome lag being OPDI's missing stand coverage.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

REPO = Path(__file__).resolve().parents[2]
FLIGHT_LIST = REPO / "data/external/opdi/flight_list/flight_list_*.parquet"

QUERY = """
with mv as (
    select MVT_ID_mvt, CALLSIGN_flt, ADEP_mvt, STAND_mvt, RUNWAY_mvt,
           BLOCK_TIME_UTC_mvt as offblock,
           MVT_TIME_UTC_mvt   as takeoff,
           TAXITIME_SEC_mvt   as taxi_sec,
           cast(MVT_TIME_UTC_mvt at time zone 'UTC' as date) as dep_date
    from read_parquet($movements)
    where PHASE_mvt = 'DEP'
      and ADEP_mvt in (select unnest($airports))
      and CALLSIGN_flt is not null
),
fl as (
    select distinct icao24,
           trim(flt_id)            as callsign,
           adep,
           cast(dof as date)       as flight_date,
           first_seen at time zone 'UTC' as first_seen
    from read_parquet($flight_list)
    where adep in (select unnest($airports))
),
ranked as (
    select mv.*, fl.icao24, fl.first_seen,
           row_number() over (
               partition by mv.MVT_ID_mvt
               order by abs(date_diff('second', mv.offblock, fl.first_seen)),
                        fl.icao24
           ) as rk
    from mv
    join fl
      on mv.CALLSIGN_flt = fl.callsign
     and mv.ADEP_mvt     = fl.adep
     and abs(date_diff('day', fl.flight_date, mv.dep_date)) <= 1
)
select MVT_ID_mvt, CALLSIGN_flt, ADEP_mvt, STAND_mvt, RUNWAY_mvt,
       offblock, takeoff, taxi_sec, dep_date, icao24, first_seen
from ranked
where rk = 1
"""


def resolve(movements: list[str], airports: list[str]) -> "duckdb.DuckDBPyRelation":
    con = duckdb.connect()
    con.execute("set timezone='UTC'")
    return con.execute(
        QUERY,
        {
            "movements": [str(p) for p in movements],
            "flight_list": str(FLIGHT_LIST),
            "airports": airports,
        },
    ).df()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--airports", nargs="+", default=["LSZH"])
    ap.add_argument("--movements", nargs="+",
                    default=sorted(str(p) for p in (REPO / "data").glob("training_2025-*.parquet")))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    df = resolve(args.movements, args.airports)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"{len(df):,} departures resolved -> {args.out}")
    print(f"{df.icao24.nunique():,} distinct aircraft, "
          f"{df.groupby(['icao24', 'dep_date']).ngroups:,} aircraft-days")


if __name__ == "__main__":
    main()
