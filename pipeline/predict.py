"""Fill the submission template with baseline predictions for Jan/Jul 2026.

Starts from data/submitting.parquet (keeping its row order and schema exactly),
joins ranking departures by MVT_ID_mvt, and writes
submissions/unique-rose_v<N>.parquet.

Run: .venv/bin/python pipeline/predict.py --version 1
"""

import argparse
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from common import RANKING, SUBMISSIONS, TEAM, TEMPLATE, connect
from train import MODEL_PATH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, required=True, help="submission number N in unique-rose_vN.parquet")
    ap.add_argument("--model", default=MODEL_PATH, help="model JSON from train.py")
    args = ap.parse_args()

    model = json.loads(open(args.model).read())

    con = connect()
    ranking = con.sql(
        f"""
        SELECT MVT_ID_mvt AS mvt_id, ADEP_mvt AS airport
        FROM read_parquet('{RANKING}')
        WHERE PHASE_mvt = 'DEP'
        """
    ).df()

    unknown = set(ranking["airport"]) - set(model["airport_mean"])
    if unknown:
        print(f"warning: {sorted(unknown)} not in model, using global mean")
    ranking["pred"] = (
        ranking["airport"].map(model["airport_mean"]).fillna(model["global_mean"])
    )

    # Fill the template in its own row order; never touch MVT_ID_mvt.
    template = pq.read_table(TEMPLATE)
    ids = template.column("MVT_ID_mvt").to_pandas()
    pred = ids.map(ranking.set_index("mvt_id")["pred"])
    assert len(ids) == len(ranking), f"template has {len(ids)} rows, ranking {len(ranking)} departures"
    assert not pred.isna().any(), "some template MVT_IDs have no prediction"

    out = template.set_column(
        template.schema.get_field_index("TAXITIME_SEC_mvt"),
        "TAXITIME_SEC_mvt",
        pa.array(np.rint(pred).astype(np.int32), type=pa.int32()),
    )
    assert out.schema.equals(template.schema), "submission schema drifted from template"

    SUBMISSIONS.mkdir(exist_ok=True)
    dest = SUBMISSIONS / f"{TEAM}_v{args.version}.parquet"
    pq.write_table(out, dest)
    filled = out.column("TAXITIME_SEC_mvt").to_pandas()
    print(f"wrote {dest.relative_to(dest.parents[1])}: {len(filled):,} rows, "
          f"pred mean {filled.mean():.0f} s, range [{filled.min()}, {filled.max()}]")
    print(f"submit with: mc cp {dest} opensky/prc-2026-{TEAM}/")


if __name__ == "__main__":
    main()
