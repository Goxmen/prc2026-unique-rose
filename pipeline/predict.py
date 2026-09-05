"""Fill the submission template with LightGBM predictions for Jan/Jul 2026.

Starts from data/submitting.parquet (keeping its row order and schema exactly),
joins ranking departures by MVT_ID_mvt, and writes
submissions/unique-rose_v<N>.parquet.

Run: .venv/bin/python pipeline/predict.py --version 2
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from common import (
    CATEGORICAL,
    MODELS,
    RANKING,
    RAW_COLS,
    SUBMISSIONS,
    TEAM,
    TEMPLATE,
    build_features,
    connect,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, required=True, help="submission number N in unique-rose_vN.parquet")
    ap.add_argument("--model", default=MODELS / "lgbm_raw.txt", help="booster from train.py")
    args = ap.parse_args()

    booster = lgb.Booster(model_file=str(args.model))
    meta = json.loads(Path(args.model).with_suffix(".meta.json").read_text())

    con = connect()
    ranking = con.sql(
        f"""
        SELECT MVT_ID_mvt AS mvt_id, {RAW_COLS}
        FROM read_parquet('{RANKING}')
        WHERE PHASE_mvt = 'DEP'
        """
    ).df()

    X = build_features(ranking)
    for c in CATEGORICAL:
        # mask levels unseen in training to NaN (LightGBM treats as missing)
        cats = pd.Index(meta["categories"][c])
        X[c] = pd.Categorical(X[c].where(X[c].isin(cats)), categories=cats)
    chunk = 20_000
    parts = [
        booster.predict(X.iloc[i : i + chunk])
        for i in tqdm(range(0, len(X), chunk), desc="predict", unit="chunk")
    ]
    ranking["pred"] = np.clip(np.concatenate(parts), 0, None)

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
    holdout = meta.get("holdout_rmse_sec")
    print(f"wrote {dest.relative_to(dest.parents[1])}: {len(filled):,} rows, "
          f"pred mean {filled.mean():.0f} s, range [{filled.min()}, {filled.max()}]"
          + (f"  (model holdout RMSE {holdout} s)" if holdout is not None else ""))
    print(f"submit with: mc cp {dest} opensky/prc-2026-{TEAM}/")


if __name__ == "__main__":
    main()
