"""Synthetic checks for the pure transport interface; never reads real predictions."""
from pathlib import Path
import copy
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "VALIDATION"))
import final_shared_clock_transport_20261013 as model


def main():
    # Shuffled IDs, both shift signs, clamp, an unchanged shared row, new IDs.
    args = dict(final_ids=np.array([30, 10, 40, 20, 50, 60]),
                old_ids=np.array([20, 10, 60, 30]),
                old_prediction=np.array([1, 100, 50, 200]),
                old_takeoff_seconds=np.array([100.25, 200.5, 300.75, 400.125]),
                final_takeoff_seconds=np.array([412.125, 196.5, 800, 98.25, 900, 300.75]),
                portable_prediction=np.array([300, 301, 302, 303, 304, 305]))
    values, mask, dt = model.transform_all_ids(**args)
    np.testing.assert_array_equal(values, [224, 92, 302, 0, 304, 50])
    np.testing.assert_array_equal(mask, [True, True, False, True, False, True])
    np.testing.assert_array_equal(dt, [12, -4, 0, -2, 0, 0])
    assert values.dtype == np.int32
    # ID permutations commute with transport; old donor order is immaterial.
    pi, oi = np.array([5, 3, 0, 2, 4, 1]), np.array([3, 1, 0, 2])
    permuted = {k: v[pi if k in ["final_ids", "final_takeoff_seconds", "portable_prediction"] else oi]
                for k, v in args.items()}
    p, pm, pd = model.transform_all_ids(**permuted)
    np.testing.assert_array_equal(p, values[pi])
    np.testing.assert_array_equal(pm, mask[pi])
    np.testing.assert_array_equal(pd, dt[pi])
    # Reversing a nonclipped view transform recovers the same integer exactly.
    for change in range(-12, 13):
        p, _, _ = model.transform_all_ids([1], [1], [100], [100], [100 + change], [7])
        back, _, _ = model.transform_all_ids([1], [1], p, [100 + change], [100], [7])
        np.testing.assert_array_equal(back, [100])
    bad = [
        ("duplicate_final", "final_ids", [30, 10, 40, 20, 50, 50]),
        ("duplicate_old", "old_ids", [20, 10, 60, 20]),
        ("missing_old_ID", "old_ids", [20, 10, 70, 30]),
        ("noninteger_ID", "old_ids", [20, 10, 60.1, 30]),
        ("unserialized_fallback", "portable_prediction", [300, 301, 302.5, 303, 304, 305]),
        ("negative_fallback", "portable_prediction", [300, 301, -1, 303, 304, 305]),
        ("nan_fallback", "portable_prediction", [300, 301, np.nan, 303, 304, 305]),
        ("unserialized_v80", "old_prediction", [1, 100.5, 50, 200]),
        ("fractional_delta", "final_takeoff_seconds", [412.625, 196.5, 800, 98.25, 900, 300.75]),
        ("outside_observed_delta", "final_takeoff_seconds", [413.125, 196.5, 800, 98.25, 900, 300.75]),
        ("int32_overflow", "old_prediction", [1, 100, 50, np.iinfo(np.int32).max]),
    ]
    rejected = []
    for name, key, value in bad:
        corrupted = copy.deepcopy(args)
        corrupted[key] = np.array(value)
        try:
            model.transform_all_ids(**corrupted)
        except AssertionError:
            rejected.append(name)
        else:
            raise AssertionError((name, "did not fail closed"))
    out = ROOT / "VALIDATION/reports/cohort_learning_20261013/final_shared_clock_transport_synthetic_v2.json"
    model.write(out, dict(status="pass", builder=model.art(model.__file__), source=model.art(__file__),
        fixtures=["exact sign, ID routing, clipping and fallback", "both ID permutations", "25 inverse nonclipped transforms"],
        rejected=rejected, real_data_read=False, final_predictions_written=False,
        validation="Pure operator correctness only; not predictive CV or departure-generator evidence"))
    print(json.dumps({"status": "pass", "report": model.art(out)}))


if __name__ == "__main__":
    main()
