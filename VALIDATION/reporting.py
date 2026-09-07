"""Flight-weighted scores and paired day-block comparisons for protocol v3."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from protocol import A_FOLDS, B_FOLDS, FOLDS, TAIL_SEC, VERSION, SCORE_VERSION

IDENTITY = ["mvt_time", "airport", "y", "has_flight_plan"]
COVERAGE = ["arrival_history_available", "arrival_history_days_available"]


@dataclass
class Loss:
    meta: pd.DataFrame  # unique movement IDs as a sorted index
    sq: np.ndarray     # seeds x flights
    seeds: tuple


def same_flights(left, right):
    if not left.index.equals(right.index) or not left[IDENTITY].equals(right[IDENTITY]):
        raise ValueError("prediction rows/targets do not align by movement ID")


def load_losses(record, directory):
    losses = {}
    seeds = tuple(record["config"]["seeds"])
    order = {f.name: i for i, f in enumerate(FOLDS)}
    for fold, info in sorted(record["folds"].items(), key=lambda item: order.get(item[0], len(order))):
        meta, sq = None, []
        for seed in seeds:
            frame = pd.read_parquet(Path(directory) / info["predictions"][str(seed)])
            if frame.mvt_id.isna().any() or frame.mvt_id.duplicated().any():
                raise ValueError(f"duplicate/missing movement IDs in {fold}/{seed}")
            if not np.isfinite(frame[["y", "pred"]].to_numpy()).all():
                raise ValueError(f"nonfinite targets/predictions in {fold}/{seed}")
            if not frame.seed.eq(seed).all() or not frame.fold.eq(fold).all():
                raise ValueError(f"prediction file identifies the wrong fold/seed: {fold}/{seed}")
            frame = frame.set_index("mvt_id").sort_index()
            frame["mvt_time"] = pd.to_datetime(frame.mvt_time, utc=True)
            if not frame.mvt_time.dt.strftime("%Y-%m").eq(info["val"]).all():
                raise ValueError(f"wrong validation month in {fold}")
            if meta is None:
                meta = frame[IDENTITY + COVERAGE].copy()
            else:
                same_flights(meta, frame)
                if not meta[COVERAGE].equals(frame[COVERAGE]):
                    raise ValueError("feature availability changed across seeds")
            sq.append((frame.y.to_numpy() - frame.pred.to_numpy()) ** 2)
        losses[fold] = Loss(meta, np.stack(sq), seeds)
    return losses


def group_loss(losses, names):
    """Equal weight per flight, averaging losses over origins and then seeds."""
    by_month = {}
    for name in names:
        loss = losses[name]
        month = loss.meta.mvt_time.dt.strftime("%Y-%m").iloc[0]
        by_month.setdefault(month, []).append(loss)
    metas, squares = [], []
    for month, parts in sorted(by_month.items()):
        first = parts[0]
        for part in parts[1:]:
            same_flights(first.meta, part.meta)
            if first.seeds != part.seeds:
                raise ValueError("seed sets differ between folds")
        meta = first.meta.copy()
        meta[COVERAGE] = sum(p.meta[COVERAGE].astype(float) for p in parts) / len(parts)
        metas.append(meta)
        squares.append(np.mean([p.sq for p in parts], axis=0))
    return Loss(pd.concat(metas), np.concatenate(squares, axis=1), parts[0].seeds)


def rmse(loss):
    return float(np.sqrt(loss.sq.mean(axis=0).mean()))


def slice_stats(loss, mask=None):
    mask = np.ones(len(loss.meta), bool) if mask is None else np.asarray(mask, bool)
    n = int(mask.sum())
    if not n:
        return {"n": 0, "rmse": None, "sse": 0.0}
    sq = loss.sq[:, mask].mean(axis=0)
    return {"n": n, "rmse": float(np.sqrt(sq.mean())), "sse": float(sq.sum())}


def summarize(loss):
    meta = loss.meta
    result = slice_stats(loss)
    tail = meta.y.to_numpy() > TAIL_SEC
    result["ordinary"] = slice_stats(loss, ~tail)
    result["tail"] = slice_stats(loss, tail)
    result["tail"]["pct_sse"] = (100 * result["tail"]["sse"] / result["sse"]
                                    if result["sse"] else 0.0)
    result["airports"] = {ap: slice_stats(loss, meta.airport.eq(ap)) for ap in sorted(meta.airport.unique())}
    result["lirf_flight_plan"] = {
        state: slice_stats(loss, meta.airport.eq("LIRF") & meta.has_flight_plan.eq(present))
        for state, present in [("present", True), ("missing", False)]}
    seed_scores = np.sqrt(loss.sq.mean(axis=1))
    result["rmse_by_seed"] = dict(zip(map(str, loss.seeds), map(float, seed_scores)))
    result["seed_sd"] = float(seed_scores.std(ddof=1)) if len(seed_scores) > 1 else None
    first = meta.mvt_time.dt.day <= 7
    result["first_seven_days"] = {
        "n": int(first.sum()),
        "arrival_mean_available_pct": float(100 * meta.loc[first, COVERAGE[0]].mean()) if first.any() else None,
        "mean_arrival_history_days": float(meta.loc[first, COVERAGE[1]].mean()) if first.any() else None,
    }
    return result


def score_groups(losses):
    specifications = {"R_A": [f.name for f in A_FOLDS], "R_B_all": [f.name for f in B_FOLDS],
                      "R_B_long": ["B3", "B4", "B7"]}
    for gap in range(3, 7):
        specifications[f"R_B_k{gap}"] = [f.name for f in B_FOLDS if f.gap == gap]
    groups = {}
    for name, wanted in specifications.items():
        present = [n for n in wanted if n in losses]
        if present:
            key = name if len(present) == len(wanted) else name + "_partial"
            groups[key] = group_loss(losses, present)
    return groups


def aging_pairs(losses):
    pairs = {f"{f.name}_vs_A{int(f.val[-2:]) - 6}": (f.name, f"A{int(f.val[-2:]) - 6}")
             for f in B_FOLDS}
    pairs["S1_vs_A1"] = ("S1", "A1")
    return {k: pair for k, pair in pairs.items() if all(f in losses for f in pair)}


def bootstrap_calendar(losses, repetitions, seed, block_days=1):
    if repetitions < 1 or block_days < 1:
        raise ValueError("bootstrap repetitions and block days must be positive")
    months = sorted({m for loss in losses.values()
                     for m in loss.meta.mvt_time.dt.strftime("%Y-%m").unique()})
    rng, calendar, chunks = np.random.default_rng(seed), [], []
    for month in months:
        p = pd.Period(month, "M")
        days = pd.date_range(p.start_time, p.end_time, freq="D", tz="UTC")
        n = len(days)
        starts = rng.integers(0, n, size=(repetitions, int(np.ceil(n / block_days))))
        indices = ((starts[:, :, None] + np.arange(block_days)) % n).reshape(repetitions, -1)[:, :n]
        weights = np.zeros((repetitions, n), dtype=np.int16)
        np.add.at(weights, (np.arange(repetitions)[:, None], indices), 1)
        chunks.append(weights)
        calendar.extend(days)
    return pd.DatetimeIndex(calendar), np.concatenate(chunks, axis=1)


def boot_scores(loss, calendar, weights):
    index = calendar.get_indexer(loss.meta.mvt_time.dt.normalize())
    if (index < 0).any():
        raise ValueError("bootstrap calendar excludes validation days")
    count = np.bincount(index, minlength=len(calendar))
    sse = np.bincount(index, weights=loss.sq.mean(axis=0), minlength=len(calendar))
    denominator = weights @ count
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt((weights @ sse) / denominator)


def interval(estimate, draws, eligible=True):
    finite = draws[np.isfinite(draws)]
    lo, hi = np.quantile(finite, [0.025, 0.975]) if len(finite) else (np.nan, np.nan)
    decision = "inconclusive"
    if not eligible:
        decision = "execution_check_only"
    elif hi < 0:
        decision = "improvement"
    elif lo > 0:
        decision = "regression"
    return {"difference": float(estimate), "ci95": [float(lo), float(hi)],
            "valid_replicates": len(finite), "decision": decision}


def validate_comparison(record, baseline, losses, baseline_losses):
    for key in ["protocol", "score_version", "data_identity", "ranking_weights"]:
        if record.get(key) != baseline.get(key):
            raise ValueError(f"baseline has incompatible {key}; rerun a matching v3 baseline")
    if baseline.get("protocol") != VERSION or baseline.get("score_version") != SCORE_VERSION:
        raise ValueError("baseline is not a v3 run")
    for key in ["seeds", "sample", "train_months", "early_stopping_rounds"]:
        if record["config"].get(key) != baseline["config"].get(key):
            raise ValueError(f"baseline configuration differs in {key}")
    if losses.keys() != baseline_losses.keys():
        raise ValueError("baseline uses different validation folds")
    for name in losses:
        for key in ["train_start", "train_end", "val", "freeze", "departure_history"]:
            if record["folds"][name][key] != baseline["folds"][name][key]:
                raise ValueError(f"baseline fold {name} differs in {key}")
        same_flights(losses[name].meta, baseline_losses[name].meta)


def report(record, directory, baseline=None):
    losses = load_losses(record, directory)
    groups = score_groups(losses)
    out = {"folds": {k: summarize(v) for k, v in losses.items()},
           "groups": {k: summarize(v) for k, v in groups.items()}}
    scores = {k: v["rmse"] for k, v in out["groups"].items()}
    w = record["ranking_weights"]["jan_fraction"]
    combined = lambda values: np.sqrt(w * values["R_A"] ** 2 + (1-w) * values["R_B_long"] ** 2)
    if {"R_A", "R_B_long"} <= scores.keys():
        scores["R_combined"] = float(combined(scores))
    out["scores"] = scores
    a_values = [rmse(losses[f.name]) for f in A_FOLDS if f.name in losses]
    out["monthly_A_descriptive"] = {"mean": float(np.mean(a_values)) if a_values else None,
                                    "sd": float(np.std(a_values, ddof=1)) if len(a_values) > 1 else None}
    cfg = record["config"]
    calendar, weights = bootstrap_calendar(losses, cfg["bootstrap_reps"], cfg["bootstrap_seed"], cfg["block_days"])
    all_metrics = {**losses, **groups}
    boots = {k: boot_scores(v, calendar, weights) for k, v in all_metrics.items()}
    if "R_combined" in scores:
        boots["R_combined"] = combined(boots)
    eligible = cfg["sample"] == 1
    pairs = aging_pairs(losses)
    out["aging"] = {}
    for name, (old, fresh) in pairs.items():
        same_flights(losses[old].meta, losses[fresh].meta)
        item = interval(rmse(losses[old]) - rmse(losses[fresh]), boots[old] - boots[fresh], eligible)
        item["difference_by_seed"] = dict(zip(map(str, losses[old].seeds),
             map(float, np.sqrt(losses[old].sq.mean(axis=1)) - np.sqrt(losses[fresh].sq.mean(axis=1)))))
        out["aging"][name] = item
    if baseline is not None:
        base, base_dir = baseline
        bl = load_losses(base, base_dir)
        validate_comparison(record, base, losses, bl)
        bg = score_groups(bl)
        bm = {**bl, **bg}
        bb = {k: boot_scores(v, calendar, weights) for k, v in bm.items()}
        bs = {k: rmse(v) for k, v in bg.items()}
        if "R_combined" in scores:
            bs["R_combined"], bb["R_combined"] = float(combined(bs)), combined(bb)
        changes = {}
        for name in scores:
            item = interval(scores[name] - bs[name], boots[name] - bb[name], eligible)
            def seed_scores(metrics):
                values = {k: np.sqrt(v.sq.mean(axis=1)) for k, v in metrics.items()}
                return combined(values) if name == "R_combined" else values[name]
            item["difference_by_seed"] = dict(zip(map(str, cfg["seeds"]),
                map(float, seed_scores(groups) - seed_scores(bg))))
            changes[name] = item
        aging_changes = {name: interval(
            (rmse(losses[old]) - rmse(losses[fresh])) - (rmse(bl[old]) - rmse(bl[fresh])),
            (boots[old] - boots[fresh]) - (bb[old] - bb[fresh]), eligible)
            for name, (old, fresh) in pairs.items()}
        airport_changes = {}
        for group in ["R_A", "R_B_long"]:
            if group not in groups:
                continue
            airport_changes[group] = {}
            headline_change = changes[group]["difference"]
            for airport, stats in out["groups"][group]["airports"].items():
                delta = stats["rmse"] - slice_stats(bg[group], bg[group].meta.airport.eq(airport))["rmse"]
                airport_changes[group][airport] = {"difference": delta,
                    "opposes_group": bool(delta * headline_change < 0)}
        out["comparison"] = {"baseline_id": base["run_id"], "scores": changes,
                             "aging_changes": aging_changes, "airports": airport_changes}
    return out


def _fmt(value, width=10, digits=2):
    return f"{'N/A':>{width}}" if value is None else f"{value:>{width}.{digits}f}"


def print_slices(title, stats):
    """The section 7 slices for one fold or score group."""
    print(f"\n{title}: {stats['n']:,} flights, RMSE {stats['rmse']:.2f} s; "
          f"seed sd {_fmt(stats['seed_sd'], 0)}")
    print(f"  <=7200 s: n {stats['ordinary']['n']:,}, RMSE {_fmt(stats['ordinary']['rmse'], 0)}"
          f" | >7200 s: n {stats['tail']['n']:,}, RMSE {_fmt(stats['tail']['rmse'], 0)},"
          f" {stats['tail']['pct_sse']:.1f}% of SSE")
    print("  airport " + "  ".join(f"{ap} {s['rmse']:.0f}/{s['n']:,}" for ap, s in stats["airports"].items()))
    lirf = stats["lirf_flight_plan"]
    print(f"  LIRF AOBT_3_flt present: RMSE {_fmt(lirf['present']['rmse'], 0)} on {lirf['present']['n']:,};"
          f" missing: RMSE {_fmt(lirf['missing']['rmse'], 0)} on {lirf['missing']['n']:,}")
    first = stats["first_seven_days"]
    print(f"  days 1-7: {first['n']:,} flights, arrival taxi-in mean available "
          f"{_fmt(first['arrival_mean_available_pct'], 0, 1)}%, mean history days "
          f"{_fmt(first['mean_arrival_history_days'], 0, 1)} of 7")


def print_report(result, slices=True):
    print("\nFold     rows       RMSE   <=7200 s   tail rows   tail %SSE   day1-7 arr.hist%")
    for name, stats in result["folds"].items():
        first = stats["first_seven_days"]["arrival_mean_available_pct"]
        print(f"{name:<7} {stats['n']:>8,} {stats['rmse']:>10.2f} {_fmt(stats['ordinary']['rmse'])} "
              f"{stats['tail']['n']:>10,} {stats['tail']['pct_sse']:>11.1f} {_fmt(first, 12, 1)}")
    print("\nFlight-weighted scores (R_combined is a provisional diagnostic, not a forecast):")
    for name, score in result["scores"].items():
        print(f"  {name:<22} {score:.2f} s")
    monthly = result.get("monthly_A_descriptive", {})
    if monthly.get("mean") is not None:
        print(f"  monthly A RMSE mean {monthly['mean']:.2f} s, sd {_fmt(monthly['sd'], 0)} (descriptive)")
    if result.get("aging"):
        print("\nLonger-gap minus next-month RMSE on the same validation flights:")
        for name, item in result["aging"].items():
            print(f"  {name:<22} {item['difference']:+.2f} s  95% [{item['ci95'][0]:+.2f}, {item['ci95'][1]:+.2f}]"
                  f"  {item['decision']}")
    if slices:
        for group in ["R_A", "R_B_long", "R_A_partial", "R_B_long_partial"]:
            if group in result["groups"]:
                print_slices(group, result["groups"][group])
    if "comparison" in result:
        comparison = result["comparison"]
        print(f"\nCandidate minus baseline {comparison['baseline_id']} (negative favours the candidate):")
        for name, item in comparison["scores"].items():
            by_seed = " ".join(f"{v:+.1f}" for v in item["difference_by_seed"].values())
            print(f"  {name:<22} {item['difference']:+.2f} s  95% [{item['ci95'][0]:+.2f}, {item['ci95'][1]:+.2f}]"
                  f"  {item['decision']}  by seed: {by_seed}")
        for name, item in comparison["aging_changes"].items():
            print(f"  aging {name:<16} {item['difference']:+.2f} s  95% [{item['ci95'][0]:+.2f}, {item['ci95'][1]:+.2f}]")
        for group, airports in comparison["airports"].items():
            opposing = [f"{ap} {v['difference']:+.0f}" for ap, v in airports.items() if v["opposes_group"]]
            print(f"  {group} airports opposing the headline: {', '.join(opposing) or 'none'}")
