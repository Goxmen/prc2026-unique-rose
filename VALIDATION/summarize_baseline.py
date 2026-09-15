"""Plot baseline training curves and compare saved 800-round predictions.

Uses an already completed run; never fits or uploads a model.
"""

import argparse
import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from records import read_run, write_json
from reporting import report, print_report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", type=int, default=800)
    args = ap.parse_args()
    record, directory = read_run(args.run)
    if record["status"] != "complete" or args.checkpoint not in record["config"]["checkpoints"]:
        raise ValueError("Expected a completed run with the requested prediction checkpoint")
    checkpoint = copy.deepcopy(record)
    checkpoint.update(run_id=record["run_id"] + f"_round{args.checkpoint}_checkpoint",
                      prediction_column=f"pred_round_{args.checkpoint}")
    checkpoint["config"]["rounds"] = args.checkpoint
    comparison = report(record, directory, (checkpoint, directory))
    write_json(directory / f"comparison_round{args.checkpoint}.json", comparison)
    print_report(comparison)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    fig.subplots_adjust(left=0.065, right=0.97, top=0.85, bottom=0.12, hspace=0.36, wspace=0.23)
    fig.suptitle(f"B0 training loss · {record['config']['rounds']:,} rounds", x=0.065,
                 ha="left", fontsize=20, weight="bold")
    fig.text(0.065, 0.91, "Six chronological training windows · three seeds · recovery off")
    for ax, (model_id, info) in zip(axes.flat, record["models"].items()):
        for seed in record["config"]["seeds"]:
            details = info["seeds"][str(seed)]
            curve = json.loads((directory / details["training_loss"]).read_text())["rmse"]
            ax.plot(range(1, len(curve) + 1), curve, label=f"Seed {seed}", linewidth=1.15)
        ax.axvline(args.checkpoint, color="#898781", linestyle="--", linewidth=0.8)
        ax.set_title(f"{model_id}: {info['train_start'][:7]} to {info['train_end'][:7]}", loc="left", fontsize=11)
        ax.grid(alpha=0.20)
        ax.set_ylabel("Training RMSE (s)")
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("Boosting round")
    axes[0, 0].legend(fontsize=8)
    fig.text(0.065, 0.035, f"Dashed line: {args.checkpoint} rounds. These curves measure training fit; "
             "held-out scores determine whether additional rounds help.", fontsize=10, color="#52514e")
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"training_loss.{suffix}", dpi=160)
    plt.close(fig)
    print(f"Saved training_loss.png/pdf and comparison_round{args.checkpoint}.json in {directory}")


if __name__ == "__main__":
    main()
