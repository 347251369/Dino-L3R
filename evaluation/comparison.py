"""Aggregate fixed-case Dice and surface reports for GSP and OCRE."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from datasets.specs import active_dataset_spec

SMALL_ORGANS = set(active_dataset_spec().small_organs)
ALL_ORGANS = set(range(1, 14))


def read_csv(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize(dice_path: str, surface_path: str) -> dict:
    dice_rows = read_csv(dice_path)
    surface_rows = read_csv(surface_path)
    if not dice_rows or not surface_rows:
        raise RuntimeError(f"Empty report: dice={dice_path} surface={surface_path}")

    def mean_dice(organ_ids: set[int]) -> float:
        values = [
            float(row["dice"])
            for row in dice_rows
            if int(row["class_id"]) in organ_ids
        ]
        return float(np.mean(values))

    per_organ = {}
    for organ_id in sorted(ALL_ORGANS):
        selected = [row for row in dice_rows if int(row["class_id"]) == organ_id]
        per_organ[str(organ_id)] = {
            "organ": selected[0]["organ"],
            "dice": float(np.mean([float(row["dice"]) for row in selected])),
        }

    assd = np.asarray([float(row["assd_mm"]) for row in surface_rows], dtype=np.float64)
    hd95 = np.asarray([float(row["hd95_mm"]) for row in surface_rows], dtype=np.float64)
    return {
        "overall": mean_dice(ALL_ORGANS),
        "small": mean_dice(SMALL_ORGANS),
        "large": mean_dice(ALL_ORGANS - SMALL_ORGANS),
        "assd_mm": float(np.nanmean(assd)),
        "hd95_mm": float(np.nanmean(hd95)),
        "fp_voxels": int(sum(int(float(row["fp_voxels"])) for row in surface_rows)),
        "fn_voxels": int(sum(int(float(row["fn_voxels"])) for row in surface_rows)),
        "per_organ": per_organ,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        nargs=3,
        action="append",
        metavar=("NAME", "DICE_PER_CASE", "SURFACE_PER_CASE"),
        required=True,
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--scope",
        default="fixed five cases; this test set had been viewed previously",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    methods = {
        name: summarize(dice_path, surface_path)
        for name, dice_path, surface_path in args.method
    }
    if args.baseline not in methods:
        raise KeyError(f"Baseline {args.baseline!r} is not one of {sorted(methods)}")
    baseline = methods[args.baseline]
    for name, metrics in methods.items():
        metrics["delta_vs_baseline"] = {
            key: metrics[key] - baseline[key]
            for key in ("overall", "small", "large", "assd_mm", "hd95_mm")
        }
    report = {
        "scope": args.scope,
        "baseline": args.baseline,
        "methods": methods,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(parse_args())
