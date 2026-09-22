"""Compute per-case and per-organ Dice scores for NIfTI predictions."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np

from utils.gsp import NUM_CLASSES, ORGAN_NAMES, dice_score, find_matching_file, nifti_to_dhw

PER_CASE_FIELDS = (
    "case",
    "class_id",
    "organ",
    "dice",
    "gt_voxels",
    "pred_voxels",
    "gt_present",
    "pred_present",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="Directory containing predicted masks.")
    parser.add_argument("--labels", required=True, help="Directory containing ground-truth masks.")
    parser.add_argument(
        "--references",
        help="Optional image directory used only to define the expected case list.",
    )
    parser.add_argument("--per_case_csv", required=True)
    parser.add_argument("--summary_csv", required=True)
    return parser.parse_args()


def list_nifti(directory: Path) -> list[Path]:
    return sorted((*directory.glob("*.nii"), *directory.glob("*.nii.gz")))


def load_mask(path: Path) -> np.ndarray:
    array = np.asanyarray(nib.load(path).dataobj).astype(np.uint8)
    return nifti_to_dhw(array)


def mean_or_nan(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def write_csv(path: Path, fieldnames: Iterable[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def collect_case_rows(
    predictions: Path,
    labels: Path,
    references: Path | None,
) -> tuple[list[dict], int]:
    reference_files = list_nifti(references or predictions)
    if not reference_files:
        raise RuntimeError(f"No NIfTI files found in {references or predictions}")

    rows: list[dict] = []
    matched_cases = 0
    for reference in reference_files:
        try:
            prediction_path = Path(find_matching_file(str(reference), str(predictions), must_exist=True))
            label_path = Path(find_matching_file(str(reference), str(labels), must_exist=True))
        except FileNotFoundError as error:
            print(f"[skip] {error}")
            continue

        prediction = load_mask(prediction_path)
        label = load_mask(label_path)
        if prediction.shape != label.shape:
            print(
                f"[skip] shape mismatch for {reference.name}: "
                f"prediction={prediction.shape}, label={label.shape}"
            )
            continue

        matched_cases += 1
        for class_id in range(1, NUM_CLASSES):
            predicted = prediction == class_id
            expected = label == class_id
            pred_voxels = int(predicted.sum())
            gt_voxels = int(expected.sum())
            dice = dice_score(predicted, expected)
            rows.append(
                {
                    "case": reference.name,
                    "class_id": class_id,
                    "organ": ORGAN_NAMES.get(class_id, str(class_id)),
                    "dice": f"{dice:.6f}",
                    "_dice_value": dice,
                    "gt_voxels": gt_voxels,
                    "pred_voxels": pred_voxels,
                    "gt_present": int(gt_voxels > 0),
                    "pred_present": int(pred_voxels > 0),
                }
            )

    if matched_cases == 0:
        raise RuntimeError("No matching prediction and label pairs were evaluated.")
    return rows, matched_cases


def summarize(rows: list[dict], case_count: int) -> list[dict]:
    by_class: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_class[int(row["class_id"])].append(row)

    summary: list[dict] = []
    for class_id in range(1, NUM_CLASSES):
        class_rows = by_class[class_id]
        all_dice = [float(row["_dice_value"]) for row in class_rows]
        gt_dice = [float(row["_dice_value"]) for row in class_rows if row["gt_present"]]
        union_dice = [
            float(row["_dice_value"])
            for row in class_rows
            if row["gt_present"] or row["pred_present"]
        ]
        gt_voxels = sum(int(row["gt_voxels"]) for row in class_rows)
        pred_voxels = sum(int(row["pred_voxels"]) for row in class_rows)
        gt_mean = mean_or_nan(gt_dice)
        union_mean = mean_or_nan(union_dice)
        summary.append(
            {
                "class_id": class_id,
                "organ": ORGAN_NAMES.get(class_id, str(class_id)),
                "num_cases": case_count,
                "gt_present_cases": len(gt_dice),
                "dice_all_cases": f"{np.mean(all_dice):.6f}",
                "dice_gt_present": "nan" if np.isnan(gt_mean) else f"{gt_mean:.6f}",
                "dice_union_present": "nan" if np.isnan(union_mean) else f"{union_mean:.6f}",
                "gt_voxels_total": gt_voxels,
                "pred_voxels_total": pred_voxels,
                "pred_gt_voxel_ratio": f"{pred_voxels / max(gt_voxels, 1):.6f}",
            }
        )
    return summary


def print_summary(summary: list[dict], case_count: int) -> None:
    print(f"\nDice summary ({case_count} cases)")
    print("ID  Organ          GT+  Dice(all)  Dice(GT+)  Pred/GT")
    for row in summary:
        print(
            f"{row['class_id']:>2}  {row['organ']:<13} "
            f"{row['gt_present_cases']:>3}  {row['dice_all_cases']:>9}  "
            f"{row['dice_gt_present']:>9}  {row['pred_gt_voxel_ratio']:>7}"
        )


def evaluate(
    predictions: Path,
    labels: Path,
    references: Path | None,
    per_case_csv: Path,
    summary_csv: Path,
) -> None:
    rows, case_count = collect_case_rows(predictions, labels, references)
    summary = summarize(rows, case_count)
    write_csv(per_case_csv, PER_CASE_FIELDS, rows)
    write_csv(summary_csv, summary[0].keys(), summary)
    print_summary(summary, case_count)
    print(f"Per-case CSV: {per_case_csv}")
    print(f"Summary CSV:  {summary_csv}")


if __name__ == "__main__":
    arguments = parse_args()
    evaluate(
        Path(arguments.predictions),
        Path(arguments.labels),
        Path(arguments.references) if arguments.references else None,
        Path(arguments.per_case_csv),
        Path(arguments.summary_csv),
    )
