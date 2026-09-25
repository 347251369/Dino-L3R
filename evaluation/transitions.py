from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from utils.physical import ALL_ORGANS, ORGAN_NAMES, index_nifti, load_nifti_dhw


def physical_gradient_magnitude(image: np.ndarray, spacing) -> np.ndarray:
    smoothed = ndimage.gaussian_filter(image.astype(np.float32), sigma=(0.5, 0.75, 0.75))
    gradients = np.gradient(smoothed, *spacing, edge_order=1)
    return np.sqrt(sum(value * value for value in gradients), dtype=np.float32)


def multiclass_boundary(labels: np.ndarray) -> np.ndarray:
    boundary = np.zeros(labels.shape, dtype=bool)
    for axis in range(3):
        left = [slice(None)] * 3
        right = [slice(None)] * 3
        left[axis] = slice(None, -1)
        right[axis] = slice(1, None)
        difference = labels[tuple(left)] != labels[tuple(right)]
        boundary[tuple(left)] |= difference
        boundary[tuple(right)] |= difference
    return boundary


def event_stats(mask: np.ndarray, gradient: np.ndarray, distance: np.ndarray) -> dict:
    values = gradient[mask]
    distances = distance[mask]
    return {
        "count": int(mask.sum()),
        "gradient_median_hu_per_mm": float(np.median(values)) if values.size else 0.0,
        "gradient_q90_hu_per_mm": float(np.quantile(values, 0.90)) if values.size else 0.0,
        "distance_median_mm": float(np.median(distances)) if distances.size else 0.0,
        "distance_q95_mm": float(np.quantile(distances, 0.95)) if distances.size else 0.0,
    }


def distance_bin_summary(event_distances: dict[str, np.ndarray]) -> list[dict]:
    edges = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, np.inf)
    rows = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        counts = {}
        for name, values in event_distances.items():
            selected = (values >= lower) & (values < upper)
            counts[name] = int(selected.sum())
        total = sum(counts.values())
        rows.append(
            {
                "range_mm": f"[{lower:g},{upper:g})" if np.isfinite(upper) else f"[{lower:g},inf)",
                **counts,
                "total": total,
                "fixed_fraction": float(counts["fixed"] / total) if total else 0.0,
                "harmful_fraction": float(
                    (counts["broken"] + counts["wrong_to_wrong"]) / total
                ) if total else 0.0,
            }
        )
    return rows


def main(args: argparse.Namespace) -> None:
    images = index_nifti(args.images)
    labels = index_nifti(args.labels)
    baseline = index_nifti(args.baseline)
    candidate = index_nifti(args.candidate)
    common = sorted(set(images) & set(labels) & set(baseline) & set(candidate))
    if args.case_ids:
        requested = set(args.case_ids)
        common = [
            case_id
            for case_id in common
            if case_id in requested or any(case_id.endswith(value) for value in requested)
        ]
    if not common:
        raise RuntimeError("No common cases found")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    event_values = {name: {"gradient": [], "distance": []} for name in ("fixed", "broken", "wrong_to_wrong")}
    event_counts = {name: 0 for name in event_values}
    organ_rows = []

    for case_index, case_id in enumerate(common, 1):
        image_nii = nib.load(images[case_id])
        spacing_xyz = image_nii.header.get_zooms()[:3]
        spacing = (float(spacing_xyz[2]), float(spacing_xyz[0]), float(spacing_xyz[1]))
        image = load_nifti_dhw(images[case_id], np.float32)
        truth = load_nifti_dhw(labels[case_id], np.uint8)
        base = load_nifti_dhw(baseline[case_id], np.uint8)
        refined = load_nifti_dhw(candidate[case_id], np.uint8)
        if not (image.shape == truth.shape == base.shape == refined.shape):
            raise ValueError(f"Shape mismatch for {case_id}")

        gradient = physical_gradient_magnitude(image, spacing)
        distance = ndimage.distance_transform_edt(~multiclass_boundary(base), sampling=spacing)
        changed = base != refined
        base_correct = base == truth
        refined_correct = refined == truth
        events = {
            "fixed": changed & ~base_correct & refined_correct,
            "broken": changed & base_correct & ~refined_correct,
            "wrong_to_wrong": changed & ~base_correct & ~refined_correct,
        }
        for name, mask in events.items():
            event_counts[name] += int(mask.sum())
            event_values[name]["gradient"].append(gradient[mask])
            event_values[name]["distance"].append(distance[mask])

        for organ_id in ALL_ORGANS:
            base_binary = base == organ_id
            refined_binary = refined == organ_id
            truth_binary = truth == organ_id
            organ_changed = base_binary != refined_binary
            fixed = organ_changed & (base_binary != truth_binary) & (refined_binary == truth_binary)
            broken = organ_changed & (base_binary == truth_binary) & (refined_binary != truth_binary)
            organ_rows.append(
                {
                    "case_id": case_id,
                    "organ_id": organ_id,
                    "organ": ORGAN_NAMES[organ_id],
                    "changed": int(organ_changed.sum()),
                    "fixed": int(fixed.sum()),
                    "broken": int(broken.sum()),
                    "net": int(fixed.sum() - broken.sum()),
                    "fixed_gradient_median_hu_per_mm": event_stats(fixed, gradient, distance)["gradient_median_hu_per_mm"],
                    "broken_gradient_median_hu_per_mm": event_stats(broken, gradient, distance)["gradient_median_hu_per_mm"],
                }
            )
        print(f"[Transition audit] {case_index}/{len(common)} {case_id}", flush=True)

    event_summary = {}
    concatenated_distances = {}
    for name, values in event_values.items():
        gradient = np.concatenate(values["gradient"]) if any(v.size for v in values["gradient"]) else np.empty(0)
        distance = np.concatenate(values["distance"]) if any(v.size for v in values["distance"]) else np.empty(0)
        concatenated_distances[name] = distance
        event_summary[name] = {
            "count": event_counts[name],
            "gradient_median_hu_per_mm": float(np.median(gradient)) if gradient.size else 0.0,
            "gradient_q90_hu_per_mm": float(np.quantile(gradient, 0.90)) if gradient.size else 0.0,
            "distance_median_mm": float(np.median(distance)) if distance.size else 0.0,
            "distance_q95_mm": float(np.quantile(distance, 0.95)) if distance.size else 0.0,
        }

    organ_summary = []
    for organ_id in ALL_ORGANS:
        selected = [row for row in organ_rows if row["organ_id"] == organ_id]
        organ_summary.append(
            {
                "organ_id": organ_id,
                "organ": ORGAN_NAMES[organ_id],
                "changed": sum(row["changed"] for row in selected),
                "fixed": sum(row["fixed"] for row in selected),
                "broken": sum(row["broken"] for row in selected),
                "net": sum(row["net"] for row in selected),
            }
        )
    summary = {
        "scope": args.scope,
        "cases": common,
        "events": event_summary,
        "distance_bins_from_gsp_boundary": distance_bin_summary(
            concatenated_distances
        ),
        "per_organ": organ_summary,
    }
    (output / "transition_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output / "transition_per_case_organ.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(organ_rows[0]))
        writer.writeheader()
        writer.writerows(organ_rows)
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case_ids", nargs="*", default=())
    parser.add_argument("--scope", default="OCRE transition audit")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
