from __future__ import annotations

import argparse
import csv
import glob
import os

import nibabel as nib
import numpy as np
from scipy import ndimage

from utils.gsp import ORGAN_NAMES, find_matching_file, nifti_to_dhw


def load_dhw(path: str, dtype=None):
    nii = nib.load(path)
    array = nifti_to_dhw(np.asanyarray(nii.dataobj))
    spacing_xyz = nii.header.get_zooms()[:3]
    spacing_dhw = (float(spacing_xyz[2]), float(spacing_xyz[0]), float(spacing_xyz[1]))
    return array.astype(dtype) if dtype is not None else array, spacing_dhw


def surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask, iterations=1, border_value=0)


def crop_union(a: np.ndarray, b: np.ndarray, spacing, margin_mm: float = 8.0):
    points = np.argwhere(a | b)
    if not len(points):
        return (slice(0, 1),) * 3
    lower = points.min(axis=0)
    upper = points.max(axis=0) + 1
    margin = np.ceil(margin_mm / np.asarray(spacing)).astype(int)
    lower = np.maximum(lower - margin, 0)
    upper = np.minimum(upper + margin, a.shape)
    return tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))


def large_component_count(mask: np.ndarray, minimum_voxels: int = 100) -> int:
    labels, count = ndimage.label(mask)
    if not count:
        return 0
    sizes = np.bincount(labels.ravel())[1:]
    return int(np.count_nonzero(sizes >= minimum_voxels))


def analyze_case(pred_path: str, gt_path: str, image_path: str):
    pred, spacing = load_dhw(pred_path, np.uint8)
    gt, _ = load_dhw(gt_path, np.uint8)
    image, _ = load_dhw(image_path, np.float32)
    image_gradient = np.gradient(image, *spacing, edge_order=1)
    gradient_magnitude = np.sqrt(sum(value * value for value in image_gradient))
    case = os.path.basename(image_path).split("_0000")[0]
    rows = []
    for organ_id in range(1, 14):
        p_full, g_full = pred == organ_id, gt == organ_id
        roi = crop_union(p_full, g_full, spacing)
        p, g = p_full[roi], g_full[roi]
        fp, fn = p & ~g, g & ~p
        intersection = int(np.count_nonzero(p & g))
        p_count, g_count = int(p.sum()), int(g.sum())
        dice = (2.0 * intersection) / max(p_count + g_count, 1)
        p_surface, g_surface = surface(p), surface(g)

        if p_surface.any() and g_surface.any():
            distance_to_p = ndimage.distance_transform_edt(~p_surface, sampling=spacing)
            distance_to_g = ndimage.distance_transform_edt(~g_surface, sampling=spacing)
            surface_distances = np.concatenate((distance_to_g[p_surface], distance_to_p[g_surface]))
            assd = float(surface_distances.mean())
            hd95 = float(np.percentile(surface_distances, 95.0))
            fn_distance = distance_to_p[fn]
            fp_distance = distance_to_g[fp]
        else:
            assd = hd95 = float("nan")
            fn_distance = np.full(int(fn.sum()), np.inf)
            fp_distance = np.full(int(fp.sum()), np.inf)

        errors = int(fp.sum() + fn.sum())
        error_distances = np.concatenate((fp_distance, fn_distance))
        recoverable = {
            distance: float(np.mean(error_distances <= distance)) if errors else 1.0
            for distance in (2.0, 3.0, 5.0)
        }
        error_map = fp | fn
        z_counts = []
        if error_map.any():
            z_indices = np.argwhere(error_map)[:, 0]
            depth = max(error_map.shape[0], 1)
            z_counts = [float(np.mean((z_indices * 3 // depth) == part)) for part in range(3)]
        else:
            z_counts = [0.0, 0.0, 0.0]
        boundary_contrast = float(gradient_magnitude[roi][g_surface].mean()) if g_surface.any() else float("nan")
        rows.append(
            {
                "case": case,
                "class_id": organ_id,
                "organ": ORGAN_NAMES.get(organ_id, str(organ_id)),
                "dice": dice,
                "assd_mm": assd,
                "hd95_mm": hd95,
                "fp_voxels": int(fp.sum()),
                "fn_voxels": int(fn.sum()),
                "pred_gt_ratio": p_count / max(g_count, 1),
                "error_within_2mm": recoverable[2.0],
                "error_within_3mm": recoverable[3.0],
                "error_within_5mm": recoverable[5.0],
                "fp_components_ge100": large_component_count(fp),
                "fn_components_ge100": large_component_count(fn),
                "error_inferior": z_counts[0],
                "error_middle": z_counts[1],
                "error_superior": z_counts[2],
                "gt_surface_gradient_hu_per_mm": boundary_contrast,
            }
        )
    return rows


def main(args):
    rows = []
    images = sorted(glob.glob(os.path.join(args.images, "*.nii.gz")))
    if args.case_ids:
        images = [
            path
            for path in images
            if any(case_id in os.path.basename(path) for case_id in args.case_ids)
        ]
    if not images:
        raise RuntimeError("No images matched the requested case IDs")
    for index, image_path in enumerate(images, 1):
        pred_path = find_matching_file(image_path, args.predictions, must_exist=True)
        gt_path = find_matching_file(image_path, args.labels, must_exist=True)
        rows.extend(analyze_case(pred_path, gt_path, image_path))
        print(f"[Error analysis] {index}/{len(images)} {os.path.basename(image_path)}")

    os.makedirs(os.path.dirname(args.per_case_csv), exist_ok=True)
    with open(args.per_case_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for organ_id in range(1, 14):
        selected = [row for row in rows if row["class_id"] == organ_id]
        entry = {"class_id": organ_id, "organ": ORGAN_NAMES.get(organ_id, str(organ_id))}
        for key in list(rows[0])[3:]:
            values = np.asarray([row[key] for row in selected], dtype=np.float64)
            entry[key] = float(np.nanmean(values))
        summary.append(entry)
    with open(args.summary_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Per-case: {args.per_case_csv}")
    print(f"Summary : {args.summary_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--per_case_csv", required=True)
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--case_ids", nargs="*", default=())
    main(parser.parse_args())
