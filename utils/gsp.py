from __future__ import annotations

import glob
import os
import re
from typing import Dict, Optional, Sequence

import numpy as np

from datasets.specs import active_dataset_spec, organ_name_map

try:
    from scipy.ndimage import binary_closing, binary_fill_holes, generate_binary_structure, label
except Exception:
    binary_closing = binary_fill_holes = generate_binary_structure = label = None


NUM_CLASSES = 14
_SPEC = active_dataset_spec()
HARD_ORGANS = _SPEC.hard_organs
COMPACT_ORGANS = _SPEC.compact_organs
ORGAN_NAMES = organ_name_map(_SPEC)


def nifti_to_dhw(array_xyz: np.ndarray) -> np.ndarray:
    if array_xyz.ndim != 3:
        raise ValueError(f"Expected a 3D array, got {array_xyz.shape}")
    return np.transpose(array_xyz, (2, 0, 1))


def clean_case_name(path_or_name: str) -> str:
    name = os.path.basename(path_or_name)
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    for prefix in ("windowed_", "raw_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    suffixes = ("_final_pred", "_pred", "_seg", "_label", "_0000")
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                changed = True
                break
    btcv_match = re.fullmatch(r"(?:img|label|case)(\d+)", name, re.IGNORECASE)
    if btcv_match:
        return btcv_match.group(1)
    return name


def index_nifti(directory: str) -> Dict[str, str]:
    files = sorted(glob.glob(os.path.join(directory, "*.nii")) + glob.glob(os.path.join(directory, "*.nii.gz")))
    return {clean_case_name(path): path for path in files}


def find_matching_file(path: str, directory: str, must_exist: bool = True) -> Optional[str]:
    key = clean_case_name(path)
    match = index_nifti(directory).get(key)
    if match is not None:
        return match
    if must_exist:
        raise FileNotFoundError(f"Could not find case {key} in {directory}")
    return None


def dice_score(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    pred_count, target_count = int(prediction.sum()), int(target.sum())
    if pred_count == 0 and target_count == 0:
        return 1.0
    if pred_count == 0 or target_count == 0:
        return 0.0
    intersection = np.logical_and(prediction, target).sum()
    return float(2.0 * intersection / (pred_count + target_count + 1e-8))


def close_and_fill(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    if binary_closing is None:
        return mask
    structure = generate_binary_structure(3, 1)
    closed = binary_closing(mask.astype(bool), structure=structure, iterations=iterations)
    return binary_fill_holes(closed)


def keep_largest_components(
    prediction: np.ndarray,
    compact_classes: Sequence[int] = (2, 3, 5, 6, 7, 8, 13),
    max_components: int = 1,
) -> np.ndarray:
    if label is None:
        return prediction.astype(np.uint8)
    output = prediction.copy()
    for class_id in compact_classes:
        mask = output == class_id
        if not mask.any():
            continue
        components, count = label(mask)
        if count <= max_components:
            continue
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        keep = np.argsort(sizes)[-max_components:]
        output[mask & ~np.isin(components, keep)] = 0
    return output.astype(np.uint8)
