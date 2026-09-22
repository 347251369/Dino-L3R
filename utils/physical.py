"""Shared physical-space data, ROI, distance-field, and inference utilities."""
from __future__ import annotations

import glob
import math
import os
import re
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from datasets.specs import active_dataset_spec, organ_name_map

try:
    from scipy import ndimage
except Exception:
    ndimage = None


NUM_CLASSES = 14
_SPEC = active_dataset_spec()
TARGET_ORGANS = tuple(sorted(_SPEC.small_organs))
ORGAN_TO_INDEX = {organ_id: index for index, organ_id in enumerate(TARGET_ORGANS)}
ALL_ORGANS = tuple(range(1, NUM_CLASSES))
ALL_ORGAN_TO_INDEX = {organ_id: index for index, organ_id in enumerate(ALL_ORGANS)}
ORGAN_NAMES = organ_name_map(_SPEC)
BBox = Tuple[int, int, int, int, int, int]


def ensure_tuple3(values: Sequence[int]) -> Tuple[int, int, int]:
    if len(values) != 3:
        raise ValueError(f"Expected three values, got {values}")
    return int(values[0]), int(values[1]), int(values[2])


def nifti_to_dhw(array_xyz: np.ndarray) -> np.ndarray:
    if array_xyz.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI array, got {array_xyz.shape}")
    return np.transpose(array_xyz, (2, 0, 1))


def dhw_to_nifti(array_dhw: np.ndarray) -> np.ndarray:
    if array_dhw.ndim != 3:
        raise ValueError(f"Expected a 3D DHW array, got {array_dhw.shape}")
    return np.transpose(array_dhw, (1, 2, 0))


def normalize_ct(image: np.ndarray, hu_min: float = -200.0, hu_max: float = 300.0) -> np.ndarray:
    image = np.nan_to_num(image.astype(np.float32, copy=False), nan=hu_min, posinf=hu_max, neginf=hu_min)
    image = np.clip(image, hu_min, hu_max)
    return (2.0 * (image - hu_min) / (hu_max - hu_min) - 1.0).astype(np.float32)


def list_nifti(directory: str) -> List[str]:
    return sorted(glob.glob(os.path.join(directory, "*.nii")) + glob.glob(os.path.join(directory, "*.nii.gz")))


def clean_case_name(path_or_name: str) -> str:
    name = os.path.basename(path_or_name)
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    prefixes = ("windowed_", "raw_")
    suffixes = ("_final_pred", "_pred", "_seg", "_label", "_0000")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                changed = True
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                changed = True
    btcv_match = re.fullmatch(r"(?:img|label|case)(\d+)", name, re.IGNORECASE)
    if btcv_match:
        return btcv_match.group(1)
    return name


def index_nifti(directory: str) -> Dict[str, str]:
    return {clean_case_name(path): path for path in list_nifti(directory)}


def paired_case_paths(image_dir: str, label_dir: str, stage1_dir: str) -> List[Tuple[str, str, str, str]]:
    labels = index_nifti(label_dir)
    stage1 = index_nifti(stage1_dir)
    records = []
    for image_path in list_nifti(image_dir):
        case_id = clean_case_name(image_path)
        if case_id not in labels or case_id not in stage1:
            raise FileNotFoundError(
                f"Missing paired data for {case_id}: label={case_id in labels}, stage1={case_id in stage1}"
            )
        records.append((case_id, image_path, labels[case_id], stage1[case_id]))
    if not records:
        raise RuntimeError(f"No paired NIfTI cases found in {image_dir}")
    return records


def center_to_bbox(center: Sequence[int], size: Sequence[int], shape: Sequence[int]) -> BBox:
    size = ensure_tuple3(size)
    shape = ensure_tuple3(shape)

    def bounds(c: int, extent: int, full: int) -> Tuple[int, int]:
        if extent >= full:
            return 0, full
        start = int(round(c - extent / 2.0))
        start = min(max(start, 0), full - extent)
        return start, start + extent

    z1, z2 = bounds(int(center[0]), size[0], shape[0])
    h1, h2 = bounds(int(center[1]), size[1], shape[1])
    w1, w2 = bounds(int(center[2]), size[2], shape[2])
    return z1, z2, h1, h2, w1, w2


def bbox_center(bbox: BBox) -> Tuple[int, int, int]:
    return (
        (bbox[0] + bbox[1]) // 2,
        (bbox[2] + bbox[3]) // 2,
        (bbox[4] + bbox[5]) // 2,
    )


def bbox_from_mask(mask: np.ndarray, pad: Sequence[int], min_size: Sequence[int]) -> Optional[BBox]:
    coords = np.where(mask)
    if coords[0].size == 0:
        return None
    pad = ensure_tuple3(pad)
    mins = [int(coords[i].min()) - pad[i] for i in range(3)]
    maxs = [int(coords[i].max()) + 1 + pad[i] for i in range(3)]
    center = [(mins[i] + maxs[i]) // 2 for i in range(3)]
    extent = [max(int(min_size[i]), maxs[i] - mins[i]) for i in range(3)]
    return center_to_bbox(center, extent, mask.shape)


def fallback_bbox(shape: Sequence[int], organ_id: int, patch_size: Sequence[int]) -> BBox:
    shape = ensure_tuple3(shape)
    rel = _SPEC.relative_centers[int(organ_id)]
    return center_to_bbox([round(rel[i] * (shape[i] - 1)) for i in range(3)], patch_size, shape)


def gsp_soft_proposal_mask(
    class_probabilities: np.ndarray,
    organ_id: int,
    presence_probabilities: np.ndarray | None = None,
    probability_threshold: float = 0.05,
    presence_threshold: float = 0.50,
    top_k: int = 2,
) -> np.ndarray:
    """Build a conservative organ proposal from the frozen GSP soft field.

    The proposal is used only to decide which OCRE patches to inspect. The
    probabilities themselves remain continuous OCRE input channels.
    """
    probabilities = np.asarray(class_probabilities)
    if probabilities.ndim != 4 or probabilities.shape[0] != NUM_CLASSES:
        raise ValueError(
            "GSP probabilities must have shape "
            f"({NUM_CLASSES},D,H,W), got {probabilities.shape}"
        )
    if not 0.0 <= probability_threshold <= 1.0:
        raise ValueError("probability_threshold must be in [0, 1]")
    if not 0.0 <= presence_threshold <= 1.0:
        raise ValueError("presence_threshold must be in [0, 1]")
    if not 1 <= int(top_k) <= NUM_CLASSES:
        raise ValueError(f"top_k must be in [1, {NUM_CLASSES}]")

    target = probabilities[int(organ_id)]
    rank = np.ones(target.shape, dtype=np.uint8)
    for class_index in range(NUM_CLASSES):
        if class_index != int(organ_id):
            rank += probabilities[class_index] > target
    proposal = (rank <= int(top_k)) & (target >= float(probability_threshold))
    if presence_probabilities is not None:
        presence = np.asarray(presence_probabilities)
        expected = (NUM_CLASSES - 1, probabilities.shape[1])
        if presence.shape != expected:
            raise ValueError(
                f"GSP presence must have shape {expected}, got {presence.shape}"
            )
        active_slices = presence[int(organ_id) - 1] >= float(presence_threshold)
        proposal &= active_slices[:, None, None]
    return proposal


def prediction_roi_bbox(
    stage1: np.ndarray,
    organ_id: int,
    patch_size: Sequence[int],
    proposal_mask: np.ndarray | None = None,
) -> BBox:
    support = stage1 == int(organ_id)
    if proposal_mask is not None:
        proposal = np.asarray(proposal_mask, dtype=bool)
        if proposal.shape != stage1.shape:
            raise ValueError(
                f"Proposal shape mismatch: {proposal.shape} vs {stage1.shape}"
            )
        support = support | proposal
    bbox = bbox_from_mask(
        support, _SPEC.roi_padding[int(organ_id)], patch_size
    )
    return bbox if bbox is not None else fallback_bbox(stage1.shape, organ_id, patch_size)


def random_bbox_from_roi(
    roi: BBox,
    shape: Sequence[int],
    patch_size: Sequence[int],
    rng: np.random.Generator,
) -> BBox:
    """Draw a training patch from a precomputed prediction-derived ROI."""
    patch_size = ensure_tuple3(patch_size)
    ranges = ((roi[0], roi[1]), (roi[2], roi[3]), (roi[4], roi[5]))
    center = []
    jitter = (4, 12, 12)
    for axis, (start, stop) in enumerate(ranges):
        low = min(stop - 1, start + patch_size[axis] // 4)
        high = max(low + 1, stop - patch_size[axis] // 4)
        center.append(int(rng.integers(low, high)) + int(rng.integers(-jitter[axis], jitter[axis] + 1)))
    return center_to_bbox(center, patch_size, shape)


def _axis_starts(start: int, stop: int, patch: int, full: int, overlap: float) -> List[int]:
    if patch >= full:
        return [0]
    first = min(max(start, 0), full - patch)
    last = min(max(stop - patch, 0), full - patch)
    if last <= first:
        return [min(max((start + stop - patch) // 2, 0), full - patch)]
    stride = max(1, int(round(patch * (1.0 - overlap))))
    starts = list(range(first, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def covering_bboxes(
    stage1: np.ndarray,
    organ_id: int,
    patch_size: Sequence[int],
    max_patches: int = 12,
    overlap: float = 0.50,
    proposal_mask: np.ndarray | None = None,
) -> List[BBox]:
    patch_size = ensure_tuple3(patch_size)
    roi = prediction_roi_bbox(
        stage1, organ_id, patch_size, proposal_mask=proposal_mask
    )
    z_starts = _axis_starts(roi[0], roi[1], patch_size[0], stage1.shape[0], overlap)
    h_starts = _axis_starts(roi[2], roi[3], patch_size[1], stage1.shape[1], overlap)
    w_starts = _axis_starts(roi[4], roi[5], patch_size[2], stage1.shape[2], overlap)
    center = np.asarray(bbox_center(roi), dtype=np.float32)
    candidates = []
    for z1 in z_starts:
        for h1 in h_starts:
            for w1 in w_starts:
                bbox = (
                    z1,
                    min(z1 + patch_size[0], stage1.shape[0]),
                    h1,
                    min(h1 + patch_size[1], stage1.shape[1]),
                    w1,
                    min(w1 + patch_size[2], stage1.shape[2]),
                )
                distance = float(np.linalg.norm(np.asarray(bbox_center(bbox)) - center))
                candidates.append((distance, bbox))
    candidates.sort(key=lambda item: item[0])
    return [bbox for _, bbox in candidates[: max(1, int(max_patches))]]


def crop_dhw(array: np.ndarray, bbox: BBox) -> np.ndarray:
    return array[bbox[0] : bbox[1], bbox[2] : bbox[3], bbox[4] : bbox[5]]


def physical_signed_distance_channel(
    target_prior: np.ndarray,
    spacing_dhw: Sequence[float],
    clip_distance_mm: float = 24.0,
) -> np.ndarray:
    """Return a signed Euclidean distance field in millimetres, normalized to [-1, 1]."""
    target_prior = target_prior.astype(bool, copy=False)
    spacing = tuple(float(value) for value in spacing_dhw)
    if len(spacing) != 3 or min(spacing) <= 0:
        raise ValueError(f"Invalid DHW spacing: {spacing_dhw}")
    if ndimage is None:
        return target_prior.astype(np.float32) * 2.0 - 1.0
    inside = (
        ndimage.distance_transform_edt(target_prior, sampling=spacing)
        if target_prior.any()
        else np.zeros_like(target_prior, dtype=np.float32)
    )
    outside = (
        ndimage.distance_transform_edt(~target_prior, sampling=spacing)
        if (~target_prior).any()
        else np.zeros_like(target_prior, dtype=np.float32)
    )
    signed_mm = np.clip(inside - outside, -clip_distance_mm, clip_distance_mm)
    return (signed_mm / float(clip_distance_mm)).astype(np.float32)


def physical_sdf_crop(
    full_mask: np.ndarray,
    bbox: BBox,
    spacing_dhw: Sequence[float],
    clip_distance_mm: float = 24.0,
) -> np.ndarray:
    """Compute an SDF with a physical halo so patch edges are not treated as anatomy boundaries."""
    spacing = tuple(float(value) for value in spacing_dhw)
    halo = [int(np.ceil(clip_distance_mm / value)) for value in spacing]
    starts = (bbox[0], bbox[2], bbox[4])
    stops = (bbox[1], bbox[3], bbox[5])
    expanded_starts = [max(0, starts[axis] - halo[axis]) for axis in range(3)]
    expanded_stops = [min(full_mask.shape[axis], stops[axis] + halo[axis]) for axis in range(3)]
    expanded = full_mask[
        expanded_starts[0]:expanded_stops[0],
        expanded_starts[1]:expanded_stops[1],
        expanded_starts[2]:expanded_stops[2],
    ]
    expanded_sdf = physical_signed_distance_channel(expanded, spacing, clip_distance_mm)
    local_starts = [starts[axis] - expanded_starts[axis] for axis in range(3)]
    sizes = [stops[axis] - starts[axis] for axis in range(3)]
    return expanded_sdf[
        local_starts[0]:local_starts[0] + sizes[0],
        local_starts[1]:local_starts[1] + sizes[1],
        local_starts[2]:local_starts[2] + sizes[2],
    ]


def source_patch_size_for_target_spacing(
    target_shape: Sequence[int],
    target_spacing_dhw: Sequence[float],
    source_spacing_dhw: Sequence[float],
) -> Tuple[int, int, int]:
    """Convert a fixed target-grid patch to an equivalent source-grid crop."""
    shape = np.asarray(tuple(int(value) for value in target_shape), dtype=np.float64)
    target_spacing = np.asarray(target_spacing_dhw, dtype=np.float64)
    source_spacing = np.asarray(source_spacing_dhw, dtype=np.float64)
    if shape.size != 3 or target_spacing.size != 3 or source_spacing.size != 3:
        raise ValueError("Patch shape and spacings must contain three values")
    if np.any(target_spacing <= 0.0) or np.any(source_spacing <= 0.0):
        raise ValueError("Physical spacings must be positive")
    source_shape = np.maximum(1, np.rint(shape * target_spacing / source_spacing))
    return tuple(int(value) for value in source_shape)


def resample_dhw(
    array: np.ndarray,
    target_shape: Sequence[int],
    mode: str,
) -> np.ndarray:
    """Resample a DHW array with PyTorch's explicit interpolation semantics."""
    tensor = torch.from_numpy(np.asarray(array, dtype=np.float32).copy())[None, None]
    kwargs = {"size": ensure_tuple3(target_shape), "mode": mode}
    if mode != "nearest":
        kwargs["align_corners"] = False
    return F.interpolate(tensor, **kwargs)[0, 0].numpy()


def resample_dhw_crop(
    array: np.ndarray,
    bbox: BBox,
    target_shape: Sequence[int],
    mode: str,
) -> np.ndarray:
    return resample_dhw(crop_dhw(array, bbox), target_shape, mode)


def augment_normalized_ct(
    image: np.ndarray,
    rng: np.random.Generator | None,
) -> np.ndarray:
    """Apply conservative nnU-Net-style CT intensity perturbations."""
    image = image.astype(np.float32, copy=True)
    if rng is None:
        return image
    if rng.random() < 0.15:
        sigma = float(rng.uniform(0.0, 0.05))
        image += rng.normal(0.0, sigma, size=image.shape).astype(np.float32)
    if rng.random() < 0.15 and ndimage is not None:
        image = ndimage.gaussian_filter(
            image, sigma=float(rng.uniform(0.5, 1.0))
        ).astype(np.float32)
    if rng.random() < 0.20:
        mean = float(image.mean())
        image = (image - mean) * float(rng.uniform(0.75, 1.25)) + mean
    if rng.random() < 0.20:
        image *= float(rng.uniform(0.75, 1.25))
    if rng.random() < 0.15:
        gamma = float(rng.uniform(0.7, 1.5))
        unit = np.clip((image + 1.0) * 0.5, 0.0, 1.0)
        image = 2.0 * np.power(unit, gamma) - 1.0
    return np.clip(image, -1.5, 1.5).astype(np.float32)


def make_target_space_physical_refiner_input(
    image_normalized: np.ndarray,
    coarse_mask: np.ndarray,
    organ_id: int,
    bbox: BBox,
    source_spacing_dhw: Sequence[float],
    target_spacing_dhw: Sequence[float],
    target_shape: Sequence[int],
    clip_distance_mm: float = 24.0,
    rng: np.random.Generator | None = None,
    coarse_sdf: np.ndarray | None = None,
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """Build physical-refiner channels on one standardized target grid."""
    target_shape = ensure_tuple3(target_shape)
    source_spacing = tuple(float(value) for value in source_spacing_dhw)
    target_spacing = tuple(float(value) for value in target_spacing_dhw)
    source_shape = tuple(int(value) for value in crop_dhw(image_normalized, bbox).shape)

    image_patch = augment_normalized_ct(
        resample_dhw_crop(image_normalized, bbox, target_shape, "trilinear"), rng
    )
    coarse_patch = np.rint(
        resample_dhw_crop(coarse_mask, bbox, target_shape, "nearest")
    ).astype(np.int64)
    coarse_tensor = torch.from_numpy(
        np.clip(coarse_patch, 0, NUM_CLASSES - 1).copy()
    ).long()
    one_hot = F.one_hot(coarse_tensor, num_classes=NUM_CLASSES).permute(3, 0, 1, 2).float()
    distance_array = (
        resample_dhw_crop(coarse_sdf, bbox, target_shape, "trilinear")
        if coarse_sdf is not None
        else resample_dhw(
            physical_sdf_crop(
                coarse_mask == int(organ_id), bbox, source_spacing, clip_distance_mm
            ),
            target_shape,
            "trilinear",
        )
    )
    distance = torch.from_numpy(distance_array.astype(np.float32)).unsqueeze(0)

    starts = (int(bbox[0]), int(bbox[2]), int(bbox[4]))
    stops = (int(bbox[1]), int(bbox[3]), int(bbox[5]))
    coordinates = []
    for axis, (start, stop, extent, full_size, source_step) in enumerate(
        zip(starts, stops, target_shape, image_normalized.shape, source_spacing)
    ):
        source_extent = max(stop - start, 1)
        source_positions = start + (
            (np.arange(extent, dtype=np.float32) + 0.5)
            * (float(source_extent) / float(extent))
            - 0.5
        )
        center_mm = 0.5 * max(full_size - 1, 1) * source_step
        values = np.clip(
            (source_positions * source_step - center_mm) / 250.0, -1.5, 1.5
        ).astype(np.float32)
        shape = [1, 1, 1, 1]
        shape[axis + 1] = extent
        coordinates.append(
            torch.from_numpy(values).view(*shape).expand(1, *target_shape)
        )
    spacing_channels = [
        torch.full(
            (1, *target_shape), min(step / 5.0, 1.0), dtype=torch.float32
        )
        for step in target_spacing
    ]
    image_tensor = torch.from_numpy(image_patch).unsqueeze(0)
    x = torch.cat(
        (image_tensor, one_hot, distance, *coordinates, *spacing_channels), dim=0
    )
    return x, source_shape


def make_physical_refiner_input(
    image_normalized: np.ndarray,
    coarse_mask: np.ndarray,
    organ_id: int,
    bbox: BBox,
    spacing_dhw: Sequence[float],
    clip_distance_mm: float = 24.0,
    class_probabilities: np.ndarray | None = None,
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """Build CT, GSP anatomy, millimetre SDF, coordinates, and spacing.

    ``class_probabilities`` optionally replaces the 14 hard one-hot anatomy
    channels while preserving the established 22-channel layout.
    """
    image_patch = crop_dhw(image_normalized, bbox).astype(np.float32, copy=False)
    mask_patch = crop_dhw(coarse_mask, bbox).astype(np.int64, copy=False)
    original_shape = image_patch.shape
    spacing = tuple(float(value) for value in spacing_dhw)

    image_tensor = torch.from_numpy(image_patch.copy()).unsqueeze(0)
    if class_probabilities is None:
        label_tensor = torch.from_numpy(
            np.clip(mask_patch, 0, NUM_CLASSES - 1).copy()
        ).long()
        anatomy = F.one_hot(
            label_tensor, num_classes=NUM_CLASSES
        ).permute(3, 0, 1, 2).float()
    else:
        probabilities = np.asarray(class_probabilities)
        expected = (NUM_CLASSES, *coarse_mask.shape)
        if probabilities.shape != expected:
            raise ValueError(
                f"GSP probability shape mismatch: {probabilities.shape} vs {expected}"
            )
        probability_patch = probabilities[
            :, bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]
        ]
        anatomy = torch.from_numpy(
            np.asarray(probability_patch, dtype=np.float32).copy()
        ).clamp_(0.0, 1.0)
    distance = torch.from_numpy(
        physical_sdf_crop(coarse_mask == int(organ_id), bbox, spacing, clip_distance_mm)
    ).unsqueeze(0)

    z1, z2, h1, h2, w1, w2 = bbox
    full_shape = image_normalized.shape
    starts = (z1, h1, w1)
    stops = (z2, h2, w2)
    coordinates = []
    for axis, (start, stop, size, step) in enumerate(zip(starts, stops, full_shape, spacing)):
        center_mm = 0.5 * max(size - 1, 1) * step
        values_mm = torch.arange(start, stop, dtype=torch.float32) * step - center_mm
        values = (values_mm / 250.0).clamp(-1.5, 1.5)
        shape = [1, 1, 1, 1]
        shape[axis + 1] = len(values)
        expand_shape = [1, *original_shape]
        coordinates.append(values.view(*shape).expand(*expand_shape))
    spacing_channels = [
        torch.full((1, *original_shape), min(step / 5.0, 1.0), dtype=torch.float32)
        for step in spacing
    ]
    x = torch.cat((image_tensor, anatomy, distance, *coordinates, *spacing_channels), dim=0)
    pad = [(8 - dim % 8) % 8 for dim in x.shape[-3:]]
    if any(pad):
        x = F.pad(x, (0, pad[2], 0, pad[1], 0, pad[0]), value=0.0)
    return x, original_shape


def make_refiner_input(
    image_normalized: np.ndarray,
    stage1: np.ndarray,
    organ_id: int,
    bbox: BBox,
    image_patch_override: Optional[np.ndarray] = None,
    stage1_patch_override: Optional[np.ndarray] = None,
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    image_patch = (
        crop_dhw(image_normalized, bbox)
        if image_patch_override is None
        else image_patch_override
    ).astype(np.float32, copy=False)
    stage1_patch = (
        crop_dhw(stage1, bbox)
        if stage1_patch_override is None
        else stage1_patch_override
    ).astype(np.int64, copy=False)
    original_shape = image_patch.shape
    image_tensor = torch.from_numpy(image_patch.copy()).unsqueeze(0)
    label_tensor = torch.from_numpy(np.clip(stage1_patch, 0, NUM_CLASSES - 1).copy()).long()
    one_hot = F.one_hot(label_tensor, num_classes=NUM_CLASSES).permute(3, 0, 1, 2).float()
    distance = torch.from_numpy(signed_distance_channel(stage1_patch == int(organ_id))).unsqueeze(0)

    z1, z2, h1, h2, w1, w2 = bbox
    full_d, full_h, full_w = image_normalized.shape
    z = torch.linspace(z1 / max(full_d - 1, 1), (z2 - 1) / max(full_d - 1, 1), original_shape[0])
    h = torch.linspace(h1 / max(full_h - 1, 1), (h2 - 1) / max(full_h - 1, 1), original_shape[1])
    w = torch.linspace(w1 / max(full_w - 1, 1), (w2 - 1) / max(full_w - 1, 1), original_shape[2])
    coords = (
        z.view(1, -1, 1, 1).expand(1, *original_shape) * 2.0 - 1.0,
        h.view(1, 1, -1, 1).expand(1, *original_shape) * 2.0 - 1.0,
        w.view(1, 1, 1, -1).expand(1, *original_shape) * 2.0 - 1.0,
    )
    x = torch.cat((image_tensor, one_hot, distance, *coords), dim=0)
    pad = [(8 - dim % 8) % 8 for dim in x.shape[-3:]]
    if any(pad):
        x = F.pad(x, (0, pad[2], 0, pad[1], 0, pad[0]), value=0.0)
    return x, original_shape


@lru_cache(maxsize=8)
def gaussian_importance_map(shape: Tuple[int, int, int]) -> np.ndarray:
    axes = []
    for size in shape:
        coord = np.linspace(-1.0, 1.0, size, dtype=np.float32)
        axes.append(np.exp(-0.5 * (coord / 0.5) ** 2))
    weight = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    return np.maximum(weight / weight.max(), 0.05).astype(np.float32)


def infer_organ_probability(
    model: torch.nn.Module,
    image_normalized: np.ndarray,
    stage1: np.ndarray,
    organ_id: int,
    device: torch.device,
    patch_size: Sequence[int],
    max_patches: int,
    use_amp: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    probability_sum = np.zeros(stage1.shape, dtype=np.float32)
    weight_sum = np.zeros(stage1.shape, dtype=np.float32)
    for bbox in covering_bboxes(stage1, organ_id, patch_size, max_patches=max_patches):
        x, original_shape = make_refiner_input(image_normalized, stage1, organ_id, bbox)
        x = x.unsqueeze(0).to(device, non_blocking=True)
        organ_index = torch.tensor([ORGAN_TO_INDEX[int(organ_id)]], device=device)
        with torch.inference_mode(), torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x, organ_index, return_aux=False)
            probability = torch.sigmoid(logits)[0, 0].float().cpu().numpy()
        od, oh, ow = original_shape
        probability = probability[:od, :oh, :ow]
        weight = gaussian_importance_map(original_shape)
        z1, z2, h1, h2, w1, w2 = bbox
        probability_sum[z1:z2, h1:h2, w1:w2] += probability * weight
        weight_sum[z1:z2, h1:h2, w1:w2] += weight
    covered = weight_sum > 0
    probability = np.zeros(stage1.shape, dtype=np.float32)
    probability[covered] = probability_sum[covered] / weight_sum[covered]
    return probability, covered


def refine_case(
    model: torch.nn.Module,
    image_normalized: np.ndarray,
    stage1: np.ndarray,
    device: torch.device,
    patch_size: Sequence[int],
    target_organs: Sequence[int] = TARGET_ORGANS,
    max_patches: int = 12,
    threshold: float = 0.50,
    prior_bonus: float = 0.15,
    use_amp: bool = True,
    allow_protected_overwrite: bool = False,
    protected_threshold: float = 0.80,
) -> np.ndarray:
    best_score = np.full(stage1.shape, -np.inf, dtype=np.float32)
    best_organ = np.zeros(stage1.shape, dtype=np.uint8)
    union_covered = np.zeros(stage1.shape, dtype=bool)

    for organ_id in target_organs:
        organ_id = int(organ_id)
        probability, covered = infer_organ_probability(
            model,
            image_normalized,
            stage1,
            organ_id,
            device,
            patch_size,
            max_patches,
            use_amp,
        )
        score = probability + float(prior_bonus) * (stage1 == organ_id)
        update = covered & (score > best_score)
        best_score[update] = score[update]
        best_organ[update] = organ_id
        union_covered |= covered

    final = stage1.copy()
    target_mask = np.isin(stage1, list(target_organs))
    final[target_mask & union_covered] = 0
    safe_region = (stage1 == 0) | target_mask
    accept = union_covered & safe_region & (best_score >= float(threshold))
    if allow_protected_overwrite:
        accept |= union_covered & ~safe_region & (best_score >= float(protected_threshold))
    final[accept] = best_organ[accept]
    return final.astype(np.uint8)


def dice_score(prediction: np.ndarray, target: np.ndarray, class_id: int) -> float:
    pred = prediction == int(class_id)
    gt = target == int(class_id)
    denominator = int(pred.sum()) + int(gt.sum())
    if denominator == 0:
        return float("nan")
    return float(2.0 * np.logical_and(pred, gt).sum() / denominator)


def dice_report(prediction: np.ndarray, target: np.ndarray) -> Dict[int, float]:
    return {class_id: dice_score(prediction, target, class_id) for class_id in range(1, NUM_CLASSES)}


def load_nifti_dhw(path: str, dtype: np.dtype) -> np.ndarray:
    nii = nib.load(path)
    return nifti_to_dhw(np.asanyarray(nii.dataobj).astype(dtype, copy=False))
