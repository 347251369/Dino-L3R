"""Batched organ-level inference for OCRE."""
from __future__ import annotations

import argparse
import glob
import os

import nibabel as nib
import numpy as np
import torch

from models.ocre import build_ocre_from_checkpoint
from utils.physical import (
    ALL_ORGAN_TO_INDEX,
    ALL_ORGANS,
    clean_case_name,
    covering_bboxes,
    gaussian_importance_map,
    load_nifti_dhw,
    make_physical_refiner_input,
    normalize_ct,
)

OCRE_MODEL = "ocre"


def configure_deterministic_inference() -> None:
    torch.manual_seed(542)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(542)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def infer_organ(
    model,
    model_type: str,
    image: np.ndarray,
    coarse: np.ndarray,
    spacing_dhw,
    organ_id: int,
    device: torch.device,
    args,
    **_unused,
):
    """Blend overlapping OCRE patches for one target organ."""
    if model_type != OCRE_MODEL:
        raise ValueError(f"Unsupported refinement model: {model_type}")
    output_sum = np.zeros((1, *coarse.shape), dtype=np.float32)
    weight_sum = np.zeros(coarse.shape, dtype=np.float32)
    pending = []

    def flush_batch() -> None:
        if not pending:
            return
        batch = torch.stack([item[0] for item in pending]).to(device)
        organ_index = torch.full(
            (len(pending),),
            ALL_ORGAN_TO_INDEX[organ_id],
            dtype=torch.long,
            device=device,
        )
        with torch.inference_mode(), torch.amp.autocast(
            device_type=device.type,
            enabled=args.amp and device.type == "cuda",
        ):
            values = torch.sigmoid(model(batch, organ_index))
        values = values.float()
        for value, (_, original_shape, bbox) in zip(values, pending):
            depth, height, width = original_shape
            value = value[:, :depth, :height, :width]
            value = value.cpu().numpy()
            weight = gaussian_importance_map(original_shape)
            z1, z2, h1, h2, w1, w2 = bbox
            output_sum[:, z1:z2, h1:h2, w1:w2] += value * weight[None]
            weight_sum[z1:z2, h1:h2, w1:w2] += weight
        pending.clear()

    for bbox in covering_bboxes(
        coarse, organ_id, args.patch_size, max_patches=args.max_patches
    ):
        x, original_shape = make_physical_refiner_input(
            image, coarse, organ_id, bbox, spacing_dhw
        )
        if pending and pending[0][0].shape != x.shape:
            flush_batch()
        pending.append((x, original_shape, bbox))
        if len(pending) >= args.inference_batch_size:
            flush_batch()
    flush_batch()
    covered = weight_sum > 0
    output = np.zeros_like(output_sum)
    output[:, covered] = output_sum[:, covered] / weight_sum[covered]
    return output, covered


def refine_with_ocre(
    model,
    image,
    coarse,
    spacing_dhw,
    device,
    args,
):
    best_probability = np.zeros(coarse.shape, dtype=np.float32)
    best_organ = np.zeros(coarse.shape, dtype=np.uint8)
    covered_union = np.zeros(coarse.shape, dtype=bool)
    for organ_id in ALL_ORGANS:
        output, covered = infer_organ(
            model, OCRE_MODEL, image, coarse, spacing_dhw,
            organ_id, device, args,
        )
        probability = output[0]
        update = covered & (probability > best_probability)
        best_probability[update] = probability[update]
        best_organ[update] = organ_id
        covered_union |= covered
    refined = coarse.copy()
    refined[(coarse > 0) & covered_union] = 0
    accepted = covered_union & (best_probability >= args.threshold)
    refined[accepted] = best_organ[accepted]
    return refined.astype(np.uint8)


def _index_files(directory: str) -> dict[str, str]:
    return {
        clean_case_name(path): path
        for path in glob.glob(os.path.join(directory, "*.nii.gz"))
    }


def _resolve_case(index: dict[str, str], case_id: str, kind: str) -> str:
    candidates = [path for key, path in index.items() if key == case_id or case_id in key]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one {kind} for {case_id}, found {candidates}")
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_type", choices=(OCRE_MODEL,), default=OCRE_MODEL)
    parser.add_argument("--images", required=True)
    parser.add_argument("--coarse_masks", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--patch_size", nargs=3, type=int, default=(64, 144, 144))
    parser.add_argument("--max_patches", type=int, default=32)
    parser.add_argument("--inference_batch_size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--case_ids", nargs="*", default=())
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    configure_deterministic_inference()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model, checkpoint_layout = build_ocre_from_checkpoint(checkpoint)
    model = model.to(device)
    model.eval()
    print(f"[OCRE] checkpoint layout: {checkpoint_layout}", flush=True)
    os.makedirs(args.output, exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(args.images, "*.nii.gz")))
    if args.case_ids:
        image_paths = [
            path for path in image_paths
            if any(case_id in os.path.basename(path) for case_id in args.case_ids)
        ]
    coarse_index = _index_files(args.coarse_masks)
    for index, image_path in enumerate(image_paths, 1):
        case_id = clean_case_name(image_path)
        nii = nib.load(image_path)
        spacing_xyz = tuple(float(value) for value in nii.header.get_zooms()[:3])
        spacing_dhw = (spacing_xyz[2], spacing_xyz[0], spacing_xyz[1])
        image = normalize_ct(load_nifti_dhw(image_path, np.float32))
        coarse = load_nifti_dhw(
            _resolve_case(coarse_index, case_id, "GSP mask"), np.uint8
        )
        refined = refine_with_ocre(
            model, image, coarse, spacing_dhw, device, args
        )
        output_path = os.path.join(args.output, f"{case_id}_0000_pred.nii.gz")
        nib.save(
            nib.Nifti1Image(np.transpose(refined, (1, 2, 0)), nii.affine, nii.header),
            output_path,
        )
        print(f"[OCRE] {index}/{len(image_paths)} {output_path}", flush=True)


if __name__ == "__main__":
    main(parse_args())
