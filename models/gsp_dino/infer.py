import argparse
import glob
import hashlib
import json
import os
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Iterable, Iterator

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from transformers import CLIPTokenizer

from .model import CTSegModel
from .semantics import ANATOMY_PROMPTS
from .train import (
    NUM_CLASSES,
    fixed_view_windows,
    list_nifti,
    normalize_ct,
    normalize_model_input,
    postprocess,
)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_name(image_path: str) -> str:
    name = os.path.basename(image_path)
    if name.endswith(".nii.gz"):
        return name[:-7] + "_pred.nii.gz"
    if name.endswith(".nii"):
        return name[:-4] + "_pred.nii"
    raise ValueError(f"Unsupported image name: {name}")


def case_id(image_path: str) -> str:
    name = os.path.basename(image_path)
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    if name.endswith("_0000"):
        name = name[:-5]
    return name


def atomic_save_npy(path: str, array: np.ndarray) -> None:
    temporary = f"{path}.tmp.npy"
    np.save(temporary, array)
    os.replace(temporary, path)


def load_case_data(item: tuple[int, str]) -> tuple[int, str, np.ndarray, np.ndarray, nib.Nifti1Header, float]:
    index, image_path = item
    image_nii = nib.load(image_path)
    image = image_nii.get_fdata(dtype=np.float32)
    if image.shape[:2] != (512, 512):
        raise RuntimeError(f"Expected 512x512 axial slices, got {image.shape}: {image_path}")
    return (
        index,
        image_path,
        image,
        np.asarray(image_nii.affine).copy(),
        image_nii.header.copy(),
        float(image_nii.header.get_zooms()[2]),
    )


def iter_prefetched_cases(
    items: Iterable[tuple[int, str]],
    executor: ThreadPoolExecutor | None,
    prefetch_cases: int,
) -> Iterator[tuple[int, str, np.ndarray, np.ndarray, nib.Nifti1Header, float]]:
    if executor is None or prefetch_cases <= 1:
        for item in items:
            yield load_case_data(item)
        return

    source = iter(items)
    pending: deque[Future] = deque()
    for _ in range(prefetch_cases):
        try:
            pending.append(executor.submit(load_case_data, next(source)))
        except StopIteration:
            break
    while pending:
        case = pending.popleft().result()
        try:
            pending.append(executor.submit(load_case_data, next(source)))
        except StopIteration:
            pass
        yield case


def save_case_artifacts(
    out_path: str,
    prediction: np.ndarray,
    affine: np.ndarray,
    header: nib.Nifti1Header,
    save_hard: bool,
    probability_path: str,
    probabilities: np.ndarray | None,
    presence_path: str,
    presence: np.ndarray | None,
) -> None:
    if save_hard:
        nib.save(
            nib.Nifti1Image(prediction.astype(np.uint8), affine, header),
            out_path,
        )
    if probability_path:
        if probabilities is None or presence is None:
            raise RuntimeError("Soft output paths require probabilities and presence")
        atomic_save_npy(probability_path, probabilities)
        atomic_save_npy(presence_path, presence)


def load_model(args: argparse.Namespace, device: torch.device) -> CTSegModel:
    model = CTSegModel(
        pretrained_weights=args.dino_weights,
        num_classes=NUM_CLASSES,
        min_load_ratio=1.0,
        allow_partial_backbone=False,
        use_depth_context_adapter=args.depth_context_adapter,
        use_presence_head=args.presence_head,
        use_spatial_view_context=args.spatial_view_context,
        use_deep_supervision=args.deep_supervision,
        trainable_block_start=args.trainable_block_start,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)

    tokenizer = CLIPTokenizer.from_pretrained(args.clip_dir)
    tokenized = tokenizer(
        ANATOMY_PROMPTS,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )
    model.cache_anatomy_semantics(
        tokenized["input_ids"].to(device),
        tokenized["attention_mask"].to(device),
    )
    model.eval()
    return model


@torch.inference_mode()
def predict_case(
    model: CTSegModel,
    image: np.ndarray,
    device: torch.device,
    batch_size: int,
    use_amp: bool,
    smooth_sigma: float,
    spacing_z: float,
    context_mm: float,
    view_mode: str,
    local_view_size: int,
    spatial_view_context: bool,
    parallel_views: bool = False,
    return_soft: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Match FullViewSliceDataset: normalize once, then retain a float16 volume.
    image = normalize_ct(image, -200.0, 300.0)
    image = image.astype(np.float16)
    height, width, depth = image.shape
    probabilities = np.empty((NUM_CLASSES, height, width, depth), dtype=np.float16)
    presence = (
        np.empty((NUM_CLASSES - 1, depth), dtype=np.float16)
        if return_soft
        else None
    )
    windows = fixed_view_windows(view_mode, local_view_size)
    blend_weights = {}
    for _, _, size in windows:
        if size == 512:
            blend_weights[size] = torch.ones((1, 1, size, size), device=device)
        else:
            axis = torch.hann_window(size, periodic=False, device=device)
            blend_weights[size] = torch.outer(axis, axis).clamp_min(0.05)[None, None]

    for start in range(0, depth, batch_size):
        z_values = list(range(start, min(start + batch_size, depth)))
        full_slices = []
        for z in z_values:
            offset = 1 if context_mm <= 0.0 else max(
                1, int(round(context_mm / max(spacing_z, 1e-6)))
            )
            z0, z2 = max(z - offset, 0), min(z + offset, depth - 1)
            full_slices.append(np.stack([
                image[:, :, z0],
                image[:, :, z],
                image[:, :, z2],
            ], axis=0))
        full_slices = np.stack(full_slices).astype(np.float32)
        probability_sum = torch.zeros(
            (len(z_values), NUM_CLASSES, height, width),
            dtype=torch.float32,
            device=device,
        )
        weight_sum = torch.zeros(
            (1, 1, height, width),
            dtype=torch.float32,
            device=device,
        )
        prepared_views = []
        prepared_contexts = []
        if parallel_views and len(windows) > 1:
            for y0, x0, size in windows:
                view = torch.from_numpy(np.ascontiguousarray(
                    full_slices[:, :, y0:y0 + size, x0:x0 + size]
                )).to(device, non_blocking=True)
                if size != 512:
                    view = F.interpolate(
                        view,
                        size=(512, 512),
                        mode="bilinear",
                        align_corners=False,
                    )
                prepared_views.append(normalize_model_input(view))
                contexts = []
                for z in z_values:
                    z_value = z / max(depth - 1, 1)
                    if spatial_view_context:
                        contexts.append([
                            z_value,
                            (y0 + size / 2.0) / 512.0,
                            (x0 + size / 2.0) / 512.0,
                            size / 512.0,
                        ])
                    else:
                        contexts.append([z_value])
                prepared_contexts.append(torch.tensor(
                    contexts,
                    dtype=torch.float32,
                    device=device,
                ))

            combined_view = torch.cat(prepared_views, dim=0)
            combined_context = torch.cat(prepared_contexts, dim=0)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                if return_soft:
                    combined_logits, combined_presence = model(
                        combined_view,
                        input_ids=None,
                        attention_mask=None,
                        z_depths=combined_context,
                        return_aux=True,
                    )
                    if combined_presence is None:
                        raise RuntimeError(
                            "Soft GSP export requires a checkpoint with the presence head"
                        )
                else:
                    combined_logits = model(
                        combined_view,
                        input_ids=None,
                        attention_mask=None,
                        z_depths=combined_context,
                    )

            window_batch = len(z_values)
            for window_index, (y0, x0, size) in enumerate(windows):
                lower = window_index * window_batch
                upper = lower + window_batch
                logits = combined_logits[lower:upper]
                if return_soft and size == 512:
                    presence[:, start:start + window_batch] = (
                        torch.sigmoid(combined_presence[lower:upper])
                        .float()
                        .cpu()
                        .numpy()
                        .T.astype(np.float16)
                    )
                probs = F.softmax(logits, dim=1).float()
                if size != 512:
                    probs = F.interpolate(
                        probs,
                        size=(size, size),
                        mode="bilinear",
                        align_corners=False,
                    )
                weight = blend_weights[size]
                probability_sum[:, :, y0:y0 + size, x0:x0 + size] += probs * weight
                weight_sum[:, :, y0:y0 + size, x0:x0 + size] += weight
        else:
            for y0, x0, size in windows:
                view = torch.from_numpy(np.ascontiguousarray(
                    full_slices[:, :, y0:y0 + size, x0:x0 + size]
                )).to(device, non_blocking=True)
                if size != 512:
                    view = F.interpolate(
                        view,
                        size=(512, 512),
                        mode="bilinear",
                        align_corners=False,
                    )
                view = normalize_model_input(view)
                contexts = []
                for z in z_values:
                    z_value = z / max(depth - 1, 1)
                    if spatial_view_context:
                        contexts.append([
                            z_value,
                            (y0 + size / 2.0) / 512.0,
                            (x0 + size / 2.0) / 512.0,
                            size / 512.0,
                        ])
                    else:
                        contexts.append([z_value])
                context_tensor = torch.tensor(
                    contexts,
                    dtype=torch.float32,
                    device=device,
                )
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    if return_soft and size == 512:
                        logits, presence_logits = model(
                            view,
                            input_ids=None,
                            attention_mask=None,
                            z_depths=context_tensor,
                            return_aux=True,
                        )
                        if presence_logits is None:
                            raise RuntimeError(
                                "Soft GSP export requires a checkpoint with the presence head"
                            )
                        presence[:, start:start + len(z_values)] = (
                            torch.sigmoid(presence_logits)
                            .float()
                            .cpu()
                            .numpy()
                            .T.astype(np.float16)
                        )
                    else:
                        logits = model(
                            view,
                            input_ids=None,
                            attention_mask=None,
                            z_depths=context_tensor,
                        )
                probs = F.softmax(logits, dim=1).float()
                if size != 512:
                    probs = F.interpolate(
                        probs,
                        size=(size, size),
                        mode="bilinear",
                        align_corners=False,
                    )
                weight = blend_weights[size]
                probability_sum[:, :, y0:y0 + size, x0:x0 + size] += probs * weight
                weight_sum[:, :, y0:y0 + size, x0:x0 + size] += weight
        probs = (probability_sum / weight_sum.clamp_min(1e-6)).cpu().numpy()
        probabilities[:, :, :, start:start + len(z_values)] = np.transpose(
            probs, (1, 2, 3, 0)
        ).astype(np.float16)

    if smooth_sigma > 0:
        probabilities = gaussian_filter1d(
            probabilities.astype(np.float32),
            sigma=smooth_sigma,
            axis=3,
        )
    hard = postprocess(np.argmax(probabilities, axis=0).astype(np.uint8))
    if not return_soft:
        return hard
    soft_dhw = np.transpose(
        probabilities.astype(np.float16, copy=False), (0, 3, 1, 2)
    )
    return hard, soft_dhw, presence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal Stage1 prediction")
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--soft_out_dir",
        default="",
        help="Optional directory for full float16 GSP probabilities and presence.",
    )
    parser.add_argument(
        "--soft_only",
        action="store_true",
        help="Reuse and verify existing hard predictions while exporting soft fields.",
    )
    parser.add_argument(
        "--resume_soft",
        action="store_true",
        help="Reuse complete per-case soft files after an interrupted export.",
    )
    parser.add_argument("--max_hard_mismatch_voxels", type=int, default=128)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected_sha256", required=True)
    parser.add_argument("--dino_weights", default="/pd/heyang/weights/model.safetensors")
    parser.add_argument("--clip_dir", default="/pd/heyang/weights/clip-vit-base-patch32")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--parallel_views",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run all fixed views for one slice batch in a single model forward pass.",
    )
    parser.add_argument("--io_workers", type=int, default=0)
    parser.add_argument("--prefetch_cases", type=int, default=2)
    parser.add_argument("--save_queue_depth", type=int, default=2)
    parser.add_argument("--smooth_sigma", type=float, default=1.0)
    parser.add_argument("--context_mm", type=float, default=0.0)
    parser.add_argument(
        "--view_mode",
        choices=("full", "fixed_multiview"),
        default="full",
    )
    parser.add_argument("--local_view_size", type=int, default=320)
    parser.add_argument("--trainable_block_start", type=int, default=6)
    parser.add_argument(
        "--depth_context_adapter",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--presence_head",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--deep_supervision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Construct decoder auxiliary heads when they are present in the checkpoint.",
    )
    parser.add_argument(
        "--spatial_view_context",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if args.soft_only and not args.soft_out_dir:
        raise ValueError("--soft_only requires --soft_out_dir")
    if args.soft_out_dir and not args.presence_head:
        raise ValueError("GSP soft export requires --presence_head")
    if args.view_mode == "fixed_multiview" and not args.spatial_view_context:
        raise ValueError(
            "fixed_multiview requires --spatial_view_context so local views retain "
            "their absolute axial position."
        )
    if args.batch_size < 1:
        raise ValueError("--batch_size must be positive")
    if args.io_workers < 0:
        raise ValueError("--io_workers cannot be negative")
    if args.prefetch_cases < 1 or args.save_queue_depth < 1:
        raise ValueError("--prefetch_cases and --save_queue_depth must be positive")
    actual_sha256 = sha256_file(args.checkpoint)
    if actual_sha256 != args.expected_sha256:
        raise RuntimeError(
            f"Checkpoint SHA-256 mismatch: actual={actual_sha256}, "
            f"expected={args.expected_sha256}"
        )

    image_paths = list_nifti(args.images_dir)
    if not image_paths:
        raise RuntimeError(f"No NIfTI images found in {args.images_dir}")
    os.makedirs(args.out_dir, exist_ok=True)
    existing = list_nifti(args.out_dir)
    if existing and not args.soft_only:
        raise RuntimeError(f"Output directory must be empty: {args.out_dir}")
    if args.soft_only and len(existing) != len(image_paths):
        raise RuntimeError(
            f"Soft-only export requires {len(image_paths)} locked hard predictions, "
            f"found {len(existing)} in {args.out_dir}"
        )
    if args.soft_out_dir:
        os.makedirs(args.soft_out_dir, exist_ok=True)
        existing_soft = glob.glob(os.path.join(args.soft_out_dir, "*_gsp_*.npy"))
        if existing_soft and not args.resume_soft:
            raise RuntimeError(
                f"Soft output directory must be empty: {args.soft_out_dir}"
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    model = load_model(args, device)

    manifest = {
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": actual_sha256,
        "images_dir": args.images_dir,
        "num_cases": len(image_paths),
        "input": "raw3[z-1,z,z+1]",
        "anatomy_semantics": "fixed_graph_prompts",
        "hu_window": [-200.0, 300.0],
        "normalized_cache_dtype": "float16",
        "image_size": [512, 512],
        "depth": "z/(Z-1)",
        "batch_size": args.batch_size,
        "effective_view_batch_size": args.batch_size * (
            len(fixed_view_windows(args.view_mode, args.local_view_size))
            if args.parallel_views
            else 1
        ),
        "parallel_views": args.parallel_views,
        "io_workers": args.io_workers,
        "prefetch_cases": args.prefetch_cases,
        "save_queue_depth": args.save_queue_depth,
        "amp": use_amp,
        "smooth_sigma_z": args.smooth_sigma,
        "context_mm": args.context_mm,
        "view_mode": args.view_mode,
        "local_view_size": args.local_view_size,
        "views_per_slice": len(fixed_view_windows(args.view_mode, args.local_view_size)),
        "fusion": "probability_hann_weighted_in_original_coordinates",
        "depth_context_adapter": args.depth_context_adapter,
        "spatial_view_context": args.spatial_view_context,
        "presence_head": args.presence_head,
        "deep_supervision": args.deep_supervision,
        "trainable_block_start": args.trainable_block_start,
        "postprocess": "gsp_dino_connected_component_and_hole_repair",
    }
    if not args.soft_only:
        with open(os.path.join(args.out_dir, "prediction_manifest.json"), "w") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
    soft_manifest = None
    if args.soft_out_dir:
        soft_manifest = {
            **manifest,
            "format": {
                "probabilities": "float16 numpy (14,D,H,W)",
                "presence": "float16 numpy (13,D)",
                "class_axis": "background_then_13_BTCV_organs",
            },
            "soft_only": bool(args.soft_only),
            "hard_verification": {},
        }
    print(json.dumps(manifest, indent=2, sort_keys=True))

    case_items = []
    for index, image_path in enumerate(image_paths, start=1):
        identifier = case_id(image_path)
        probability_path = (
            os.path.join(args.soft_out_dir, f"{identifier}_gsp_probabilities.npy")
            if args.soft_out_dir
            else ""
        )
        presence_path = (
            os.path.join(args.soft_out_dir, f"{identifier}_gsp_presence.npy")
            if args.soft_out_dir
            else ""
        )
        if (
            args.resume_soft
            and probability_path
            and os.path.exists(probability_path)
            and os.path.exists(presence_path)
        ):
            print(f"[Soft GSP] {index}/{len(image_paths)} reused {identifier}")
            continue
        case_items.append((index, image_path))

    io_pool = (
        ThreadPoolExecutor(max_workers=args.io_workers, thread_name_prefix="gsp-io")
        if args.io_workers > 0
        else None
    )
    save_futures: deque[tuple[int, str, str, tuple[int, ...] | None, tuple[int, ...] | None, Future]] = deque()

    def finish_save() -> None:
        index, identifier, out_path, probability_shape, presence_shape, future = save_futures.popleft()
        future.result()
        if not args.soft_only:
            print(f"[Predict] {index}/{len(image_paths)} saved {out_path}")
        if probability_shape is not None:
            print(
                f"[Soft GSP] {index}/{len(image_paths)} saved {identifier} "
                f"probabilities={probability_shape} presence={presence_shape}"
            )

    try:
        loaded_cases = iter_prefetched_cases(
            case_items,
            io_pool,
            args.prefetch_cases,
        )
        for index, image_path, image, affine, header, spacing_z in loaded_cases:
            identifier = case_id(image_path)
            probability_path = (
                os.path.join(args.soft_out_dir, f"{identifier}_gsp_probabilities.npy")
                if args.soft_out_dir
                else ""
            )
            presence_path = (
                os.path.join(args.soft_out_dir, f"{identifier}_gsp_presence.npy")
                if args.soft_out_dir
                else ""
            )
            result = predict_case(
                model,
                image,
                device,
                args.batch_size,
                use_amp,
                args.smooth_sigma,
                spacing_z,
                args.context_mm,
                args.view_mode,
                args.local_view_size,
                args.spatial_view_context,
                parallel_views=args.parallel_views,
                return_soft=bool(args.soft_out_dir),
            )
            if args.soft_out_dir:
                prediction, probabilities, presence = result
            else:
                prediction = result
                probabilities = presence = None
            out_path = os.path.join(args.out_dir, output_name(image_path))
            if args.soft_only:
                locked = np.asarray(nib.load(out_path).dataobj, dtype=np.uint8)
                mismatch = int(np.count_nonzero(locked != prediction))
                soft_manifest["hard_verification"][identifier] = mismatch
                if mismatch > args.max_hard_mismatch_voxels:
                    raise RuntimeError(
                        f"Hard prediction drift for {identifier}: {mismatch} voxels "
                        f"> {args.max_hard_mismatch_voxels}"
                    )

            if io_pool is None:
                save_case_artifacts(
                    out_path,
                    prediction,
                    affine,
                    header,
                    not args.soft_only,
                    probability_path,
                    probabilities,
                    presence_path,
                    presence,
                )
                if not args.soft_only:
                    print(f"[Predict] {index}/{len(image_paths)} saved {out_path}")
                if probabilities is not None:
                    print(
                        f"[Soft GSP] {index}/{len(image_paths)} saved {identifier} "
                        f"probabilities={probabilities.shape} presence={presence.shape}"
                    )
                continue

            while len(save_futures) >= args.save_queue_depth:
                finish_save()
            future = io_pool.submit(
                save_case_artifacts,
                out_path,
                prediction,
                affine,
                header,
                not args.soft_only,
                probability_path,
                probabilities,
                presence_path,
                presence,
            )
            save_futures.append((
                index,
                identifier,
                out_path,
                probabilities.shape if probabilities is not None else None,
                presence.shape if presence is not None else None,
                future,
            ))
        while save_futures:
            finish_save()
    finally:
        if io_pool is not None:
            io_pool.shutdown(wait=True)
    if soft_manifest is not None:
        manifest_path = os.path.join(args.soft_out_dir, "soft_prediction_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(soft_manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")


if __name__ == "__main__":
    main(parse_args())
