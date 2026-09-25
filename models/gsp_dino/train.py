import argparse
import csv
import glob
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterator, List, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.transforms import InterpolationMode
from transformers import CLIPTokenizer

from .model import CTSegModel
from .semantics import ANATOMY_PROMPTS, ORGAN_NAMES
from datasets.specs import active_dataset_spec
from utils.gsp import COMPACT_ORGANS, HARD_ORGANS, close_and_fill, keep_largest_components


NUM_CLASSES = 14
_SPEC = active_dataset_spec()
DIFFICULT_ORGANS = sorted(_SPEC.small_organs)
LARGE_ORGANS = sorted(set(range(1, NUM_CLASSES)) - set(DIFFICULT_ORGANS))
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def organ_name(class_id: int) -> str:
    if 0 <= class_id < len(ORGAN_NAMES):
        return str(ORGAN_NAMES[class_id])
    return str(class_id)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_ct(image: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    image = np.nan_to_num(image.astype(np.float32, copy=False), nan=hu_min, posinf=hu_max, neginf=hu_min)
    image = np.clip(image, hu_min, hu_max)
    return ((image - hu_min) / (hu_max - hu_min)).astype(np.float32)


def list_nifti(directory: str) -> List[str]:
    return sorted(glob.glob(os.path.join(directory, "*.nii")) + glob.glob(os.path.join(directory, "*.nii.gz")))


def case_key(path: str) -> str:
    name = os.path.basename(path)
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    if name.endswith("_0000"):
        name = name[:-5]
    return name


def pair_nifti(image_dir: str, label_dir: str) -> List[Tuple[str, str]]:
    images = {case_key(p): p for p in list_nifti(image_dir)}
    labels = {case_key(p): p for p in list_nifti(label_dir)}
    missing_labels = sorted(set(images) - set(labels))
    missing_images = sorted(set(labels) - set(images))
    if missing_labels or missing_images:
        raise RuntimeError(
            f"Image/label name mismatch. Missing labels={missing_labels[:5]}, "
            f"missing images={missing_images[:5]}"
        )
    return [(images[name], labels[name]) for name in sorted(images)]


@dataclass
class CachedVolume:
    name: str
    image: np.ndarray
    label: np.ndarray
    spacing_z: float


def fixed_view_windows(view_mode: str, local_view_size: int) -> List[Tuple[int, int, int]]:
    if view_mode not in {"full", "fixed_multiview"}:
        raise ValueError(f"Unknown view mode: {view_mode}")
    if not 1 <= int(local_view_size) <= 512:
        raise ValueError("local_view_size must be in [1, 512]")
    windows = [(0, 0, 512)]
    if view_mode == "fixed_multiview":
        edge = 512 - int(local_view_size)
        windows.extend([
            (0, 0, int(local_view_size)),
            (0, edge, int(local_view_size)),
            (edge, 0, int(local_view_size)),
            (edge, edge, int(local_view_size)),
        ])
    return windows


class FullViewSliceDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        hu_min: float = -200.0,
        hu_max: float = 300.0,
        augment: bool = True,
        balance_cap: float = 2.5,
        augmentation_profile: str = "standard",
        context_mm: float = 0.0,
        view_mode: str = "full",
        local_view_size: int = 320,
        spatial_view_context: bool = False,
    ):
        self.augment = bool(augment)
        self.augmentation_profile = str(augmentation_profile)
        self.context_mm = float(context_mm)
        self.view_mode = str(view_mode)
        self.local_view_size = int(local_view_size)
        self.spatial_view_context = bool(spatial_view_context)
        if self.augmentation_profile not in {"standard", "btcv_strong"}:
            raise ValueError(f"Unknown augmentation profile: {self.augmentation_profile}")
        self.view_windows = fixed_view_windows(self.view_mode, self.local_view_size)
        self.volumes: List[CachedVolume] = []
        self.samples: List[Tuple[int, int, int]] = []
        self.slice_organs: List[Tuple[int, ...]] = []
        self.organ_slice_counts = {c: 0 for c in range(1, NUM_CLASSES)}
        self.voxel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

        pairs = pair_nifti(image_dir, label_dir)
        print(f"[Data] Preloading {len(pairs)} image/label volumes into RAM...")
        for volume_index, (image_path, label_path) in enumerate(pairs):
            image_nii = nib.load(image_path)
            label_nii = nib.load(label_path)
            image = image_nii.get_fdata(dtype=np.float32)
            label = np.asanyarray(label_nii.dataobj).astype(np.uint8)
            if image.shape != label.shape:
                raise RuntimeError(f"Shape mismatch for {image_path}: image={image.shape}, label={label.shape}")
            if image.shape[:2] != (512, 512):
                raise RuntimeError(f"Expected 512x512 axial slices, got {image.shape} for {image_path}")

            image = normalize_ct(image, hu_min, hu_max)
            image = image.astype(np.float16)
            label = np.ascontiguousarray(label)
            self.voxel_counts += np.bincount(label.reshape(-1), minlength=NUM_CLASSES)[:NUM_CLASSES]
            spacing_z = float(image_nii.header.get_zooms()[2])
            self.volumes.append(
                CachedVolume(os.path.basename(image_path), image, label, spacing_z)
            )

            for z in range(label.shape[2]):
                for view_index, (y0, x0, size) in enumerate(self.view_windows):
                    view_label = label[y0:y0 + size, x0:x0 + size, z]
                    present = tuple(
                        int(c)
                        for c in np.unique(view_label)
                        if 0 < int(c) < NUM_CLASSES
                    )
                    self.samples.append((volume_index, z, view_index))
                    self.slice_organs.append(present)
                    for c in present:
                        self.organ_slice_counts[c] += 1

            print(f"[Data] cached {volume_index + 1:02d}/{len(pairs):02d}: {os.path.basename(image_path)} {image.shape}")

        max_count = max(self.organ_slice_counts.values())
        sample_weights = []
        for present in self.slice_organs:
            if not present:
                sample_weights.append(1.0)
                continue
            weight = max(math.sqrt(max_count / max(self.organ_slice_counts[c], 1)) for c in present)
            sample_weights.append(min(float(balance_cap), weight))
        self.sample_weights = torch.tensor(sample_weights, dtype=torch.double)

        negative = sum(1 for organs in self.slice_organs if not organs)
        memory_gb = sum(v.image.nbytes + v.label.nbytes for v in self.volumes) / (1024 ** 3)
        print(
            f"[Data] slices={len(self.samples)}, background_only={negative}, "
            f"views={len(self.view_windows)}, cache={memory_gb:.2f} GiB, "
            f"organ_slice_counts={self.organ_slice_counts}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def class_weights(self, background_weight: float = 0.25, cap: float = 2.5) -> torch.Tensor:
        foreground = self.voxel_counts[1:].astype(np.float64)
        reference = float(np.median(foreground))
        weights = np.ones(NUM_CLASSES, dtype=np.float32)
        weights[0] = float(background_weight)
        weights[1:] = np.clip(np.sqrt(reference / np.maximum(foreground, 1.0)), 0.5, float(cap))
        return torch.from_numpy(weights)

    def _augment(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        strong = self.augmentation_profile == "btcv_strong"
        if random.random() < (0.7 if strong else 0.5):
            angle = random.uniform(-12.0, 12.0) if strong else random.uniform(-7.0, 7.0)
            scale = random.uniform(0.90, 1.10) if strong else random.uniform(0.95, 1.05)
            translation = 0.05 if strong else 0.03
            max_dx = int(round(image.shape[-1] * translation))
            max_dy = int(round(image.shape[-2] * translation))
            translate = [random.randint(-max_dx, max_dx), random.randint(-max_dy, max_dy)]
            image = TF.affine(
                image,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )
            label = TF.affine(
                label.unsqueeze(0).float(),
                angle=angle,
                translate=translate,
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.NEAREST,
                fill=0.0,
            ).squeeze(0).long()

        if random.random() < (0.9 if strong else 0.7):
            gain = random.uniform(0.85, 1.15) if strong else random.uniform(0.95, 1.05)
            bias = random.uniform(-0.06, 0.06) if strong else random.uniform(-0.03, 0.03)
            image = image * gain + bias
        if strong and random.random() < 0.35:
            gamma = random.uniform(0.8, 1.25)
            image = image.clamp(0.0, 1.0).pow(gamma)
        if random.random() < (0.35 if strong else 0.2):
            upper = 0.02 if strong else 0.01
            image = image + torch.randn_like(image) * random.uniform(0.003, upper)
        if strong and random.random() < 0.20:
            kernel = random.choice((3, 5))
            image = TF.gaussian_blur(image, [kernel, kernel], [0.1, 1.0])
        return image.clamp_(0.0, 1.0), label

    def neighbor_offset(self, volume: CachedVolume) -> int:
        if self.context_mm <= 0.0:
            return 1
        return max(1, int(round(self.context_mm / max(volume.spacing_z, 1e-6))))

    def __getitem__(self, index: int):
        volume_index, z, view_index = self.samples[index]
        volume = self.volumes[volume_index]
        max_z = volume.image.shape[2] - 1
        offset = self.neighbor_offset(volume)
        z0, z2 = max(z - offset, 0), min(z + offset, max_z)
        image = np.stack(
            [volume.image[:, :, z0], volume.image[:, :, z], volume.image[:, :, z2]],
            axis=0,
        ).astype(np.float32, copy=False)
        label = volume.label[:, :, z]
        y0, x0, size = self.view_windows[view_index]
        image = image[:, y0:y0 + size, x0:x0 + size]
        label = label[y0:y0 + size, x0:x0 + size]
        image_t = torch.from_numpy(np.ascontiguousarray(image))
        label_t = torch.from_numpy(np.ascontiguousarray(label)).long()
        if size != 512:
            image_t = F.interpolate(
                image_t.unsqueeze(0),
                size=(512, 512),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            label_t = F.interpolate(
                label_t[None, None].float(),
                size=(512, 512),
                mode="nearest",
            )[0, 0].long()
        if self.augment:
            image_t, label_t = self._augment(image_t, label_t)
        image_t = (image_t - IMAGENET_MEAN) / IMAGENET_STD
        z_value = z / max(max_z, 1)
        if self.spatial_view_context:
            context = torch.tensor([
                z_value,
                (y0 + size / 2.0) / 512.0,
                (x0 + size / 2.0) / 512.0,
                size / 512.0,
            ], dtype=torch.float32)
        else:
            context = torch.tensor([z_value], dtype=torch.float32)
        return image_t, label_t, context


class OrganBalancedEpochSampler(Sampler[int]):

    def __init__(
        self,
        dataset: FullViewSliceDataset,
        num_samples: int,
        seed: int,
        organ_probability: float = 0.8,
    ):
        if not 0.0 <= organ_probability <= 1.0:
            raise ValueError("organ_probability must be in [0, 1]")
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.organ_probability = float(organ_probability)
        self.epoch = 0
        self.organ_ids = []
        self.organ_indices: Dict[int, torch.Tensor] = {}
        frequency_weights = []
        for class_id in range(1, NUM_CLASSES):
            indices = [
                index
                for index, present in enumerate(dataset.slice_organs)
                if class_id in present
            ]
            if not indices:
                continue
            self.organ_ids.append(class_id)
            self.organ_indices[class_id] = torch.tensor(indices, dtype=torch.long)
            frequency_weights.append(1.0 / math.sqrt(len(indices)))
        if not self.organ_ids:
            raise RuntimeError("Organ-balanced sampling found no foreground slices")
        weights = torch.tensor(frequency_weights, dtype=torch.double)
        self.organ_weights = weights / weights.sum()
        self.dataset_size = len(dataset)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        draws = []
        for _ in range(self.num_samples):
            if torch.rand((), generator=generator).item() < self.organ_probability:
                organ_position = int(torch.multinomial(
                    self.organ_weights,
                    1,
                    replacement=True,
                    generator=generator,
                ).item())
                class_id = self.organ_ids[organ_position]
                candidates = self.organ_indices[class_id]
                candidate_position = int(torch.randint(
                    len(candidates),
                    (1,),
                    generator=generator,
                ).item())
                draws.append(int(candidates[candidate_position]))
            else:
                draws.append(int(torch.randint(
                    self.dataset_size,
                    (1,),
                    generator=generator,
                ).item()))
        return iter(draws)

    def __len__(self) -> int:
        return self.num_samples


class CoverageBalancedEpochSampler(Sampler[int]):

    def __init__(
        self,
        dataset: FullViewSliceDataset,
        num_samples: int,
        seed: int,
    ):
        self.dataset_size = len(dataset)
        self.num_samples = max(int(num_samples), self.dataset_size)
        self.seed = int(seed)
        self.epoch = 0
        self.organ_ids = []
        self.organ_indices: Dict[int, torch.Tensor] = {}
        for class_id in range(1, NUM_CLASSES):
            indices = [
                index
                for index, present in enumerate(dataset.slice_organs)
                if class_id in present
            ]
            if not indices:
                continue
            self.organ_ids.append(class_id)
            self.organ_indices[class_id] = torch.tensor(indices, dtype=torch.long)
        if not self.organ_ids:
            raise RuntimeError("Coverage-balanced sampling found no foreground slices")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        draws = torch.randperm(self.dataset_size, generator=generator).tolist()
        supplement = self.num_samples - self.dataset_size
        organ_order = torch.randperm(len(self.organ_ids), generator=generator).tolist()
        for position in range(supplement):
            if position and position % len(organ_order) == 0:
                organ_order = torch.randperm(
                    len(self.organ_ids), generator=generator
                ).tolist()
            class_id = self.organ_ids[organ_order[position % len(organ_order)]]
            candidates = self.organ_indices[class_id]
            candidate_position = int(torch.randint(
                len(candidates),
                (1,),
                generator=generator,
            ).item())
            draws.append(int(candidates[candidate_position]))

        order = torch.randperm(len(draws), generator=generator).tolist()
        return iter([draws[index] for index in order])

    def __len__(self) -> int:
        return self.num_samples


class PresentClassDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        losses = []
        for c in range(1, logits.shape[1]):
            target_c = (target == c).to(dtype=probs.dtype)
            if target_c.sum() <= 0:
                continue
            prob_c = probs[:, c]
            intersection = (prob_c * target_c).sum()
            denominator = prob_c.sum() + target_c.sum()
            losses.append(1.0 - (2.0 * intersection + self.smooth) / (denominator + self.smooth))
        if not losses:
            return logits.sum() * 0.0
        return torch.stack(losses).mean()


class PresentClassTverskyLoss(nn.Module):

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-5):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        losses = []
        for class_id in range(1, logits.shape[1]):
            target_c = (target == class_id).to(dtype=probs.dtype)
            if target_c.sum() <= 0:
                continue
            prob_c = probs[:, class_id]
            true_positive = (prob_c * target_c).sum()
            false_positive = (prob_c * (1.0 - target_c)).sum()
            false_negative = ((1.0 - prob_c) * target_c).sum()
            score = (true_positive + self.smooth) / (
                true_positive
                + self.alpha * false_positive
                + self.beta * false_negative
                + self.smooth
            )
            losses.append(1.0 - score)
        if not losses:
            return logits.sum() * 0.0
        return torch.stack(losses).mean()


class BalancedPresenceBCELoss(nn.Module):

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        element_loss = F.binary_cross_entropy_with_logits(
            logits,
            target.to(dtype=logits.dtype),
            reduction="none",
        )
        positive = target > 0.5
        negative = ~positive
        terms = []
        if positive.any():
            terms.append(element_loss[positive].mean())
        if negative.any():
            terms.append(element_loss[negative].mean())
        return torch.stack(terms).mean() if terms else logits.sum() * 0.0


def presence_targets(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.stack(
        [(labels == class_id).flatten(1).any(dim=1) for class_id in range(1, num_classes)],
        dim=1,
    ).to(dtype=torch.float32)


def postprocess(prediction: np.ndarray) -> np.ndarray:
    output = keep_largest_components(
        prediction,
        compact_classes=COMPACT_ORGANS,
        max_components=1,
    )
    for class_id in HARD_ORGANS:
        mask = output == class_id
        if not mask.any():
            continue
        filled = close_and_fill(mask, iterations=1)
        safe = (output == 0) | (output == class_id)
        output[filled & safe] = class_id
    return output.astype(np.uint8)


def normalize_model_input(batch: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(dtype=batch.dtype, device=batch.device)
    std = IMAGENET_STD.to(dtype=batch.dtype, device=batch.device)
    return (batch - mean) / std


@torch.inference_mode()
def validate(
    model: CTSegModel,
    dataset: FullViewSliceDataset,
    device: torch.device,
    batch_size: int,
    use_amp: bool,
    smooth_sigma: float,
) -> Dict[str, object]:
    model.eval()
    per_class: Dict[int, List[float]] = {c: [] for c in range(1, NUM_CLASSES)}
    volume_ratios: Dict[int, List[float]] = {c: [] for c in range(1, NUM_CLASSES)}

    for case_index, volume in enumerate(dataset.volumes, start=1):
        height, width, depth = volume.image.shape
        probabilities = np.empty((NUM_CLASSES, height, width, depth), dtype=np.float16)
        for start in range(0, depth, batch_size):
            z_values = list(range(start, min(start + batch_size, depth)))
            slices = []
            for z in z_values:
                offset = dataset.neighbor_offset(volume)
                z0, z2 = max(z - offset, 0), min(z + offset, depth - 1)
                slices.append(np.stack([
                    volume.image[:, :, z0],
                    volume.image[:, :, z],
                    volume.image[:, :, z2],
                ], axis=0))
            images = torch.from_numpy(np.ascontiguousarray(np.stack(slices).astype(np.float32)))
            images = normalize_model_input(images).to(device, non_blocking=True)
            if dataset.spatial_view_context:
                context_values = [
                    [z / max(depth - 1, 1), 0.5, 0.5, 1.0]
                    for z in z_values
                ]
            else:
                context_values = [
                    [z / max(depth - 1, 1)]
                    for z in z_values
                ]
            z_depths = torch.tensor(
                context_values,
                dtype=torch.float32,
                device=device,
            )
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(
                    images,
                    input_ids=None,
                    attention_mask=None,
                    z_depths=z_depths,
                )
            probs = F.softmax(logits, dim=1).float().cpu().numpy()
            probabilities[:, :, :, start:start + len(z_values)] = np.transpose(probs, (1, 2, 3, 0)).astype(np.float16)

        if smooth_sigma > 0:
            probabilities = gaussian_filter1d(
                probabilities.astype(np.float32),
                sigma=smooth_sigma,
                axis=3,
            )
        prediction = postprocess(np.argmax(probabilities, axis=0).astype(np.uint8))
        label = volume.label

        for class_id in range(1, NUM_CLASSES):
            pred_mask = prediction == class_id
            gt_mask = label == class_id
            pred_count = int(pred_mask.sum())
            gt_count = int(gt_mask.sum())
            if gt_count <= 0:
                continue
            intersection = int(np.logical_and(pred_mask, gt_mask).sum())
            dice = (2.0 * intersection) / max(pred_count + gt_count, 1)
            per_class[class_id].append(float(dice))
            volume_ratios[class_id].append(pred_count / max(gt_count, 1))
        print(f"[Val] {case_index}/{len(dataset.volumes)} {volume.name} complete")

    class_dice = {
        c: float(np.mean(values)) if values else float("nan")
        for c, values in per_class.items()
    }
    class_volume_ratio = {
        c: float(np.mean(values)) if values else float("nan")
        for c, values in volume_ratios.items()
    }
    overall = float(np.nanmean(list(class_dice.values())))
    large = float(np.nanmean([class_dice[c] for c in LARGE_ORGANS]))
    difficult = float(np.nanmean([class_dice[c] for c in DIFFICULT_ORGANS]))
    return {
        "mean_dice": overall,
        "large_mean_dice": large,
        "difficult_mean_dice": difficult,
        "class_dice": class_dice,
        "class_volume_ratio": class_volume_ratio,
    }


def save_resume_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_score: float,
    bad_validations: int,
    args: argparse.Namespace,
) -> None:
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": int(epoch),
        "best_score": float(best_score),
        "bad_validations": int(bad_validations),
        "args": vars(args),
    }, path)


def write_report_row(path: str, epoch: int, metrics: Dict[str, object]) -> None:
    fieldnames = [
        "epoch", "mean_dice", "large_mean_dice", "difficult_mean_dice",
    ] + [f"dice_{c}_{organ_name(c).replace(' ', '_')}" for c in range(1, NUM_CLASSES)]
    row = {
        "epoch": epoch,
        "mean_dice": f"{metrics['mean_dice']:.6f}",
        "large_mean_dice": f"{metrics['large_mean_dice']:.6f}",
        "difficult_mean_dice": f"{metrics['difficult_mean_dice']:.6f}",
    }
    class_dice = metrics["class_dice"]
    for c in range(1, NUM_CLASSES):
        row[fieldnames[3 + c]] = f"{class_dice[c]:.6f}"
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the final fixed-semantics GSP-DINO model.")
    parser.add_argument("--train_images", default="datasets/flare22/train/images")
    parser.add_argument("--train_labels", default="datasets/flare22/train/labels")
    parser.add_argument("--val_images", default="datasets/flare22/test/images")
    parser.add_argument("--val_labels", default="datasets/flare22/test/labels")
    parser.add_argument("--dino_weights", default="/pd/heyang/weights/model.safetensors")
    parser.add_argument("--clip_dir", default="/pd/heyang/weights/clip-vit-base-patch32")
    parser.add_argument("--resume", default="")
    parser.add_argument(
        "--init_checkpoint",
        default="",
        help="Initialize from a plain Stage1 model state without optimizer state.",
    )
    parser.add_argument("--latest_path", default="results/flare22/checkpoints/gsp_dino_resume.pth")
    parser.add_argument("--best_path", default="results/flare22/checkpoints/gsp_dino_final.pth")
    parser.add_argument("--report_path", default="results/flare22/reports/gsp_dino_validation.csv")
    parser.add_argument(
        "--save_every",
        type=int,
        default=0,
        help="Save a model-only epoch checkpoint every N epochs; zero disables periodic saves.",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--samples_per_epoch", type=int, default=5000)
    parser.add_argument(
        "--max_steps_per_epoch",
        type=int,
        default=0,
        help="Optional smoke-test limit; zero trains the complete epoch.",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr_backbone", type=float, default=5e-7)
    parser.add_argument("--lr_backbone_early", type=float, default=None)
    parser.add_argument("--lr_backbone_middle", type=float, default=None)
    parser.add_argument("--lr_backbone_late", type=float, default=None)
    parser.add_argument("--lr_head", type=float, default=2e-5)
    parser.add_argument("--trainable_block_start", type=int, default=6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=1,
        help="Number of linear-warmup epochs; zero starts directly at the configured learning rates.",
    )
    parser.add_argument("--balance_cap", type=float, default=2.5)
    parser.add_argument("--class_weight_cap", type=float, default=2.5)
    parser.add_argument("--background_weight", type=float, default=0.25)
    parser.add_argument(
        "--uniform_class_weights",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use unweighted cross-entropy; class balance is then handled by Dice and sampling.",
    )
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument(
        "--sampling_mode",
        choices=("organ_balanced", "coverage_balanced"),
        default="coverage_balanced",
    )
    parser.add_argument("--organ_sampling_probability", type=float, default=0.8)
    parser.add_argument("--loss_ce_weight", type=float, default=0.5)
    parser.add_argument("--loss_dice_weight", type=float, default=0.5)
    parser.add_argument("--loss_tversky_weight", type=float, default=0.0)
    parser.add_argument("--loss_presence_weight", type=float, default=0.0)
    parser.add_argument("--tversky_alpha", type=float, default=0.3)
    parser.add_argument("--tversky_beta", type=float, default=0.7)
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
        help="Apply auxiliary segmentation supervision at two decoder scales during training.",
    )
    parser.add_argument(
        "--deep_supervision_weights",
        type=float,
        nargs=2,
        default=(0.25, 0.125),
        metavar=("W_X1", "W_X2"),
    )
    parser.add_argument(
        "--augmentation_profile",
        choices=("standard", "btcv_strong"),
        default="standard",
    )
    parser.add_argument(
        "--context_mm",
        type=float,
        default=0.0,
        help="Physical distance to neighboring 2.5D slices; zero keeps adjacent slices.",
    )
    parser.add_argument(
        "--view_mode",
        choices=("full", "fixed_multiview"),
        default="full",
    )
    parser.add_argument("--local_view_size", type=int, default=320)
    parser.add_argument(
        "--spatial_view_context",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--val_every", type=int, default=2)
    parser.add_argument("--early_stop_validations", type=int, default=4)
    parser.add_argument("--smooth_sigma", type=float, default=1.0)
    parser.add_argument("--hu_min", type=float, default=-200.0)
    parser.add_argument("--hu_max", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--no_validation",
        action="store_true",
        help="Train for exactly --epochs and save the final epoch without validation or early stopping.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    if args.resume and args.init_checkpoint:
        raise ValueError("Use either --resume or --init_checkpoint, not both.")
    if args.view_mode == "fixed_multiview" and not args.spatial_view_context:
        raise ValueError(
            "fixed_multiview requires --spatial_view_context so local views retain "
            "their absolute axial position."
        )
    if args.view_mode == "fixed_multiview" and not args.no_validation:
        raise ValueError(
            "fixed_multiview currently requires --no_validation; formal evaluation "
            "uses the probability-fusion inference path after training."
        )
    os.makedirs(os.path.dirname(args.latest_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.best_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.report_path), exist_ok=True)

    train_dataset = FullViewSliceDataset(
        args.train_images,
        args.train_labels,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        augment=True,
        balance_cap=args.balance_cap,
        augmentation_profile=args.augmentation_profile,
        context_mm=args.context_mm,
        view_mode=args.view_mode,
        local_view_size=args.local_view_size,
        spatial_view_context=args.spatial_view_context,
    )
    val_dataset = None
    if not args.no_validation:
        val_dataset = FullViewSliceDataset(
            args.val_images,
            args.val_labels,
            hu_min=args.hu_min,
            hu_max=args.hu_max,
            augment=False,
            balance_cap=args.balance_cap,
            augmentation_profile="standard",
            context_mm=args.context_mm,
            view_mode=args.view_mode,
            local_view_size=args.local_view_size,
            spatial_view_context=args.spatial_view_context,
        )
    if args.sampling_mode == "organ_balanced":
        sampler = OrganBalancedEpochSampler(
            train_dataset,
            args.samples_per_epoch,
            args.seed,
            args.organ_sampling_probability,
        )
    elif args.sampling_mode == "coverage_balanced":
        sampler = CoverageBalancedEpochSampler(
            train_dataset,
            args.samples_per_epoch,
            args.seed,
        )
    else:
        raise ValueError(f"Unsupported sampling mode: {args.sampling_mode}")
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
        worker_init_fn=worker_init_fn,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

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
    if args.init_checkpoint:
        initial_state = torch.load(args.init_checkpoint, map_location="cpu")
        if isinstance(initial_state, dict) and "model" in initial_state:
            initial_state = initial_state["model"]
        if args.deep_supervision:
            incompatible = model.load_state_dict(initial_state, strict=False)
            allowed_missing = {
                "decoder.aux_x1.weight",
                "decoder.aux_x1.bias",
                "decoder.aux_x2.weight",
                "decoder.aux_x2.bias",
            }
            unexpected_missing = set(incompatible.missing_keys) - allowed_missing
            if unexpected_missing or incompatible.unexpected_keys:
                raise RuntimeError(
                    "Unexpected checkpoint incompatibility: "
                    f"missing={sorted(unexpected_missing)}, "
                    f"unexpected={incompatible.unexpected_keys}"
                )
            print(
                "[Train] initialized zero-valued deep-supervision heads not present "
                "in the source checkpoint"
            )
        else:
            model.load_state_dict(initial_state, strict=True)
        print(f"[Train] initialized GSP-DINO from {args.init_checkpoint}")
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

    backbone_groups = {"early": [], "middle": [], "late": []}
    head_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if not name.startswith("backbone."):
            head_params.append(parameter)
            continue
        block_id = None
        parts = name.split(".")
        if len(parts) > 2 and parts[1] == "blocks" and parts[2].isdigit():
            block_id = int(parts[2])
        if block_id is not None and block_id < 12:
            backbone_groups["early"].append(parameter)
        elif block_id is not None and block_id < 18:
            backbone_groups["middle"].append(parameter)
        else:
            backbone_groups["late"].append(parameter)

    grouped_lrs = {
        "early": args.lr_backbone_early,
        "middle": args.lr_backbone_middle,
        "late": args.lr_backbone_late,
    }
    use_grouped_backbone_lr = any(value is not None for value in grouped_lrs.values())
    if use_grouped_backbone_lr and not all(value is not None for value in grouped_lrs.values()):
        raise ValueError("Set all three grouped backbone learning rates or none of them")
    if use_grouped_backbone_lr:
        optimizer_groups = [
            {"params": backbone_groups[name], "lr": grouped_lrs[name], "name": f"backbone_{name}"}
            for name in ("early", "middle", "late")
            if backbone_groups[name]
        ]
    else:
        optimizer_groups = [{
            "params": [parameter for group in backbone_groups.values() for parameter in group],
            "lr": args.lr_backbone,
            "name": "backbone",
        }]
    optimizer_groups.append({"params": head_params, "lr": args.lr_head, "name": "head"})
    optimizer = optim.AdamW(optimizer_groups, weight_decay=args.weight_decay)
    if args.warmup_epochs < 0 or args.warmup_epochs >= args.epochs:
        raise ValueError("--warmup_epochs must be in [0, epochs - 1]")
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs - args.warmup_epochs, 1),
        eta_min=1e-7,
    )
    if args.warmup_epochs > 0:
        warmup = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=args.warmup_epochs,
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            [warmup, cosine],
            milestones=[args.warmup_epochs],
        )
    else:
        scheduler = cosine
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if args.uniform_class_weights:
        class_weights = torch.ones(NUM_CLASSES, dtype=torch.float32, device=device)
    else:
        class_weights = train_dataset.class_weights(
            background_weight=args.background_weight,
            cap=args.class_weight_cap,
        ).to(device)
    ce_loss = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )
    dice_loss = PresentClassDiceLoss().to(device)
    tversky_loss = PresentClassTverskyLoss(
        alpha=args.tversky_alpha,
        beta=args.tversky_beta,
    ).to(device)
    presence_loss = BalancedPresenceBCELoss().to(device)
    loss_weights = {
        "ce": float(args.loss_ce_weight),
        "dice": float(args.loss_dice_weight),
        "tversky": float(args.loss_tversky_weight),
        "presence": float(args.loss_presence_weight),
    }
    if any(value < 0.0 for value in loss_weights.values()):
        raise ValueError(f"Loss weights must be non-negative: {loss_weights}")
    if not math.isclose(sum(loss_weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"Loss weights must sum to 1.0: {loss_weights}")
    if loss_weights["presence"] > 0.0 and not args.presence_head:
        raise ValueError("Positive presence loss weight requires --presence_head")
    print(f"[Train] class_weights={class_weights.cpu().tolist()}")
    if isinstance(sampler, CoverageBalancedEpochSampler):
        sampling_detail = (
            f"coverage={sampler.dataset_size}, "
            f"balanced_supplement={len(sampler) - sampler.dataset_size}"
        )
    else:
        sampling_detail = f"organ_probability={args.organ_sampling_probability:.2f}"
    print(
        f"[Train] sampling={args.sampling_mode}, "
        f"{sampling_detail}, "
        f"loss_weights={loss_weights}, tversky=({args.tversky_alpha:.2f}, "
        f"{args.tversky_beta:.2f}), depth_adapter={args.depth_context_adapter}, "
        f"presence_head={args.presence_head}, deep_supervision={args.deep_supervision}, "
        f"deep_supervision_weights={tuple(args.deep_supervision_weights)}"
    )

    start_epoch = 0
    best_score = -1.0
    bad_validations = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_epoch = int(checkpoint["epoch"])
        best_score = float(checkpoint.get("best_score", -1.0))
        bad_validations = int(checkpoint.get("bad_validations", 0))
        model.cache_anatomy_semantics(
            tokenized["input_ids"].to(device),
            tokenized["attention_mask"].to(device),
        )
        print(f"[Train] resumed {args.resume} at epoch={start_epoch}, best={best_score:.6f}")

    print(
        f"[Train] device={device}, amp={use_amp}, epochs={args.epochs}, "
        f"samples_per_epoch={args.samples_per_epoch}, batches={len(loader)}, "
        f"batch_size={args.batch_size}, val_batch_size={args.val_batch_size}, "
        f"augmentation={args.augmentation_profile}, context_mm={args.context_mm}, "
        f"view_mode={args.view_mode}, local_view_size={args.local_view_size}, "
        f"spatial_context={args.spatial_view_context}, "
        f"trainable_parameters={sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    print(
        "[Train] optimizer_groups="
        + ", ".join(
            f"{group.get('name', index)}:{group['lr']:.3e}"
            for index, group in enumerate(optimizer.param_groups)
        )
    )

    for epoch in range(start_epoch + 1, args.epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        model.text_encoder.eval()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        total_loss = total_ce = total_dice = total_tversky = total_presence = 0.0
        steps_seen = 0
        for step, batch in enumerate(loader, start=1):
            steps_seen = step
            images, labels, z_depths = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            z_depths = z_depths.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                if args.deep_supervision:
                    logits, presence_logits, deep_logits = model(
                        images,
                        input_ids=None,
                        attention_mask=None,
                        z_depths=z_depths,
                        return_aux=True,
                        return_deep_supervision=True,
                    )
                else:
                    logits, presence_logits = model(
                        images,
                        input_ids=None,
                        attention_mask=None,
                        z_depths=z_depths,
                        return_aux=True,
                    )
                    deep_logits = []
                loss_ce = ce_loss(logits, labels)
                loss_dice = dice_loss(logits, labels)
                loss_tversky = tversky_loss(logits, labels)
                if presence_logits is None:
                    loss_presence = logits.sum() * 0.0
                else:
                    loss_presence = presence_loss(
                        presence_logits,
                        presence_targets(labels, logits.shape[1]),
                    )
                segmentation_loss = (
                    loss_weights["ce"] * loss_ce
                    + loss_weights["dice"] * loss_dice
                    + loss_weights["tversky"] * loss_tversky
                )
                if deep_logits:
                    weighted_losses = [segmentation_loss]
                    total_scale_weight = 1.0
                    for auxiliary_logits, scale_weight in zip(
                        deep_logits,
                        args.deep_supervision_weights,
                    ):
                        auxiliary_labels = F.interpolate(
                            labels[:, None].float(),
                            size=auxiliary_logits.shape[-2:],
                            mode="nearest",
                        )[:, 0].long()
                        auxiliary_loss = (
                            loss_weights["ce"] * ce_loss(auxiliary_logits, auxiliary_labels)
                            + loss_weights["dice"] * dice_loss(auxiliary_logits, auxiliary_labels)
                            + loss_weights["tversky"] * tversky_loss(auxiliary_logits, auxiliary_labels)
                        )
                        weighted_losses.append(float(scale_weight) * auxiliary_loss)
                        total_scale_weight += float(scale_weight)
                    segmentation_loss = sum(weighted_losses) / total_scale_weight
                loss = segmentation_loss + loss_weights["presence"] * loss_presence
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item())
            total_ce += float(loss_ce.item())
            total_dice += float(loss_dice.item())
            total_tversky += float(loss_tversky.item())
            total_presence += float(loss_presence.item())
            if step == 1 or step % 50 == 0:
                print(
                    f"Epoch {epoch:03d}/{args.epochs:03d} Step {step:05d}/{len(loader):05d} "
                    f"Loss {loss.item():.4f} CE {loss_ce.item():.4f} "
                    f"DiceLoss {loss_dice.item():.4f} "
                    f"Tversky {loss_tversky.item():.4f} "
                    f"Presence {loss_presence.item():.4f}"
                )
            if args.max_steps_per_epoch > 0 and step >= args.max_steps_per_epoch:
                print(f"[Smoke] stopping epoch after {step} steps")
                break

        scheduler.step()
        batches = max(steps_seen, 1)
        print(
            f"[Epoch {epoch:03d}] loss={total_loss / batches:.5f}, "
            f"ce={total_ce / batches:.5f}, dice_loss={total_dice / batches:.5f}, "
            f"tversky={total_tversky / batches:.5f}, "
            f"presence={total_presence / batches:.5f}, "
            "lr="
            + "/".join(
                f"{group.get('name', index)}:{group['lr']:.3e}"
                for index, group in enumerate(optimizer.param_groups)
            )
        )
        if device.type == "cuda":
            print(
                f"[GPU {epoch:03d}] allocated={torch.cuda.memory_allocated(device) / 1024 ** 3:.2f}GiB, "
                f"reserved={torch.cuda.memory_reserved(device) / 1024 ** 3:.2f}GiB, "
                f"peak={torch.cuda.max_memory_allocated(device) / 1024 ** 3:.2f}GiB"
            )

        if args.no_validation:
            save_resume_checkpoint(
                args.latest_path,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_score,
                0,
                args,
            )
            print(f"[Checkpoint] fixed-epoch resume state saved to {args.latest_path}")
            if args.save_every > 0 and epoch % args.save_every == 0:
                stem, suffix = os.path.splitext(args.best_path)
                epoch_path = f"{stem}_epoch_{epoch:03d}{suffix or '.pth'}"
                torch.save(model.state_dict(), epoch_path)
                print(f"[Checkpoint] epoch model saved to {epoch_path}")
            if epoch == args.epochs:
                torch.save(model.state_dict(), args.best_path)
                print(f"[Fixed Epoch] final model saved to {args.best_path}")
            continue

        should_validate = epoch % args.val_every == 0 or epoch == args.epochs
        if not should_validate:
            continue

        metrics = validate(
            model,
            val_dataset,
            device,
            args.val_batch_size,
            use_amp,
            args.smooth_sigma,
        )
        print(
            f"[Validation {epoch:03d}] mean={metrics['mean_dice']:.6f}, "
            f"large={metrics['large_mean_dice']:.6f}, "
            f"difficult={metrics['difficult_mean_dice']:.6f}"
        )
        for c in range(1, NUM_CLASSES):
            print(
                f"  {c:02d} {organ_name(c):<20} "
                f"dice={metrics['class_dice'][c]:.6f} "
                f"pred_gt={metrics['class_volume_ratio'][c]:.4f}"
            )
        write_report_row(args.report_path, epoch, metrics)

        score = float(metrics["mean_dice"])
        if score > best_score:
            best_score = score
            bad_validations = 0
            torch.save(model.state_dict(), args.best_path)
            print(f"[Validation] new best mean Dice={best_score:.6f}; saved {args.best_path}")
        else:
            bad_validations += 1
            print(f"[Validation] no improvement; bad_validations={bad_validations}")

        save_resume_checkpoint(
            args.latest_path,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_score,
            bad_validations,
            args,
        )
        print(f"[Checkpoint] saved resume state {args.latest_path}")

        if bad_validations >= args.early_stop_validations:
            print(f"[Early Stop] no validation improvement for {bad_validations} validations")
            break

    if args.no_validation:
        print(f"[Done] fixed epoch={args.epochs}; final checkpoint={args.best_path}")
    else:
        print(f"[Done] best mean Dice={best_score:.6f}; best checkpoint={args.best_path}")


if __name__ == "__main__":
    main(parse_args())
