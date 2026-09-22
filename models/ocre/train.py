"""Train the final OCRE model used by Dino-L3R."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.physical_dataset import PhysicalAllOrganDataset, prepare_physical_cache
from .losses import DeepSupervisionLoss
from .model import OCRE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_images", required=True)
    parser.add_argument("--train_labels", required=True)
    parser.add_argument("--coarse_masks", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--patch_size", nargs=3, type=int, default=(64, 144, 144))
    parser.add_argument("--base_channels", type=int, default=20)
    parser.add_argument("--samples_per_epoch", type=int, default=2200)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--seed", type=int, default=542)
    parser.add_argument("--boundary_probability", type=float, default=0.75)
    parser.add_argument("--prior_shift_probability", type=float, default=0.65)
    parser.add_argument("--prior_shift_max_mm", type=float, default=3.0)
    parser.add_argument("--loss_dice_weight", type=float, default=0.65)
    parser.add_argument("--loss_bce_weight", type=float, default=0.35)
    parser.add_argument("--init_checkpoint")
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    cases = prepare_physical_cache(
        args.train_images, args.train_labels, args.coarse_masks, args.cache_dir
    )
    dataset = PhysicalAllOrganDataset(
        cases,
        args.patch_size,
        args.samples_per_epoch,
        args.seed,
        boundary_centered=True,
        boundary_probability=args.boundary_probability,
        prior_shift_probability=args.prior_shift_probability,
        prior_shift_max_mm=args.prior_shift_max_mm,
        intensity_augmentation=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    model = OCRE(args.base_channels).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
        print(f"[OCRE] initialized from {args.init_checkpoint}", flush=True)

    criterion = DeepSupervisionLoss(
        dice_weight=args.loss_dice_weight,
        bce_weight=args.loss_bce_weight,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = np.zeros(6, dtype=np.float64)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for step, (x, target, _target_sdf, _spacing, organ_index) in enumerate(loader, 1):
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            organ_index = organ_index.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(x, organ_index, return_aux=True)
                loss, dice = criterion(outputs, target)
                logits = outputs[0]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            probability = torch.sigmoid(logits.detach())
            prediction = probability >= 0.5
            target_bool = target > 0.5
            tp = (prediction & target_bool).sum().float()
            precision = tp / prediction.sum().clamp_min(1)
            recall = tp / target_bool.sum().clamp_min(1)
            brier = (probability - target).square().mean()
            nll = torch.nn.functional.binary_cross_entropy(probability, target)
            values = (loss, dice, precision, recall, brier, nll)
            sums += np.asarray([float(value.item()) for value in values])
            if step == 1 or step % 25 == 0 or step == len(loader):
                print(
                    f"Epoch {epoch:03d}/{args.epochs:03d} Step {step:04d}/{len(loader):04d} "
                    f"Loss {loss.item():.4f} Dice {dice.item():.4f} "
                    f"Precision {precision.item():.4f} Recall {recall.item():.4f} "
                    f"Brier {brier.item():.4f} NLL {nll.item():.4f}",
                    flush=True,
                )
        scheduler.step()
        means = sums / max(len(loader), 1)
        peak = (
            torch.cuda.max_memory_allocated(device) / 2**30
            if device.type == "cuda"
            else 0.0
        )
        print(
            f"[OCRE {epoch:03d}] loss={means[0]:.5f} dice={means[1]:.5f} "
            f"precision={means[2]:.5f} recall={means[3]:.5f} "
            f"brier={means[4]:.5f} nll={means[5]:.5f} "
            f"lr={optimizer.param_groups[0]['lr']:.8f} peak_gpu_gib={peak:.2f}",
            flush=True,
        )
        checkpoint = {"model": model.state_dict(), "epoch": epoch, "config": vars(args)}
        torch.save(checkpoint, args.save_path)
        if args.save_every > 0 and epoch % args.save_every == 0:
            path = Path(args.save_path)
            epoch_path = path.with_name(f"{path.stem}_epoch_{epoch:03d}{path.suffix}")
            torch.save(checkpoint, epoch_path)


if __name__ == "__main__":
    train(parse_args())
