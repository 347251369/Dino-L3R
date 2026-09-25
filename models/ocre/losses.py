from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegionLoss(nn.Module):
    def __init__(self, dice_weight: float = 0.65, bce_weight: float = 0.35) -> None:
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.bce_weight = float(bce_weight)
        if min(self.dice_weight, self.bce_weight) < 0.0:
            raise ValueError("Loss weights must be non-negative")
        if self.dice_weight + self.bce_weight <= 0.0:
            raise ValueError("At least one loss weight must be positive")

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        target = target.float()
        probability = torch.sigmoid(logits)
        dims = tuple(range(2, target.ndim))
        intersection = (probability * target).sum(dims)
        dice = (
            (2.0 * intersection + 1e-5)
            / (probability.sum(dims) + target.sum(dims) + 1e-5)
        ).mean()
        bce = F.binary_cross_entropy_with_logits(logits, target)
        loss = self.dice_weight * (1.0 - dice) + self.bce_weight * bce
        return loss, dice.detach()


class DeepSupervisionLoss(nn.Module):
    def __init__(
        self,
        dice_weight: float = 0.65,
        bce_weight: float = 0.35,
        deep_weights: Sequence[float] = (1.0, 0.35, 0.15),
    ) -> None:
        super().__init__()
        self.region_loss = RegionLoss(dice_weight, bce_weight)
        self.deep_weights = tuple(float(weight) for weight in deep_weights)

    def forward(self, outputs, target: torch.Tensor):
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        outputs = tuple(outputs)
        weights = self.deep_weights[: len(outputs)]
        if len(weights) != len(outputs) or sum(weights) <= 0.0:
            raise ValueError("Invalid number of deep-supervision outputs")
        losses = []
        main_dice = None
        for output, weight in zip(outputs, weights):
            loss, dice = self.region_loss(output, target)
            losses.append(weight * loss)
            if main_dice is None:
                main_dice = dice
        return sum(losses) / sum(weights), main_dice
