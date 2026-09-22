"""Final organ-conditioned three-dimensional correction network."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import DownBlock3D, ResidualBlock3D, UpBlock3D


class OrganConditioning(nn.Module):
    """Zero-initialized organ conditioning at successive decoder scales."""

    def __init__(self, embedding_dim: int, channels: Sequence[int]) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Linear(embedding_dim, int(channel) * 2) for channel in channels]
        )
        for projection in self.projections:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(
        self, feature: torch.Tensor, embedding: torch.Tensor, level: int
    ) -> torch.Tensor:
        gamma, beta = self.projections[level](embedding).chunk(2, dim=1)
        gamma = gamma[:, :, None, None, None]
        beta = beta[:, :, None, None, None]
        return feature * (1.0 + gamma) + beta


class OCRE(nn.Module):
    """OCRE used by the final Dino-L3R model.

    Axial resolution is preserved in the first encoder stage. The network uses
    organ conditioning at the bottleneck and decoder scales and predicts a
    bounded residual with respect to the GSP-DINO signed-distance prior.
    """

    def __init__(
        self,
        base_channels: int = 20,
        num_organs: int = 13,
        max_logit_correction: float = 4.0,
        embedding_dim: int = 32,
        max_channels: int = 256,
    ) -> None:
        super().__init__()
        c1 = int(base_channels)
        c2 = min(c1 * 2, max_channels)
        c3 = min(c1 * 4, max_channels)
        c4 = min(c1 * 8, max_channels)
        c5 = min(c1 * 12, max_channels)
        self.max_logit_correction = float(max_logit_correction)

        self.enc1 = ResidualBlock3D(22, c1, kernel_size=(1, 3, 3))
        self.enc2 = DownBlock3D(c1, c2, stride=(1, 2, 2))
        self.enc3 = DownBlock3D(c2, c3)
        self.enc4 = DownBlock3D(c3, c4)
        self.bottleneck = DownBlock3D(c4, c5)

        self.organ_embedding = nn.Embedding(num_organs, embedding_dim)
        self.bottleneck_film = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(embedding_dim * 2, c5 * 2),
        )
        nn.init.zeros_(self.bottleneck_film[-1].weight)
        nn.init.zeros_(self.bottleneck_film[-1].bias)

        self.dec4 = UpBlock3D(c5, c4, c4)
        self.dec3 = UpBlock3D(c4, c3, c3)
        self.dec2 = UpBlock3D(c3, c2, c2)
        self.dec1 = UpBlock3D(c2, c1, c1, stride=(1, 2, 2))
        self.decoder_conditioning = OrganConditioning(
            embedding_dim, (c4, c3, c2, c1)
        )

        self.head = nn.Conv3d(c1, 1, 1)
        self.aux2 = nn.Conv3d(c2, 1, 1)
        self.aux3 = nn.Conv3d(c3, 1, 1)
        for head in (self.head, self.aux2, self.aux3):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        x: torch.Tensor,
        organ_index: torch.Tensor,
        return_aux: bool = False,
    ):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        bottleneck = self.bottleneck(s4)

        embedding = self.organ_embedding(organ_index.view(-1).long())
        gamma, beta = self.bottleneck_film(embedding).chunk(2, dim=1)
        bottleneck = bottleneck * (1.0 + gamma[:, :, None, None, None])
        bottleneck = bottleneck + beta[:, :, None, None, None]

        d4 = self.decoder_conditioning(self.dec4(bottleneck, s4), embedding, 0)
        d3 = self.decoder_conditioning(self.dec3(d4, s3), embedding, 1)
        d2 = self.decoder_conditioning(self.dec2(d3, s2), embedding, 2)
        d1 = self.decoder_conditioning(self.dec1(d2, s1), embedding, 3)

        anchor = x[:, 15:16] * 8.0
        main = anchor + self.max_logit_correction * torch.tanh(self.head(d1))
        if not return_aux:
            return main
        aux2 = F.interpolate(
            self.aux2(d2), size=main.shape[-3:], mode="trilinear", align_corners=False
        )
        aux3 = F.interpolate(
            self.aux3(d3), size=main.shape[-3:], mode="trilinear", align_corners=False
        )
        return (
            main,
            anchor + self.max_logit_correction * torch.tanh(aux2),
            anchor + self.max_logit_correction * torch.tanh(aux3),
        )
