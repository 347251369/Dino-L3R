from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import DownBlock3D, ResidualBlock3D, UpBlock3D


class OCRE(nn.Module):
    def __init__(
        self,
        base_channels: int = 20,
        num_organs: int = 13,
        max_logit_correction: float = 4.0,
        max_channels: int = 256,
    ) -> None:
        super().__init__()

        c1 = int(base_channels)
        c2 = min(c1 * 2, max_channels)
        c3 = min(c1 * 4, max_channels)
        c4 = min(c1 * 8, max_channels)
        c5 = min(c1 * 12, max_channels)

        self.num_organs = int(num_organs)
        self.max_logit_correction = float(max_logit_correction)

        self.enc1 = ResidualBlock3D(22 + self.num_organs, c1, kernel_size=(1, 3, 3))
        self.enc2 = DownBlock3D(c1, c2, stride=(1, 2, 2))
        self.enc3 = DownBlock3D(c2, c3)
        self.enc4 = DownBlock3D(c3, c4)
        self.bottleneck = DownBlock3D(c4, c5)

        self.dec4 = UpBlock3D(c5, c4, c4)
        self.dec3 = UpBlock3D(c4, c3, c3)
        self.dec2 = UpBlock3D(c3, c2, c2)
        self.dec1 = UpBlock3D(c2, c1, c1, stride=(1, 2, 2))

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
        organ_index = organ_index.view(-1).long()

        if organ_index.shape[0] != x.shape[0]:
            raise ValueError("organ_index and input batch size must match")

        if organ_index.min() < 1 or organ_index.max() > self.num_organs:
            raise ValueError("organ_index values must be between 1 and num_organs")

        organ_one_hot = F.one_hot(
            organ_index - 1,
            num_classes=self.num_organs,
        ).to(dtype=x.dtype)

        organ_one_hot = organ_one_hot[:, :, None, None, None]
        organ_one_hot = organ_one_hot.expand(
            -1,
            -1,
            x.shape[-3],
            x.shape[-2],
            x.shape[-1],
        )

        x = torch.cat((x, organ_one_hot), dim=1)

        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        bottleneck = self.bottleneck(s4)

        d4 = self.dec4(bottleneck, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        anchor = x[:, 15:16] * 8.0
        main = anchor + self.max_logit_correction * torch.tanh(self.head(d1))

        if not return_aux:
            return main

        aux2 = F.interpolate(
            self.aux2(d2),
            size=main.shape[-3:],
            mode="trilinear",
            align_corners=False,
        )

        aux3 = F.interpolate(
            self.aux3(d3),
            size=main.shape[-3:],
            mode="trilinear",
            align_corners=False,
        )

        return (
            main,
            anchor + self.max_logit_correction * torch.tanh(aux2),
            anchor + self.max_logit_correction * torch.tanh(aux3),
        )