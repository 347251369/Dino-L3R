from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int, int] = (3, 3, 3),
    ) -> None:
        super().__init__()
        padding = tuple(size // 2 for size in kernel_size)
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size, padding=padding, bias=False
        )
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size, padding=padding, bias=False
        )
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, 1, bias=False)
        )
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + identity)


class DownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int, int] = (2, 2, 2),
    ) -> None:
        super().__init__()
        self.down = nn.Conv3d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.block = ResidualBlock3D(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.down(x))


class UpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        stride: tuple[int, int, int] = (2, 2, 2),
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=stride, stride=stride
        )
        self.block = ResidualBlock3D(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))
