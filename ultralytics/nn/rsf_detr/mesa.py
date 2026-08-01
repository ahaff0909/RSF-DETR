"""Multi-Scale Edge Selection Aggregation (MESA)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.block import C2f
from ultralytics.nn.modules.conv import Conv


class _ChannelPool(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((x.max(dim=1, keepdim=True)[0], x.mean(dim=1, keepdim=True)), dim=1)


class _SpatialGate(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.compress = _ChannelPool()
        self.spatial = Conv(2, 1, 3, act=False)
        self.dw1 = nn.Sequential(
            Conv(channels, channels, 5, d=2, g=channels, act=nn.GELU()),
            Conv(channels, channels, 7, d=3, g=channels, act=nn.GELU()),
        )
        self.dw2 = Conv(channels, channels, 3, g=channels, act=nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_weight = self.spatial(self.compress(x))
        return self.dw1(x) * spatial_weight + self.dw2(x)


class _LocalAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(channels, 1, 1))
        self.b = nn.Parameter(torch.ones(channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        centered = x - x.mean(dim=(2, 3), keepdim=True)
        return self.a * centered * x + self.b * x


class _DualDomainSelection(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.spatial_gate = _SpatialGate(channels)
        self.local_attention = _LocalAttention(channels)
        self.a = nn.Parameter(torch.zeros(channels, 1, 1))
        self.b = nn.Parameter(torch.ones(channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        selected = self.local_attention(self.spatial_gate(x))
        return self.a * selected + self.b * x


class _EdgeEnhancer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.out_conv = Conv(channels, channels, act=nn.Sigmoid())
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.out_conv(x - self.pool(x))


class _MESAUnit(nn.Module):
    def __init__(self, channels: int, bins=(3, 6, 9, 12)):
        super().__init__()
        branch_channels = channels // len(bins)
        self.features = nn.ModuleList(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(bin_size),
                Conv(channels, branch_channels, 1),
                Conv(branch_channels, branch_channels, 3, g=branch_channels),
            )
            for bin_size in bins
        )
        self.ees = nn.ModuleList(_EdgeEnhancer(branch_channels) for _ in bins)
        self.local_conv = Conv(channels, channels, 3)
        self.dsm = _DualDomainSelection(channels * 2)
        self.final_conv = Conv(channels * 2, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_size = x.shape[2:]
        outputs = [self.local_conv(x)]
        for feature, enhancer in zip(self.features, self.ees):
            branch = F.interpolate(feature(x), output_size, mode="bilinear", align_corners=True)
            outputs.append(enhancer(branch))
        return self.final_conv(self.dsm(torch.cat(outputs, dim=1)))


class MESA(C2f):
    """Multi-Scale Edge Selection Aggregation described in the RSF-DETR paper."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(_MESAUnit(self.c) for _ in range(n))
