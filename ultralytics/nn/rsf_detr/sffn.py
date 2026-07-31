"""Spectral-Enhanced Feed-Forward Network (SFFN)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class _SpectralFeedForward(nn.Module):
    def __init__(self, channels: int, expansion: float = 2.0):
        super().__init__()
        hidden_channels = int(channels * expansion)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, 1, bias=False)
        self.dwconv = nn.Conv2d(
            hidden_channels * 2,
            hidden_channels * 2,
            3,
            padding=2,
            dilation=2,
            groups=hidden_channels * 2,
            bias=False,
        )
        self.project_out = nn.Conv2d(hidden_channels, channels, 1, bias=False)
        self.fft_channel_weight = nn.Parameter(torch.randn(1, hidden_channels * 2, 1, 1))
        self.fft_channel_bias = nn.Parameter(torch.randn(1, hidden_channels * 2, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_dtype = x.dtype
        x = self.dwconv(self.project_in(x))
        original_width = x.shape[-1]
        pad_width = (-original_width) % 2
        if pad_width:
            x = F.pad(x, (0, pad_width))
        spectrum = torch.fft.rfft2(x.float())
        spectrum = self.fft_channel_weight * spectrum + self.fft_channel_bias
        x = torch.fft.irfft2(spectrum, s=x.shape[-2:])
        if pad_width:
            x = x[..., :original_width]
        gate, value = x.chunk(2, dim=1)
        return self.project_out((F.silu(gate) * value).to(output_dtype))


class SFFN(nn.Module):
    """AIFI encoder layer whose feed-forward branch is the paper's SFFN."""

    def __init__(
        self,
        c1: int,
        cm: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.0,
        act: nn.Module = nn.GELU(),
        normalize_before: bool = False,
    ):
        super().__init__()
        del cm, act
        self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)
        self.ffn = _SpectralFeedForward(c1)
        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.normalize_before = normalize_before

    @staticmethod
    def _with_position(tensor: torch.Tensor, position: Optional[torch.Tensor]) -> torch.Tensor:
        return tensor if position is None else tensor + position

    def _forward_post(self, src: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = src.shape
        src = src.flatten(2).permute(0, 2, 1)
        query = key = self._with_position(src, position)
        attended = self.ma(query, key, value=src)[0]
        src = self.norm1(src + self.dropout1(attended))
        spectral = self.ffn(
            attended.permute(0, 2, 1).view(batch, channels, height, width).contiguous()
        ).flatten(2).permute(0, 2, 1)
        return self.norm2(src + self.dropout2(spectral))

    def _forward_pre(self, src: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = src.shape
        normalized = self.norm1(src.flatten(2).permute(0, 2, 1))
        query = key = self._with_position(normalized, position)
        attended = self.ma(query, key, value=normalized)[0]
        src = src.flatten(2).permute(0, 2, 1) + self.dropout1(attended)
        spectral_input = self.norm2(src)
        spectral = self.ffn(
            spectral_input.permute(0, 2, 1).view(batch, channels, height, width).contiguous()
        ).flatten(2).permute(0, 2, 1)
        return src + self.dropout2(spectral)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels, height, width = x.shape[1:]
        position = self._build_2d_sincos_position_embedding(width, height, channels)
        position = position.to(device=x.device, dtype=x.dtype)
        encoded = self._forward_pre(x, position) if self.normalize_before else self._forward_post(x, position)
        return encoded.permute(0, 2, 1).view(-1, channels, height, width).contiguous()

    @staticmethod
    def _build_2d_sincos_position_embedding(
        width: int,
        height: int,
        embed_dim: int,
        temperature: float = 10000.0,
    ) -> torch.Tensor:
        if embed_dim % 4:
            raise ValueError("SFFN embedding dimension must be divisible by four.")
        grid_width = torch.arange(width, dtype=torch.float32)
        grid_height = torch.arange(height, dtype=torch.float32)
        grid_width, grid_height = torch.meshgrid(grid_width, grid_height, indexing="ij")
        omega = torch.arange(embed_dim // 4, dtype=torch.float32) / (embed_dim // 4)
        omega = 1.0 / (temperature ** omega)
        width_embedding = grid_width.flatten()[..., None] @ omega[None]
        height_embedding = grid_height.flatten()[..., None] @ omega[None]
        return torch.cat(
            (
                width_embedding.sin(),
                width_embedding.cos(),
                height_embedding.sin(),
                height_embedding.cos(),
            ),
            dim=1,
        )[None]
