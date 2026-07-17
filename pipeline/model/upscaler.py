from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- building blocks
class ResBlock(nn.Module):
    """Simple residual block: Conv -> ReLU -> Conv, plus skip."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.conv1(x), inplace=True)
        y = self.conv2(y)
        return x + y


class UpsampleBlock(nn.Module):
    """One 2x upsample stage via sub-pixel convolution."""

    def __init__(self, channels: int):
        super().__init__()
        # Conv produces 4x channels, PixelShuffle reshuffles into 2x spatial size.
        self.conv = nn.Conv2d(channels, channels * 4, 3, padding=1)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.shuffle(self.conv(x)), inplace=True)


# ---------------------------------------------------------------------- model
class Upscaler(nn.Module):
    def __init__(
        self,
        scale: int = 4,
        channels: int = 64,
        num_res_blocks: int = 8,
        in_channels: int = 3,
    ):
        super().__init__()
        if scale < 2 or (scale & (scale - 1)) != 0:
            raise ValueError(f"scale must be a power of 2 >= 2, got {scale}")

        self.scale = scale

        # Head: lift RGB to feature channels.
        self.head = nn.Conv2d(in_channels, channels, 3, padding=1)

        # Body: stack of residual blocks, all at LR resolution.
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(num_res_blocks)])

        # Body-tail conv, then a long skip inside the body (SRResNet-style).
        self.body_tail = nn.Conv2d(channels, channels, 3, padding=1)

        # After concatenating LR (in_channels) with body features (channels),
        # a 1x1 conv fuses them back to `channels` before upsampling.
        self.fuse = nn.Conv2d(channels + in_channels, channels, 1)

        # Upsampler: log2(scale) stages of 2x.
        num_stages = int(round(torch.log2(torch.tensor(float(scale))).item()))
        self.upsampler = nn.Sequential(*[UpsampleBlock(channels) for _ in range(num_stages)])

        # Tail: project features back to RGB.
        self.tail = nn.Conv2d(channels, in_channels, 3, padding=1)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        # CNN body in LR space, with an internal long skip (standard SRResNet).
        f = self.head(lr)
        residual = f
        f = self.body(f)
        f = self.body_tail(f)
        f = f + residual

        # Skip from the raw LR image, concatenated with learned features.
        f = torch.cat([f, lr], dim=1)
        f = self.fuse(f)

        # Upsample to HR, then project to RGB.
        f = self.upsampler(f)
        sr = self.tail(f)
        return sr


# --------------------------------------------------------------------- quick check
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Upscaler(scale=4, channels=64, num_res_blocks=8).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device: {device}")
    print(f"params: {n_params:,}")

    lr = torch.randn(2, 3, 48, 48, device=device)
    with torch.no_grad():
        sr = model(lr)
    print(f"in : {tuple(lr.shape)}")
    print(f"out: {tuple(sr.shape)}   (expected (2, 3, 192, 192))")
