"""
DRSA-Net Multi-Stream Parallel Encoder
=======================================
Four independent CNN branches process parallel image representations
simultaneously.  Feature maps are concatenated along the channel axis
(STRICT parallel, not sequential) then projected to a shared embedding dim.

Branch layout:
  Branch 1 — Original RGB         (3, H, W)  → (C, H/4, W/4)
  Branch 2 — Felzenszwalb mask 1  (1, H, W)  → (C, H/4, W/4)
  Branch 3 — Felzenszwalb mask 2  (1, H, W)  → (C, H/4, W/4)  ← parallel
            ─ concatenated with Branch 2 inside one StreamEncoder (2 → C)
  Branch 4 — CLAHE image          (3, H, W)  → (C, H/4, W/4)
  Branch 5 — Watershed map        (1, H, W)  → (C, H/4, W/4)

  Concat →  (4C, H/4, W/4)
  Project → (embed_dim, H/4, W/4)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# --------------------------------------------------------------------------- #
#  Building blocks                                                             #
# --------------------------------------------------------------------------- #

class ConvBnGelu(nn.Module):
    """Conv2d → BatchNorm2d → GELU  (optionally strided)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.isnan(x).any():
             # Only print if not NaN to avoid clutter, but wait, I want to see BEFORE NaN
             pass
        
        for i, sublayer in enumerate(self.block):
            if i == 0:
                print(f"      [Conv2d Input] shape={x.shape}, min={x.min().item():.4f}, max={x.max().item():.4f}")
            
            x = sublayer(x)
            
            if torch.isnan(x).any():
                print(f"      [NaN after sublayer {i}] {sublayer}")
                raise RuntimeError(f"NaN in ConvBnGelu sublayer {i}: {sublayer}")
        return x


class ResidualBlock(nn.Module):
    """
    Lightweight residual block: 3×3 → 3×3 with skip.
    Maintains spatial resolution (no stride here).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.act   = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + residual)


class StreamEncoder(nn.Module):
    """
    Single representation branch encoder.

    Architecture (3 strided blocks, total downsampling = 4×):
      [in_ch, H, W]
        → ConvBnGelu(in_ch, C, stride=2)   → [C, H/2, W/2]
        → ResidualBlock(C)
        → ConvBnGelu(C, C, stride=2)        → [C, H/4, W/4]
        → ResidualBlock(C)

    Parameters
    ----------
    in_channels : int
        Number of input channels for this branch.
    branch_channels : int
        Number of output channels (C).
    """

    def __init__(self, in_channels: int, branch_channels: int):
        super().__init__()
        C = branch_channels
        self.encoder = nn.Sequential(
            ConvBnGelu(in_channels, C, stride=2),   # H/2
            ResidualBlock(C),
            ConvBnGelu(C, C, stride=2),              # H/4
            ResidualBlock(C),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if torch.isnan(x).any():
                raise RuntimeError(f"NaN detected in StreamEncoder layer {i}: {layer}")
        return x


# --------------------------------------------------------------------------- #
#  Multi-Stream Encoder                                                        #
# --------------------------------------------------------------------------- #

class MultiStreamEncoder(nn.Module):
    """
    Parallel 4-branch encoder.

    Inputs (all tensors in the same batch):
        rgb       : (B, 3, H, W)
        clahe     : (B, 3, H, W)
        felz_cat  : (B, 2, H, W)  ← felz1 + felz2 concatenated on channel dim
        watershed : (B, 1, H, W)

    Each branch independently encodes its input → (B, C, H/4, W/4).

    Channel-wise concatenation → (B, 4C, H/4, W/4)
    1×1 projection              → (B, embed_dim, H/4, W/4)

    This is the strict PARALLEL architecture — no information crosses
    between branches until the final concatenation.
    """

    def __init__(self, branch_channels: int, embed_dim: int):
        super().__init__()
        C = branch_channels
        D = embed_dim

        # Four independent branches
        self.branch_rgb       = StreamEncoder(in_channels=3, branch_channels=C)
        self.branch_clahe     = StreamEncoder(in_channels=3, branch_channels=C)
        self.branch_felz      = StreamEncoder(in_channels=2, branch_channels=C)  # 2 masks
        self.branch_watershed = StreamEncoder(in_channels=1, branch_channels=C)

        # 1×1 channel projection: 4C → embed_dim
        self.projection = nn.Sequential(
            nn.Conv2d(4 * C, D, kernel_size=1, bias=False),
            nn.BatchNorm2d(D),
            nn.GELU(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Normal init with small std is very stable
                nn.init.normal_(m.weight, std=0.02)
                if torch.isnan(m.weight).any():
                    raise RuntimeError(f"NaN in weights of {m}")
            elif isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-5
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        rgb: torch.Tensor,
        clahe: torch.Tensor,
        felz1: torch.Tensor,
        felz2: torch.Tensor,
        watershed: torch.Tensor,
    ) -> torch.Tensor:
        # Parallel encoding
        f_rgb  = self.branch_rgb(rgb)
        f_cla  = self.branch_clahe(clahe)
        f_felz = self.branch_felz(torch.cat([felz1, felz2], dim=1))
        f_ws   = self.branch_watershed(watershed)

        # DEBUG: Check for NaNs
        if torch.isnan(f_rgb).any(): print("  [DEBUG] f_rgb contains NaN")
        if torch.isnan(f_cla).any(): print("  [DEBUG] f_cla contains NaN")
        if torch.isnan(f_felz).any(): print("  [DEBUG] f_felz contains NaN")
        if torch.isnan(f_ws).any(): print("  [DEBUG] f_ws contains NaN")

        # Strict parallel concatenation
        fused = torch.cat([f_rgb, f_cla, f_felz, f_ws], dim=1)  # (B, 4C, H/4, W/4)

        # Shared projection
        return self.projection(fused)  # (B, D, H/4, W/4)

    def get_branch_features(
        self,
        rgb: torch.Tensor,
        clahe: torch.Tensor,
        felz1: torch.Tensor,
        felz2: torch.Tensor,
        watershed: torch.Tensor,
    ) -> dict:
        """
        Return individual branch feature maps for CAM visualization.
        Useful for introspection and CAM fusion weights.
        """
        return {
            'f_rgb':  self.branch_rgb(rgb),
            'f_clahe': self.branch_clahe(clahe),
            'f_felz': self.branch_felz(torch.cat([felz1, felz2], dim=1)),
            'f_ws':   self.branch_watershed(watershed),
        }
