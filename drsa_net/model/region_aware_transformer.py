"""
DRSA-Net Region-Aware Transformer
===================================
A ViT-style transformer where attention is CONSTRAINED by superpixel
adjacency graphs — not global.

Core formula
------------
Standard:
    Attention(Q,K,V) = Softmax(QKᵀ / √d) · V

Ours (Superpixel-Constrained):
    Attention(Q,K,V) = Softmax((QKᵀ / √d) + M_adj) · V

where M_adj[i,j]:
    =   0   if superpixels i and j are adjacent (allowed)
    = -inf  otherwise                             (blocked)

Two attention modes with learned gating:
    local:      1-hop adjacency (tight lesion boundary)
    semi-global: k-hop adjacency (disease propagation context)
    gate α ∈ [0,1] interpolates between the two mask types.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Superpixel-Constrained Multi-Head Attention                                 #
# --------------------------------------------------------------------------- #

class SuperpixelConstrainedAttention(nn.Module):
    """
    Multi-head self-attention with additive superpixel adjacency masking.

    The mask is a per-image (B, N, N) float tensor:
        0.0   → allowed (adjacent superpixels)
       -inf   → blocked (non-adjacent)

    A learnable gate α blends local (1-hop) and semi-global (k-hop) masks,
    allowing the model to decide how far to propagate attention.

    Parameters
    ----------
    embed_dim  : total embedding dimension (D)
    num_heads  : H
    dropout    : attention dropout probability
    gate_init  : initial value of the gating scalar α
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        gate_init: float = 0.5,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim  = embed_dim
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.scale      = math.sqrt(self.head_dim)

        self.qkv      = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)

        # Learned scalar gate: interpolates local ↔ semi-global mask
        # sigmoid(gate_logit) → α ∈ (0, 1)
        gate_logit_init = math.log(gate_init / (1.0 - gate_init + 1e-6))
        self.gate_logit = nn.Parameter(torch.tensor(gate_logit_init))

    def forward(
        self,
        x: torch.Tensor,
        local_mask: Optional[torch.Tensor] = None,
        semiglobal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x               : (B, N, D)
        local_mask      : (B, N, N) float  — 1-hop adjacency mask
        semiglobal_mask : (B, N, N) float  — k-hop adjacency mask

        Returns
        -------
        out      : (B, N, D)
        attn_map : (B, H, N, N)  attention weights (for CAM)
        """
        B, N, D = x.shape
        H = self.num_heads

        # Q, K, V projections
        qkv = self.qkv(x).reshape(B, N, 3, H, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)         # (3, B, H, N, head_dim)
        q, k, v = qkv.unbind(0)                   # each: (B, H, N, head_dim)

        # Scaled dot-product attention logits
        attn_logits = (q @ k.transpose(-2, -1)) / self.scale   # (B, H, N, N)

        # Build blended adjacency mask
        if local_mask is not None and semiglobal_mask is not None:
            alpha = torch.sigmoid(self.gate_logit)
            # Blend: at α=0 → pure local, α=1 → pure semi-global
            # We blend the *allowance* (0 = allowed, -inf = blocked)
            # Treat as: blended = local_mask * (1-α) + semiglobal_mask * α
            # but since values are 0 or -inf, use maximum (union) for
            # semi-global and intersection for local — then lerp the logit
            blended_mask = (
                (1.0 - alpha) * local_mask +
                alpha          * semiglobal_mask
            )  # (B, N, N)
            # Expand for heads
            attn_logits = attn_logits + blended_mask.unsqueeze(1)
        elif local_mask is not None:
            attn_logits = attn_logits + local_mask.unsqueeze(1)

        # Softmax + dropout
        attn_weights = F.softmax(attn_logits, dim=-1)    # (B, H, N, N)
        attn_weights = self.attn_drop(attn_weights)

        # Weighted sum of values
        out = (attn_weights @ v).transpose(1, 2).reshape(B, N, D)  # (B, N, D)
        out = self.out_proj(out)

        return out, attn_weights


# --------------------------------------------------------------------------- #
#  Transformer Layer                                                           #
# --------------------------------------------------------------------------- #

class RegionAwareTransformerLayer(nn.Module):
    """
    Single Region-Aware Transformer block:
        LayerNorm → SuperpixelConstrainedAttention → residual
        LayerNorm → FFN (MLP) → residual
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        gate_init: float = 0.5,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn  = SuperpixelConstrainedAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            gate_init=gate_init,
        )
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        local_mask: Optional[torch.Tensor] = None,
        semiglobal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x               : (B, N, D)
        local_mask      : (B, N, N)
        semiglobal_mask : (B, N, N)

        Returns
        -------
        x        : (B, N, D)
        attn_map : (B, H, N, N)   from the attention sub-layer
        """
        normed = self.norm1(x)
        attn_out, attn_map = self.attn(normed, local_mask, semiglobal_mask)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_map


# --------------------------------------------------------------------------- #
#  Full Region-Aware Transformer                                               #
# --------------------------------------------------------------------------- #

class RegionAwareTransformer(nn.Module):
    """
    Stack of N RegionAwareTransformerLayers.

    Stores attention maps from ALL layers — they are used later
    by the CAM generator as multi-layer attention evidence.

    Parameters
    ----------
    num_layers  : int   — depth
    embed_dim   : int   — D
    num_heads   : int   — H
    mlp_ratio   : float — FFN expansion factor
    dropout     : float
    gate_init   : float — initial local/semi-global gating
    num_classes : int   — number of disease classes (for cls token)
    """

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        gate_init: float = 0.5,
        num_classes: int = 8,
    ):
        super().__init__()
        self.embed_dim   = embed_dim
        self.num_layers  = num_layers
        self.num_classes = num_classes

        # Learnable [CLS] token (prepended to the superpixel token sequence)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.layers = nn.ModuleList([
            RegionAwareTransformerLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                gate_init=gate_init,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Classification head (used in weakly-supervised mode)
        self.cls_head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        local_mask: Optional[torch.Tensor] = None,
        semiglobal_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Parameters
        ----------
        tokens          : (B, N, D)   superpixel tokens
        local_mask      : (B, N, N)   1-hop adjacency mask
        semiglobal_mask : (B, N, N)   k-hop adjacency mask

        Returns
        -------
        dict:
            'cls_logits'  : (B, num_classes)
            'tokens_out'  : (B, N, D)   final token features
            'cls_out'     : (B, D)      CLS token feature
            'attn_maps'   : list of (B, H, N+1, N+1) per layer
        """
        B, N, D = tokens.shape

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)         # (B, 1, D)
        x = torch.cat([cls, tokens], dim=1)             # (B, N+1, D)

        # Expand masks to include CLS row/col (CLS attends to all)
        if local_mask is not None:
            B, N_orig, _ = local_mask.shape
            extended_local = torch.zeros(
                B, N_orig + 1, N_orig + 1,
                device=local_mask.device, dtype=local_mask.dtype,
            )
            extended_local[:, 1:, 1:] = local_mask
            local_mask = extended_local

        if semiglobal_mask is not None:
            extended_sg = torch.zeros(
                B, N_orig + 1, N_orig + 1,
                device=semiglobal_mask.device, dtype=semiglobal_mask.dtype,
            )
            extended_sg[:, 1:, 1:] = semiglobal_mask
            semiglobal_mask = extended_sg

        attn_maps = []
        for layer in self.layers:
            x, attn = layer(x, local_mask, semiglobal_mask)
            attn_maps.append(attn)   # each: (B, H, N+1, N+1)

        x = self.norm(x)

        cls_out    = x[:, 0]      # (B, D)
        tokens_out = x[:, 1:]    # (B, N, D)

        cls_logits = self.cls_head(cls_out)   # (B, num_classes)

        return {
            'cls_logits': cls_logits,
            'tokens_out': tokens_out,
            'cls_out':    cls_out,
            'attn_maps':  attn_maps,
        }

    def get_averaged_attention(
        self,
        attn_maps: list,
        layers: Optional[list] = None,
    ) -> torch.Tensor:
        """
        Average attention maps across specified layers and heads.

        Parameters
        ----------
        attn_maps : list of (B, H, N+1, N+1)  one per layer
        layers    : list of layer indices to average (None = all)

        Returns
        -------
        avg_attn : (B, N+1, N+1)
        """
        if layers is None:
            layers = list(range(len(attn_maps)))

        selected = [attn_maps[i] for i in layers]
        stacked  = torch.stack(selected, dim=0)     # (L, B, H, N+1, N+1)
        return stacked.mean(dim=(0, 2))              # (B, N+1, N+1)

    def get_rollout_attention(self, attn_maps: list) -> torch.Tensor:
        """
        Attention Rollout (Abnar & Zuidema, 2020) for more accurate
        attention flow from CLS token to all superpixels.

        Returns
        -------
        rollout : (B, N)  — CLS-to-each-token attention (excl. CLS itself)
        """
        B, H, Np1, _ = attn_maps[0].shape
        rollout = torch.eye(Np1, device=attn_maps[0].device).unsqueeze(0).expand(B, -1, -1)

        for attn in attn_maps:
            # Average over heads
            attn_mean = attn.mean(dim=1)                      # (B, N+1, N+1)
            # Add residual (identity connection in attention rollout)
            attn_mean = 0.5 * attn_mean + 0.5 * torch.eye(
                Np1, device=attn.device
            ).unsqueeze(0)
            attn_mean = attn_mean / (attn_mean.sum(dim=-1, keepdim=True) + 1e-6)
            rollout = rollout @ attn_mean                     # (B, N+1, N+1)

        return rollout[:, 0, 1:]  # CLS row, drop CLS column → (B, N)
