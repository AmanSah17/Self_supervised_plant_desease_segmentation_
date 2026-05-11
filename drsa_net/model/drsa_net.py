"""
DRSA-Net: Top-level model
=========================
Composes all stages into one nn.Module:

  MultiStreamEncoder
    → SuperpixelTokenizer
      → RegionAwareTransformer
        → DiseaseAwareCAMGenerator
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from drsa_net.config import DRSAConfig
from drsa_net.model.multistream_encoder   import MultiStreamEncoder
from drsa_net.model.superpixel_tokenizer  import SuperpixelTokenizer
from drsa_net.model.region_aware_transformer import RegionAwareTransformer
from drsa_net.model.cam_generator         import DiseaseAwareCAMGenerator


class DRSANet(nn.Module):
    """
    Disease-Region Superpixel-Aware Network (DRSA-Net).

    Forward pass returns a dict with all intermediate outputs needed
    for loss computation, CAM visualization, and metric logging.

    Parameters
    ----------
    config : DRSAConfig
    """

    def __init__(self, config: DRSAConfig):
        super().__init__()
        config.validate()
        self.config = config

        # Stage 1: Parallel multi-representation encoder
        self.encoder = MultiStreamEncoder(
            branch_channels=config.branch_channels,
            embed_dim=config.embed_dim,
        )

        # Stage 2: Superpixel tokenizer
        self.tokenizer = SuperpixelTokenizer(
            embed_dim=config.embed_dim,
            max_segments=config.max_superpixels,
            k_hop=config.attention_k_hop,
        )

        # Stage 3: Region-aware transformer
        self.transformer = RegionAwareTransformer(
            num_layers=config.num_transformer_layers,
            embed_dim=config.embed_dim,
            num_heads=config.num_attention_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            gate_init=config.gate_init,
            num_classes=config.num_classes,
        )

        # Stage 4-6: CAM generation pipeline
        self.cam_gen = DiseaseAwareCAMGenerator(
            cam_threshold=config.cam_threshold,
            propagation_alpha=config.cam_propagation_alpha,
            propagation_steps=config.cam_propagation_steps,
        )

    def forward(
        self,
        rgb:       torch.Tensor,
        clahe:     torch.Tensor,
        felz1:     torch.Tensor,
        felz2:     torch.Tensor,
        watershed: torch.Tensor,
        segments:  torch.Tensor,
        return_cam: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        rgb       : (B, 3, H, W)
        clahe     : (B, 3, H, W)
        felz1     : (B, 1, H, W)
        felz2     : (B, 1, H, W)
        watershed : (B, 1, H, W)
        segments  : (B, H, W)   int64 superpixel label map
        return_cam: bool  — set False during pure training for speed

        Returns
        -------
        dict with keys:
            'cls_logits'     : (B, num_classes)
            'tokens_out'     : (B, N_sp, D)
            'attn_maps'      : list of (B, H, N+1, N+1) per layer
            'rollout'        : (B, N_sp)
            'feature_map'    : (B, D, H/4, W/4)
            --- if return_cam=True ---
            'raw_cam'        : (B, 1, H, W)
            'grown_cam'      : (B, 1, H, W)
            'propagated_cam' : (B, 1, H, W)
            'fusion_weights' : (B, 4)
        """
        B, _, H, W = rgb.shape

        # ── Stage 1: Parallel Encoder ──────────────────────────────────────
        feature_map = self.encoder(rgb, clahe, felz1, felz2, watershed)
        # (B, D, H/4, W/4)

        # ── Stage 2: Superpixel Tokenization ──────────────────────────────
        tokens, local_mask, semiglobal_mask, tok_info = self.tokenizer(
            feature_map, segments,
        )
        # tokens : (B, N_sp, D)
        # masks  : (B, N_sp, N_sp)

        # ── Stage 3: Region-Aware Transformer ─────────────────────────────
        trans_out = self.transformer(tokens, local_mask, semiglobal_mask)
        # trans_out keys: cls_logits, tokens_out, cls_out, attn_maps

        # Attention Rollout → per-superpixel scores for CAM
        rollout = self.transformer.get_rollout_attention(trans_out['attn_maps'])
        # (B, N_sp)

        # Build adjacency matrix from superpixel tokenizer (local, 1-hop)
        # We reuse local_mask (0 → adjacent, -inf → non-adjacent) and
        # convert it to a boolean adjacency matrix
        adj_bool = (local_mask == 0.0)   # (B, N_sp, N_sp)

        out = {
            'cls_logits':  trans_out['cls_logits'],
            'tokens_out':  trans_out['tokens_out'],
            'cls_out':     trans_out['cls_out'],
            'attn_maps':   trans_out['attn_maps'],
            'rollout':     rollout,
            'feature_map': feature_map,
            'seg_ds':      tok_info['seg_ds'],
        }

        # ── Stages 4-6: CAM Generation ────────────────────────────────────
        if return_cam:
            # ws_map and felz2 need channel dim → (B,1,H,W)
            ws_4d   = watershed          # already (B,1,H,W)
            felz2_4d = felz2             # already (B,1,H,W)
            clahe_4d = clahe             # (B,3,H,W)

            cam_out = self.cam_gen(
                rollout_scores=rollout,
                ws_map=ws_4d,
                felz2=felz2_4d,
                clahe=clahe_4d,
                segments_ds=tok_info['seg_ds'],
                adj_matrix=adj_bool.float(),
                target_size=(H, W),
            )
            out.update(cam_out)

        return out

    @torch.no_grad()
    def predict_cam(
        self,
        rgb:       torch.Tensor,
        clahe:     torch.Tensor,
        felz1:     torch.Tensor,
        felz2:     torch.Tensor,
        watershed: torch.Tensor,
        segments:  torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Inference-only forward pass. Returns all CAM variants."""
        self.eval()
        return self.forward(
            rgb, clahe, felz1, felz2, watershed, segments, return_cam=True,
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        total = self.count_parameters()
        lines = [
            "=" * 60,
            "DRSA-Net Architecture Summary",
            "=" * 60,
            f"  Embed dim        : {self.config.embed_dim}",
            f"  Branch channels  : {self.config.branch_channels}",
            f"  Transformer layers: {self.config.num_transformer_layers}",
            f"  Attention heads  : {self.config.num_attention_heads}",
            f"  Max superpixels  : {self.config.max_superpixels}",
            f"  k-hop attention  : {self.config.attention_k_hop}",
            f"  CAM prop steps   : {self.config.cam_propagation_steps}",
            f"  Trainable params : {total:,}",
            "=" * 60,
        ]
        return "\n".join(lines)
