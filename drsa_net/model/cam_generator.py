"""
DRSA-Net Disease-Aware CAM Generator
======================================
Fuses multiple evidence sources into a disease-aware Class Activation Map:

  1. Transformer Attention Maps (rollout over all layers)
  2. Watershed Gradient Maps
  3. Felzenszwalb Region Boundaries
  4. CLAHE Lesion Enhancement Saliency
  ↓
  Weighted Fusion (weights learned by a small MLP)
  ↓
  Region-Growing CAM Refinement
  ↓
  Graph-Propagated CAM (Y^(t+1) = αAY^(t) + (1-α)Y^(0))

All operations are differentiable so gradients flow back to the encoder.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Helper: dense CAM from superpixel token scores                             #
# --------------------------------------------------------------------------- #

def superpixel_scores_to_dense(
    scores: torch.Tensor,
    segments_ds: torch.Tensor,
    Hf: int,
    Wf: int,
) -> torch.Tensor:
    """
    Broadcast per-superpixel scalar scores to a dense spatial map.

    Parameters
    ----------
    scores      : (B, N_sp)   — one score per superpixel
    segments_ds : (B, Hf, Wf) — downsampled segment integer map
    Hf, Wf      : target spatial dims

    Returns
    -------
    dense : (B, 1, Hf, Wf)
    """
    B, N_sp = scores.shape
    dense = torch.zeros(B, 1, Hf, Wf, device=scores.device, dtype=scores.dtype)
    for b in range(B):
        _, remapped = torch.unique(segments_ds[b], return_inverse=True)
        remapped = remapped.clamp(0, N_sp - 1)              # (Hf*Wf,)
        # Gather scores per pixel
        pixel_scores = scores[b][remapped].reshape(Hf, Wf)
        dense[b, 0] = pixel_scores
    return dense


# --------------------------------------------------------------------------- #
#  Multi-Representation CAM Fusion                                             #
# --------------------------------------------------------------------------- #

class MultiRepCAMFusion(nn.Module):
    """
    Fuses 4 evidence maps into a single disease-aware CAM using
    learned per-image weights (MLP fusion).

    Evidence inputs (all at feature map resolution H', W'):
        attn_cam  : (B, 1, H', W')  — from transformer attention rollout
        ws_map    : (B, 1, H', W')  — watershed gradient map
        felz_map  : (B, 1, H', W')  — Felzenszwalb boundary map
        clahe_sal : (B, 1, H', W')  — CLAHE saliency (L-channel gradient)

    Weight prediction:
        Global avg-pool each map → 4 scalars → MLP → 4 softmax weights

    Output:
        fused_cam : (B, 1, H', W')  in [0,1]
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        # MLP that predicts 4 fusion weights from global evidence statistics
        self.weight_mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(
        self,
        attn_cam:  torch.Tensor,
        ws_map:    torch.Tensor,
        felz_map:  torch.Tensor,
        clahe_sal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        fused_cam  : (B, 1, H', W')
        weights    : (B, 4)   fusion weights (for logging)
        """
        # Stack maps: (B, 4, H', W')
        maps = torch.cat([attn_cam, ws_map, felz_map, clahe_sal], dim=1)

        # Normalize each map to [0,1]
        maps_min = maps.flatten(2).min(dim=2)[0].unsqueeze(-1).unsqueeze(-1)
        maps_max = maps.flatten(2).max(dim=2)[0].unsqueeze(-1).unsqueeze(-1)
        maps_norm = (maps - maps_min) / (maps_max - maps_min + 1e-6)

        # Global average pool for weight prediction
        gap = maps_norm.mean(dim=(2, 3))      # (B, 4)
        weights = F.softmax(self.weight_mlp(gap), dim=-1)  # (B, 4)

        # Weighted sum
        w = weights.unsqueeze(-1).unsqueeze(-1)  # (B, 4, 1, 1)
        fused = (maps_norm * w).sum(dim=1, keepdim=True)  # (B, 1, H', W')
        fused = fused.clamp(0, 1)

        return fused, weights


# --------------------------------------------------------------------------- #
#  CLAHE Saliency (gradient of CLAHE L-channel)                               #
# --------------------------------------------------------------------------- #

class CLAHESaliencyExtractor(nn.Module):
    """
    Compute a saliency map from the CLAHE branch feature map
    using gradient magnitude.
    """

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        ).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, clahe_tensor: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        clahe_tensor : (B, 3, H, W) or (B, 1, H, W) CLAHE image / feature

        Returns
        -------
        saliency : (B, 1, H, W) normalized gradient magnitude
        """
        # Use first channel (L-channel proxy after RGB normalisation)
        x = clahe_tensor[:, :1]  # (B, 1, H, W)
        x_pad = F.pad(x, (1, 1, 1, 1), mode='reflect')
        gx = F.conv2d(x_pad, self.sobel_x)
        gy = F.conv2d(x_pad, self.sobel_y)
        mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        # Normalize per image
        b = mag.shape[0]
        mag_flat = mag.reshape(b, -1)
        mag_min  = mag_flat.min(dim=1)[0].view(b, 1, 1, 1)
        mag_max  = mag_flat.max(dim=1)[0].view(b, 1, 1, 1)
        return (mag - mag_min) / (mag_max - mag_min + 1e-6)


# --------------------------------------------------------------------------- #
#  Region-Growing CAM Refinement                                               #
# --------------------------------------------------------------------------- #

class RegionGrowingCAM(nn.Module):
    """
    Region Growing: if a CAM activates only PART of a superpixel,
    expand the activation to the ENTIRE superpixel region.

    Algorithm
    ---------
    1. For each superpixel, compute mean CAM score
    2. If mean score > threshold → mark entire superpixel as activated
    3. Replace pixel-level CAM with superpixel-level binary expansion

    This is the "Region Growing CAM" from weakly-supervised segmentation.
    """

    def __init__(self, threshold: float = 0.3):
        super().__init__()
        self.threshold = threshold

    def forward(
        self,
        cam: torch.Tensor,
        segments_ds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        cam         : (B, 1, H', W')  raw fused CAM
        segments_ds : (B, H', W')     downsampled segment map

        Returns
        -------
        refined_cam : (B, 1, H', W')
        """
        B, _, Hf, Wf = cam.shape
        refined = cam.clone()

        for b in range(B):
            cam_b = cam[b, 0]          # (H', W')
            seg_b = segments_ds[b]     # (H', W')

            unique_ids = seg_b.unique()
            for sid in unique_ids:
                mask = (seg_b == sid)
                region_score = cam_b[mask].mean()
                if region_score > self.threshold:
                    # Expand: set entire superpixel to max CAM value in region
                    refined[b, 0][mask] = cam_b[mask].max()

        return refined.clamp(0, 1)


# --------------------------------------------------------------------------- #
#  Graph-Propagated CAM Refinement                                             #
# --------------------------------------------------------------------------- #

class GraphCAMPropagation(nn.Module):
    """
    Graph-Based CAM Propagation:

        Y^(t+1) = α · A · Y^(t) + (1 - α) · Y^(0)

    where:
        A   = row-normalised adjacency matrix (from superpixel graph)
        Y^0 = initial CAM scores per superpixel
        α   = propagation strength (learnable, initialised from config)
        T   = number of propagation steps

    This spreads activation across connected lesion regions.
    """

    def __init__(self, alpha: float = 0.5, num_steps: int = 3):
        super().__init__()
        # Learnable propagation strength
        self.log_alpha = nn.Parameter(torch.tensor(alpha).log())
        self.num_steps = num_steps

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.log_alpha)

    def forward(
        self,
        cam_scores: torch.Tensor,
        adj_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        cam_scores : (B, N_sp)      per-superpixel CAM scores
        adj_matrix : (B, N_sp, N_sp) boolean or float adjacency

        Returns
        -------
        propagated : (B, N_sp)
        """
        # Row-normalise adjacency  A_norm[i,j] = adj[i,j] / sum_j adj[i,j]
        adj_float = adj_matrix.float()
        row_sum   = adj_float.sum(dim=-1, keepdim=True).clamp(min=1.0)
        A_norm    = adj_float / row_sum                     # (B, N_sp, N_sp)

        Y  = cam_scores.unsqueeze(-1)    # (B, N_sp, 1)
        Y0 = Y.clone()
        alpha = self.alpha

        for _ in range(self.num_steps):
            Y = alpha * (A_norm @ Y) + (1.0 - alpha) * Y0

        return Y.squeeze(-1).clamp(0, 1)   # (B, N_sp)


# --------------------------------------------------------------------------- #
#  Full CAM Generator                                                          #
# --------------------------------------------------------------------------- #

class DiseaseAwareCAMGenerator(nn.Module):
    """
    End-to-end differentiable CAM pipeline.

    Input:
        - Transformer attention maps (list of per-layer attention tensors)
        - Attention rollout scores per superpixel (B, N_sp)
        - Raw input tensors: ws_map, felz2 (boundary), clahe
        - Superpixel info: segments_ds, num_segments
        - Adjacency matrix (B, N_sp, N_sp)

    Output:
        - final_cam_dense : (B, 1, H, W)  at original image resolution
        - intermediate maps for logging
    """

    def __init__(
        self,
        cam_threshold: float = 0.3,
        propagation_alpha: float = 0.5,
        propagation_steps: int = 3,
    ):
        super().__init__()
        self.clahe_sal     = CLAHESaliencyExtractor()
        self.fusion        = MultiRepCAMFusion(hidden_dim=64)
        self.region_grow   = RegionGrowingCAM(threshold=cam_threshold)
        self.graph_prop    = GraphCAMPropagation(
            alpha=propagation_alpha,
            num_steps=propagation_steps,
        )

    def forward(
        self,
        rollout_scores: torch.Tensor,
        ws_map: torch.Tensor,
        felz2: torch.Tensor,
        clahe: torch.Tensor,
        segments_ds: torch.Tensor,
        adj_matrix: torch.Tensor,
        target_size: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        rollout_scores : (B, N_sp)          from transformer rollout
        ws_map         : (B, 1, H, W)       watershed map (original res)
        felz2          : (B, 1, H, W)       Felzenszwalb boundary map
        clahe          : (B, 3, H, W)       CLAHE image
        segments_ds    : (B, Hf, Wf)        downsampled segment map
        adj_matrix     : (B, N_sp, N_sp)    adjacency (bool or float)
        target_size    : (H, W)             final output resolution

        Returns
        -------
        dict:
            'raw_cam'        : (B, 1, H, W)
            'grown_cam'      : (B, 1, H, W)
            'propagated_cam' : (B, 1, H, W)
            'fusion_weights' : (B, 4)
        """
        B = rollout_scores.shape[0]
        _, _, Hf, Wf = segments_ds.unsqueeze(1).shape if segments_ds.dim() == 3 \
            else (B, 1, *segments_ds.shape[1:])
        Hf, Wf = segments_ds.shape[1], segments_ds.shape[2]
        H, W   = target_size

        # -- Step 1: Convert rollout scores → dense attention CAM ----------
        attn_cam = superpixel_scores_to_dense(
            rollout_scores, segments_ds, Hf, Wf,
        )  # (B, 1, Hf, Wf)
        attn_cam = F.interpolate(attn_cam, size=(H, W), mode='bilinear',
                                 align_corners=False)

        # Resize auxiliary maps to target
        ws_resized   = F.interpolate(ws_map,   size=(H, W), mode='bilinear', align_corners=False)
        felz_resized = F.interpolate(felz2,    size=(H, W), mode='bilinear', align_corners=False)

        # -- Step 2: CLAHE saliency ----------------------------------------
        clahe_resized = F.interpolate(clahe, size=(H, W), mode='bilinear', align_corners=False)
        clahe_sal = self.clahe_sal(clahe_resized)   # (B, 1, H, W)

        # -- Step 3: Multi-representation fusion ---------------------------
        raw_cam, fusion_weights = self.fusion(
            attn_cam, ws_resized, felz_resized, clahe_sal,
        )  # (B, 1, H, W)

        # -- Step 4: Region-Growing CAM ------------------------------------
        # Upsample segment map to target size for region growing
        seg_target = F.interpolate(
            segments_ds.float().unsqueeze(1), size=(H, W), mode='nearest',
        ).long().squeeze(1)

        grown_cam = self.region_grow(raw_cam, seg_target)  # (B, 1, H, W)

        # -- Step 5: Graph-Propagated CAM ----------------------------------
        # Convert grown CAM → per-superpixel scores at feature map resolution
        grown_cam_ds = F.interpolate(
            grown_cam, size=(Hf, Wf), mode='bilinear', align_corners=False,
        )
        cam_scores = torch.zeros(B, adj_matrix.shape[1],
                                 device=rollout_scores.device)
        for b in range(B):
            segs = segments_ds[b]
            unique_ids = segs.unique()
            cam_b = grown_cam_ds[b, 0]
            for i, sid in enumerate(unique_ids):
                if i >= cam_scores.shape[1]:
                    break
                cam_scores[b, i] = cam_b[segs == sid].mean()

        propagated_scores = self.graph_prop(cam_scores, adj_matrix)  # (B, N_sp)

        # Back to dense
        propagated_cam = superpixel_scores_to_dense(
            propagated_scores, segments_ds, Hf, Wf,
        )
        propagated_cam = F.interpolate(
            propagated_cam, size=(H, W), mode='bilinear', align_corners=False,
        )

        return {
            'raw_cam':        raw_cam,
            'grown_cam':      grown_cam,
            'propagated_cam': propagated_cam,
            'fusion_weights': fusion_weights,
            'attn_cam':       attn_cam,
            'clahe_sal':      clahe_sal,
        }
