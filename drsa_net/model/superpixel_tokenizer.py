"""
DRSA-Net Superpixel Tokenizer
==============================
Converts dense feature maps (B, D, H', W') into superpixel token sequences
(B, N_sp, D) by scatter-pooling features within each Felzenszwalb region.

Also builds the superpixel adjacency masks needed by the Region-Aware
Transformer (Section 3 of the plan).

Key operations
--------------
1. scatter_mean: average all pixels inside each superpixel → one token
2. build_adjacency_mask: converts integer segment map → boolean adjacency
   matrix M_adj ∈ {0, 1}^(N_sp × N_sp)  (and k-hop extension)
3. detokenize: scatter tokens back to dense feature maps for CAM generation
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Scatter mean (without torch_scatter dependency)                             #
# --------------------------------------------------------------------------- #

def scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """
    Compute mean of `src` values for each segment index.

    Parameters
    ----------
    src   : (N, D)  — feature vectors for each pixel (flattened spatial)
    index : (N,)    — segment id for each pixel  [int64]
    num_segments : int

    Returns
    -------
    out : (num_segments, D)
    """
    D = src.shape[1]
    out   = torch.zeros(num_segments, D, device=src.device, dtype=src.dtype)
    count = torch.zeros(num_segments, 1, device=src.device, dtype=src.dtype)

    out.scatter_add_(0, index.unsqueeze(1).expand(-1, D), src)
    count.scatter_add_(0, index.unsqueeze(1), torch.ones(index.shape[0], 1,
                                                          device=src.device,
                                                          dtype=src.dtype))
    return out / (count + 1e-6)


def scatter_tokens_back(
    tokens: torch.Tensor,
    segments: torch.Tensor,
    h: int,
    w: int,
) -> torch.Tensor:
    """
    Broadcast superpixel tokens back to dense spatial map.

    Parameters
    ----------
    tokens   : (N_sp, D)
    segments : (H, W) int64
    h, w     : spatial dimensions

    Returns
    -------
    dense : (D, H, W)
    """
    flat_seg = segments.reshape(-1)       # (H*W,)
    dense_flat = tokens[flat_seg]          # (H*W, D)
    return dense_flat.reshape(h, w, -1).permute(2, 0, 1)   # (D, H, W)


# --------------------------------------------------------------------------- #
#  Adjacency mask builder                                                      #
# --------------------------------------------------------------------------- #

def build_adjacency_matrix(
    segments: torch.Tensor,
    num_segments: int,
    k_hop: int = 1,
) -> torch.Tensor:
    """
    Build a boolean adjacency matrix from an integer superpixel label map.

    Adjacency is defined by direct pixel neighbourship (4-connectivity):
      segments[i, j] is adjacent to segments[i, j+1], segments[i+1, j]

    Parameters
    ----------
    segments    : (H, W) int64 tensor
    num_segments: int  — total number of unique superpixels
    k_hop       : int  — if > 1, expand adjacency to k-hop neighbourhood

    Returns
    -------
    adj : (num_segments, num_segments) bool tensor
    """
    H, W = segments.shape
    N = num_segments
    device = segments.device

    adj = torch.zeros(N, N, dtype=torch.bool, device=device)

    # Horizontal neighbours
    s_left  = segments[:, :-1].reshape(-1)
    s_right = segments[:, 1: ].reshape(-1)
    mask_h  = s_left != s_right
    adj[s_left[mask_h], s_right[mask_h]] = True
    adj[s_right[mask_h], s_left[mask_h]] = True

    # Vertical neighbours
    s_top = segments[:-1, :].reshape(-1)
    s_bot = segments[1:,  :].reshape(-1)
    mask_v = s_top != s_bot
    adj[s_top[mask_v], s_bot[mask_v]] = True
    adj[s_bot[mask_v], s_top[mask_v]] = True

    # Self-loops
    adj.fill_diagonal_(True)

    # k-hop expansion via repeated matrix multiplication (boolean algebra)
    if k_hop > 1:
        adj_float = adj.float()
        reached = adj_float.clone()
        for _ in range(k_hop - 1):
            reached = torch.clamp(reached @ adj_float, 0, 1)
        adj = reached.bool()

    return adj


def adjacency_to_attention_mask(adj: torch.Tensor) -> torch.Tensor:
    """
    Convert boolean adjacency → additive attention bias.

        allowed (adj=True)  →   0.0
        blocked (adj=False) →  -inf

    Shape: (N_sp, N_sp)
    """
    mask = torch.full_like(adj, float('-inf'), dtype=torch.float32)
    mask[adj] = 0.0
    return mask


# --------------------------------------------------------------------------- #
#  Superpixel Tokenizer                                                        #
# --------------------------------------------------------------------------- #

class SuperpixelTokenizer(nn.Module):
    """
    Converts a dense feature map into a sequence of superpixel tokens.

    Additionally builds:
      - Local adjacency mask  (1-hop)
      - Semi-global adjacency mask (k-hop)
      - Positional embeddings (learnable, per-token centroid)

    Parameters
    ----------
    embed_dim    : int  — feature dimension D
    max_segments : int  — maximum number of superpixels (for embedding table)
    k_hop        : int  — semi-global attention neighbourhood size
    """

    def __init__(
        self,
        embed_dim: int,
        max_segments: int = 512,
        k_hop: int = 2,
    ):
        super().__init__()
        self.embed_dim    = embed_dim
        self.max_segments = max_segments
        self.k_hop        = k_hop

        # Learnable positional embedding table (index = segment id)
        self.pos_embedding = nn.Embedding(max_segments, embed_dim)
        nn.init.trunc_normal_(self.pos_embedding.weight, std=0.02)

    def forward(
        self,
        feature_map: torch.Tensor,
        segments: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Parameters
        ----------
        feature_map : (B, D, H', W')  — output of MultiStreamEncoder
        segments    : (B, H, W) int64 — superpixel label map (at original res)

        Returns
        -------
        tokens       : (B, N_sp, D)
        local_mask   : (B, N_sp, N_sp) — 1-hop attention mask (additive, 0 or -inf)
        semiglobal_mask : (B, N_sp, N_sp) — k-hop attention mask
        info         : dict with 'num_segments', 'segments_ds' (downsampled map)
        """
        B, D, Hf, Wf = feature_map.shape
        _, H, W = segments.shape

        # Downsample segments to feature map resolution
        seg_ds = F.interpolate(
            segments.float().unsqueeze(1), size=(Hf, Wf),
            mode='nearest',
        ).long().squeeze(1)   # (B, Hf, Wf)

        all_tokens:  List[torch.Tensor] = []
        all_local:   List[torch.Tensor] = []
        all_semiglo: List[torch.Tensor] = []
        all_N:       List[int] = []

        for b in range(B):
            fm_b  = feature_map[b]   # (D, Hf, Wf)
            seg_b = seg_ds[b]        # (Hf, Wf)

            # Remap segment ids to [0, N_sp)
            unique_ids, remapped = torch.unique(seg_b, return_inverse=True)
            N_sp = unique_ids.shape[0]
            N_sp = min(N_sp, self.max_segments)

            # Clamp remapped ids to max_segments
            remapped = remapped.clamp(0, N_sp - 1)   # (Hf, Wf)

            # Scatter mean: pixels → superpixel tokens
            flat_feat = fm_b.permute(1, 2, 0).reshape(-1, D)   # (Hf*Wf, D)
            flat_seg  = remapped.reshape(-1)                    # (Hf*Wf,)
            tok = scatter_mean(flat_feat, flat_seg, N_sp)       # (N_sp, D)

            # Positional embedding (by segment index)
            pos_ids = torch.arange(N_sp, device=feature_map.device)
            tok = tok + self.pos_embedding(pos_ids)

            # Build adjacency masks
            adj_local    = build_adjacency_matrix(remapped, N_sp, k_hop=1)
            adj_semiglobal = build_adjacency_matrix(remapped, N_sp, k_hop=self.k_hop)

            local_mask_b  = adjacency_to_attention_mask(adj_local)       # (N_sp, N_sp)
            semiglo_mask_b = adjacency_to_attention_mask(adj_semiglobal)  # (N_sp, N_sp)

            all_tokens.append(tok)
            all_local.append(local_mask_b)
            all_semiglo.append(semiglo_mask_b)
            all_N.append(N_sp)

        # Pad all sequences to the same N_sp for batching
        max_N = max(all_N)
        tokens_padded  = torch.zeros(B, max_N, D, device=feature_map.device)
        local_padded   = torch.full((B, max_N, max_N), float('-inf'),
                                    device=feature_map.device)
        semiglo_padded = torch.full((B, max_N, max_N), float('-inf'),
                                    device=feature_map.device)

        for b, (tok, lm, sg, n) in enumerate(
            zip(all_tokens, all_local, all_semiglo, all_N)
        ):
            tokens_padded[b, :n]     = tok
            local_padded[b, :n, :n]  = lm
            semiglo_padded[b, :n, :n] = sg
            # Diagonal of padded region should be 0 (don't mask self)
            for i in range(n, max_N):
                local_padded[b, i, i]   = 0.0
                semiglo_padded[b, i, i] = 0.0

        info = {
            'num_segments': all_N,
            'max_segments': max_N,
            'seg_ds':       seg_ds,
        }
        return tokens_padded, local_padded, semiglo_padded, info

    def detokenize(
        self,
        tokens: torch.Tensor,
        segments_ds: torch.Tensor,
        Hf: int,
        Wf: int,
    ) -> torch.Tensor:
        """
        Broadcast tokens back to dense feature maps.

        Parameters
        ----------
        tokens      : (B, N_sp, D)
        segments_ds : (B, Hf, Wf)  downsampled segment map
        Hf, Wf      : target spatial dims

        Returns
        -------
        dense : (B, D, Hf, Wf)
        """
        B, N_sp, D = tokens.shape
        dense = torch.zeros(B, D, Hf, Wf,
                            device=tokens.device, dtype=tokens.dtype)
        for b in range(B):
            _, remapped = torch.unique(segments_ds[b], return_inverse=True)
            remapped = remapped.clamp(0, N_sp - 1)
            dense[b] = scatter_tokens_back(tokens[b], remapped, Hf, Wf)
        return dense
