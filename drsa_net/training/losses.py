"""
DRSA-Net Self-Supervised + Weakly-Supervised Loss Functions
============================================================

1. ClassificationLoss      — Cross-entropy (weakly supervised, class labels)
2. CAMConsistencyLoss      — CAMs from two augmented views should agree
3. ContrastiveLoss         — NT-Xent (SimCLR-style) on CLS tokens
4. SuperpixelCompactnessLoss — tokens of same pred class should cluster
5. GraphSmoothnessLoss     — adjacent superpixels → similar token features

Total loss:
    L = w_cls * L_cls
      + w_cam * L_cam_consist
      + w_ctr * L_contrastive
      + w_cmp * L_compact
      + w_gs  * L_graph_smooth
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from drsa_net.config import DRSAConfig


# --------------------------------------------------------------------------- #
#  Adaptive Multi-Task Loss Weighting (Kendall et al.)                        #
# --------------------------------------------------------------------------- #

class AdaptiveMTLLoss(nn.Module):
    """
    Homoscedastic Uncertainty Weighting for Multi-Task Learning.
    Dynamically balances multiple loss functions by learning a log-variance
    parameter for each task.
    
    Formula: L = sum( exp(-log_var_i) * L_i + log_var_i )
    """
    def __init__(self, num_tasks: int = 5):
        super().__init__()
        # Initialize log variances at 0 (meaning weight is exp(0) = 1)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        
    def forward(self, losses: list[torch.Tensor]) -> torch.Tensor:
        total_loss = 0.0
        for i, loss in enumerate(losses):
            # To prevent extreme values, clamp log_vars during forward
            log_var = torch.clamp(self.log_vars[i], min=-10.0, max=10.0)
            # Add 0.5 * exp(-log_var) * loss + 0.5 * log_var
            # (The 0.5 is a constant and can be omitted, we use the standard implementation)
            weight = torch.exp(-log_var)
            total_loss += weight * loss + log_var
            
        return total_loss

    def get_weights(self) -> list[float]:
        """Returns the effective weights exp(-log_var) for logging."""
        with torch.no_grad():
            return torch.exp(-self.log_vars).tolist()


# --------------------------------------------------------------------------- #
#  Individual Task Losses                                                      #
# --------------------------------------------------------------------------- #

class ClassificationLoss(nn.Module):
    """Standard cross-entropy on the [CLS] token logits."""

    def __init__(self, num_classes: int, label_smoothing: float = 0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return self.ce(logits, labels)


# --------------------------------------------------------------------------- #
#  2. CAM Consistency Loss                                                     #
# --------------------------------------------------------------------------- #

class CAMConsistencyLoss(nn.Module):
    """
    CAMs produced from two augmented views of the same image should agree.

    L_cam = MSE(CAM_view1, CAM_view2.detach())

    We detach view2 to use it as a target (stop-gradient on the weaker view).
    """

    def forward(
        self,
        cam1: torch.Tensor,
        cam2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        cam1, cam2 : (B, 1, H, W)  CAMs from two augmented views
        """
        return F.mse_loss(cam1, cam2.detach())


# --------------------------------------------------------------------------- #
#  3. NT-Xent Contrastive Loss (SimCLR-style)                                #
# --------------------------------------------------------------------------- #

class ContrastiveLoss(nn.Module):
    """
    NT-Xent loss on [CLS] token embeddings.

    Positive pair: (view1_cls, view2_cls) of the same image.
    Negatives:      all other embeddings in the batch.

    L = -log( exp(sim(z_i, z_j) / τ) / Σ_{k≠i} exp(sim(z_i, z_k) / τ) )
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        z1, z2 : (B, D)  L2-normalised CLS token embeddings
        """
        B, D = z1.shape
        # L2 normalise
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # Concatenate: (2B, D)
        z = torch.cat([z1, z2], dim=0)

        # Similarity matrix: (2B, 2B)
        sim = torch.mm(z, z.T) / self.temperature

        # Mask out self-similarity
        mask = torch.eye(2 * B, device=z.device).bool()
        sim.masked_fill_(mask, float('-inf'))

        # Positive pairs: (i, B+i) and (B+i, i)
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(0, B,     device=z.device),
        ])

        return F.cross_entropy(sim, labels)


# --------------------------------------------------------------------------- #
#  4. Superpixel Compactness Loss                                              #
# --------------------------------------------------------------------------- #

class SuperpixelCompactnessLoss(nn.Module):
    """
    Tokens predicted to belong to the same disease class should be
    close together in embedding space.

    For each class predicted with high probability:
        variance of token embeddings within that class should be small.

    L_compact = mean_over_classes( var of tokens in class )
    """

    def forward(
        self,
        tokens: torch.Tensor,
        cls_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        tokens     : (B, N, D)
        cls_logits : (B, num_classes)  — from [CLS] head (not per-token)

        We use Gumbel-softmax to get differentiable soft assignments
        based on the overall image classification as a prior.
        """
        # Soft class probabilities
        probs = F.softmax(cls_logits, dim=-1)   # (B, num_classes)

        # Compute token variance weighted by class confidence
        # (simple proxy: cosine variance across tokens within batch)
        tokens_norm = F.normalize(tokens, dim=-1)   # (B, N, D)
        # Mean token per batch
        mean_tok = tokens_norm.mean(dim=1, keepdim=True)  # (B, 1, D)
        variance = ((tokens_norm - mean_tok) ** 2).mean()

        # Weight by max class confidence (higher confidence → push compactness)
        max_conf = probs.max(dim=-1)[0].mean()
        return variance * max_conf


# --------------------------------------------------------------------------- #
#  5. Graph Smoothness Loss                                                    #
# --------------------------------------------------------------------------- #

class GraphSmoothnessLoss(nn.Module):
    """
    Adjacent superpixels with similar visual features should have
    similar token representations.

    L_smooth = Σ_{(i,j) ∈ E} adj_weight_{ij} * ||f_i - f_j||²

    adj_weight_{ij} = edge weight from superpixel adjacency graph
    (closer features = higher weight = stronger smoothness pressure)
    """

    def forward(
        self,
        tokens: torch.Tensor,
        adj_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        tokens     : (B, N, D)
        adj_matrix : (B, N, N) float  — 0/1 or weighted adjacency
        """
        # Pairwise L2 distances between tokens: (B, N, N)
        # ||f_i - f_j||² = ||f_i||² + ||f_j||² - 2 f_i·f_j
        f  = tokens                                       # (B, N, D)
        ff = (f * f).sum(dim=-1, keepdim=True)           # (B, N, 1)
        dist_sq = ff + ff.transpose(1, 2) - 2 * torch.bmm(f, f.transpose(1, 2))
        dist_sq = dist_sq.clamp(min=0)                   # (B, N, N)

        # Weight by adjacency, normalise
        adj = adj_matrix.float()
        num_edges = adj.sum().clamp(min=1.0)
        return (adj * dist_sq).sum() / num_edges


# --------------------------------------------------------------------------- #
#  Combined Loss                                                               #
# --------------------------------------------------------------------------- #

class DRSALoss(nn.Module):
    """
    Weighted combination of all DRSA-Net losses.

    In training mode, the forward receives outputs from TWO augmented
    views of the same batch, enabling CAM consistency and contrastive loss.
    """

    def __init__(self, config: DRSAConfig):
        super().__init__()
        self.config = config

        self.cls_loss     = ClassificationLoss(config.num_classes)
        self.cam_consist  = CAMConsistencyLoss()
        self.contrastive  = ContrastiveLoss(config.contrastive_temperature)
        self.compactness  = SuperpixelCompactnessLoss()
        self.graph_smooth = GraphSmoothnessLoss()        # Multi-Task Learning Wrapper
        self.adaptive_mtl = AdaptiveMTLLoss(num_tasks=5)

    def forward(
        self,
        out1: Dict[str, torch.Tensor],
        out2: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        adj_matrix: torch.Tensor,
        return_components: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        out1, out2  : model outputs from two augmented views
        labels      : (B,) int64 class labels
        adj_matrix  : (B, N_sp, N_sp) adjacency matrix
        return_components : if True, return individual loss terms

        Returns
        -------
        dict with 'total' and optionally individual terms
        """
        losses = {}

        # Classification (only from view 1)
        if self.config.training_mode == 'weakly_supervised':
            l_cls = self.cls_loss(out1['cls_logits'], labels)
            losses['cls'] = l_cls
        else:
            l_cls = torch.tensor(0.0, device=labels.device)
            losses['cls'] = l_cls

        # CAM consistency
        if 'propagated_cam' in out1 and 'propagated_cam' in out2:
            l_cam = self.cam_consist(out1['propagated_cam'], out2['propagated_cam'])
            losses['cam_consist'] = l_cam
        else:
            l_cam = torch.tensor(0.0, device=labels.device)
            losses['cam_consist'] = l_cam

        # Contrastive (CLS tokens)
        l_cont = self.contrastive(out1['cls_out'], out2['cls_out'])
        losses['contrastive'] = l_cont

        # Superpixel compactness
        l_comp = self.compactness(out1['tokens_out'], out1['cls_logits'])
        losses['compactness'] = l_comp

        # Graph smoothness
        l_gs = self.graph_smooth(out1['tokens_out'], adj_matrix.float())
        losses['graph_smooth'] = l_gs

        # Check for NaN in any component
        for k, v in losses.items():
            if torch.isnan(v).any():
                print(f"  [ERROR] NaN detected in {k} loss component!")
                # Fallback to zero to prevent total collapse during debug
                losses[k] = torch.tensor(0.0, device=labels.device, requires_grad=True)

        # Adaptive Multi-Task weighting
        loss_list = [l_cls, l_cam, l_cont, l_comp, l_gs]
        total = self.adaptive_mtl(loss_list)

        losses['total'] = total
        return losses if return_components else total

    def get_mtl_weights(self):
        """Return dynamic weights for logging."""
        return self.adaptive_mtl.get_weights()
