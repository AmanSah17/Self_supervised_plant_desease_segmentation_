"""
Stage 5: CAM-Guided Attention Fusion Module.
Fuses structural anomaly evidence (Stage 3) with semantic class evidence (Stage 4).
Uses spatial attention to produce high-fidelity pseudo-masks.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional
import cv2

class CAMAttentionModule(nn.Module):
    """
    Spatial Attention Module for CAM and Anomaly Map Fusion.
    
    This module learns to weigh class evidence (CAM) against structural 
    deviation (Anomaly Map) to create a refined pseudo-mask.
    """
    def __init__(self, anomaly_weight: float = 0.7, cam_weight: float = 0.3):
        super().__init__()
        self.anomaly_weight = anomaly_weight
        self.cam_weight = cam_weight
        
    def forward(
        self, 
        anomaly_map: torch.Tensor, 
        cam_map: torch.Tensor, 
        fg_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse maps using attention.
        
        Parameters:
        -----------
        anomaly_map : (B, 1, H, W) normalized anomaly scores
        cam_map     : (B, 1, H, W) class activation map probabilities
        fg_mask     : (B, 1, H, W) binary foreground (leaf) mask
        
        Returns:
        --------
        refined_evidence : (B, 1, H, W) fused attention map
        """
        # Ensure all are on same device
        device = anomaly_map.device
        
        # 1. Structural Attention
        # High anomaly + High Foreground = Likely disease spot
        structural_attn = anomaly_map * fg_mask
        
        # 2. Semantic Modulation
        # Use CAM to boost the confidence of the specific disease class
        fused_evidence = (self.anomaly_weight * structural_attn) + (self.cam_weight * cam_map)
        
        # 3. Spatial Sharpening
        # Apply a sigmoid-like sharpening to the fusion
        refined_evidence = torch.sigmoid(10 * (fused_evidence - 0.5))
        
        return refined_evidence * fg_mask

class CAMAttentionRefiner:
    """
    Orchestrator for Stage 5 pseudo-mask refinement.
    Integrates CAM evidence with anomaly maps and superpixel boundaries.
    """
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.fusion_module = CAMAttentionModule().to(device)

    def refine(
        self,
        rgb_img: np.ndarray,
        anomaly_map: np.ndarray,
        cam_map: np.ndarray,
        fg_map: np.ndarray,
        segments: np.ndarray,
        cat_label: int,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Refine pseudo-mask using attention fusion and superpixel consensus.
        
        Parameters:
        -----------
        rgb_img     : (H, W, 3) Original image
        anomaly_map : (H, W) Normalized anomaly scores
        cam_map     : (H, W) Class specific CAM
        fg_map      : (H, W) Foreground probability map
        segments    : (H, W) Felzenszwalb superpixel segments
        cat_label   : int, Final segmentation label for the class
        
        Returns:
        --------
        pseudo_mask : (H, W) 3-way pseudo labels [0, 1, cat_label]
        """
        # Convert to tensors for fusion
        a_tensor = torch.from_numpy(anomaly_map).float().unsqueeze(0).unsqueeze(0).to(self.device)
        c_tensor = torch.from_numpy(cam_map).float().unsqueeze(0).unsqueeze(0).to(self.device)
        f_tensor = (torch.from_numpy(fg_map).float() > 0.35).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            fused = self.fusion_module(a_tensor, c_tensor, f_tensor)
            fused_np = fused.squeeze().cpu().numpy()
            
        # 3-way label generation
        pseudo_mask = np.zeros_like(segments, dtype=np.uint8)
        unique_segments = np.unique(segments)
        
        for seg_id in unique_segments:
            mask = (segments == seg_id)
            
            m_fg = fg_map[mask].mean()
            m_fuse = fused_np[mask].mean()
            
            if m_fg < 0.35:
                pseudo_mask[mask] = 0 # Background
            elif m_fuse >= threshold:
                pseudo_mask[mask] = cat_label # Category specific
            else:
                pseudo_mask[mask] = 1 # Healthy Leaf
                
        return pseudo_mask
