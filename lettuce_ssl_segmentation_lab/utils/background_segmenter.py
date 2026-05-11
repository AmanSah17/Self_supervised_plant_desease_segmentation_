"""
Background Segmenter for Lettuce Leaves.
Uses DINOv2 features and color priors to separate foreground (leaf/weed) from background (soil).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from sklearn.decomposition import PCA

class DINOv2BackgroundSegmenter:
    """
    Separates foreground objects (lettuce, weeds) from background (soil).
    Combines DINOv2 patch PCA and HSV color masking.
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device

    def segment(self, image_rgb: np.ndarray, patch_features: torch.Tensor) -> np.ndarray:
        """
        Produce a foreground probability map.
        image_rgb: (H, W, 3) numpy array
        patch_features: (1, D, H_p, W_p) tensor
        returns: (H, W) probability map [0, 1]
        """
        # 1. DINOv2 Structural Evidence (PCA)
        # Patch features: (1, D, H_p, W_p) -> (H_p*W_p, D)
        B, D, H_p, W_p = patch_features.shape
        features_flat = patch_features[0].permute(1, 2, 0).reshape(-1, D).cpu().numpy()
        
        # PCA to 1 component
        pca = PCA(n_components=1)
        pca_feat = pca.fit_transform(features_flat) # (N, 1)
        pca_feat = pca_feat.reshape(H_p, W_p)
        
        # Normalize PCA feature
        pca_min, pca_max = pca_feat.min(), pca_feat.max()
        pca_norm = (pca_feat - pca_min) / (pca_max - pca_min + 1e-8)
        
        # Invert if necessary (sometimes PCA 1st component is background)
        # We assume the center of the image is more likely to be foreground
        center_h, center_w = H_p // 2, W_p // 2
        if pca_norm[center_h, center_w] < 0.5:
            pca_norm = 1.0 - pca_norm
            
        # Upsample PCA map to full resolution
        pca_full = cv2.resize(pca_norm, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # 2. Color Evidence (HSV Green Mask)
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        # Define range for green (lettuce)
        lower_green = np.array([30, 40, 40])
        upper_green = np.array([90, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green).astype(np.float32) / 255.0
        
        # 3. Fusion
        # Fg = PCA * (1 + GreenMask) / 2
        fg_map = (pca_full * 0.6) + (green_mask * 0.4)
        
        return fg_map

    def get_binary_mask(self, fg_map: np.ndarray, threshold: float = 0.4) -> np.ndarray:
        """Get binary foreground mask."""
        mask = (fg_map >= threshold).astype(np.uint8)
        
        # Post-processing: Remove small noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
