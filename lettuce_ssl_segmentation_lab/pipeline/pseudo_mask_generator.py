"""
Multi-Class Pseudo Mask Generator.
Fuses background evidence, anomaly maps, and class-specific CAMs.
Produces 3-way labels: [Background, Healthy Leaf, Category-Specific].
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
import cv2

class PseudoMaskGenerator:
    """
    Fuses multiple evidence sources to generate multi-class pseudo masks.
    Output mapping:
    - 0: Background
    - 1: Healthy Leaf
    - class_idx + 1: Category-Specific (Disease or Weed)
    """
    
    def __init__(
        self, 
        anomaly_weight: float = 0.8, 
        cam_weight: float = 0.2,
        fg_threshold: float = 0.35,
        cat_threshold: float = 0.55
    ):
        self.anomaly_weight = anomaly_weight
        self.cam_weight = cam_weight
        self.fg_threshold = fg_threshold
        self.cat_threshold = cat_threshold

    def generate_multi_class(
        self, 
        anomaly_map: np.ndarray, 
        cam_map: np.ndarray, 
        fg_map: np.ndarray,
        segments: np.ndarray,
        target_class_idx: int
    ) -> Dict[str, np.ndarray]:
        """
        Generate 3-way pseudo mask.
        target_class_idx: The class index from the classifier head (e.g., 0-7)
        returns: Dict with fused map and pseudo mask
        """
        # 1. Normalize anomaly map
        am_min, am_max = anomaly_map.min(), anomaly_map.max()
        if am_max > am_min:
            anomaly_norm = (anomaly_map - am_min) / (am_max - am_min)
        else:
            anomaly_norm = anomaly_map

        # 2. Category Fusion (Disease/Weed evidence)
        cat_evidence = (self.anomaly_weight * anomaly_norm) + (self.cam_weight * cam_map)
        
        # 3. Superpixel consensus
        # Final mask values: 0 (Bg), 1 (Hlty), (target_class_idx + 1) (Cat)
        # Note: We need a mapping from target_class_idx to final segmentation label.
        # For simplicity, we'll use target_class_idx + 1 if it's not HLTY, or 1 if it is HLTY.
        # But wait, HLTY images should just be all 1.
        
        final_mask = np.zeros_like(segments, dtype=np.uint8)
        unique_segments = np.unique(segments)
        
        # Categorize each superpixel
        for seg_id in unique_segments:
            mask = (segments == seg_id)
            
            mean_fg = fg_map[mask].mean()
            mean_cat = cat_evidence[mask].mean()
            
            if mean_fg < self.fg_threshold:
                final_mask[mask] = 0 # Background
            elif mean_cat >= self.cat_threshold:
                # This is a bit tricky: what is the label?
                # We'll pass the correct label from the orchestrator
                pass 
            else:
                final_mask[mask] = 1 # Healthy Leaf
                
        return {
            "cat_evidence": cat_evidence,
            "fg_map": fg_map,
            # final_mask is partially filled, needs the category label
        }

    def refine_multi_class(
        self,
        fg_map: np.ndarray,
        cat_evidence: np.ndarray,
        segments: np.ndarray,
        cat_label: int,
        is_healthy_image: bool = False
    ) -> np.ndarray:
        """
        Full refinement with target class label.
        cat_label: The final segmentation label (e.g. 2 for BACT, 8 for SPW)
        """
        refined_mask = np.zeros_like(segments, dtype=np.uint8)
        unique_segments = np.unique(segments)
        
        for seg_id in unique_segments:
            mask = (segments == seg_id)
            
            m_fg = fg_map[mask].mean()
            m_cat = cat_evidence[mask].mean()
            
            if m_fg < self.fg_threshold:
                refined_mask[mask] = 0 # Background
            else:
                if not is_healthy_image and m_cat >= self.cat_threshold:
                    refined_mask[mask] = cat_label # Category-specific (Disease/Weed)
                else:
                    refined_mask[mask] = 1 # Healthy Leaf
                    
        return refined_mask

    def visualize_multi_class(
        self, 
        rgb: np.ndarray, 
        fg_map: np.ndarray,
        cat_evidence: np.ndarray, 
        pseudo_mask: np.ndarray,
        class_name: str
    ) -> np.ndarray:
        """Create a comparison visualization for multi-class."""
        import matplotlib.pyplot as plt
        from io import BytesIO
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        rgb_disp = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        axes[0].imshow(rgb_disp)
        axes[0].set_title(f"RGB ({class_name})")
        
        axes[1].imshow(fg_map, cmap='gray')
        axes[1].set_title("Foreground Evidence")
        
        axes[2].imshow(cat_evidence, cmap='jet')
        axes[2].set_title("Category Evidence")
        
        # Color-coded mask for visualization
        # 0: Black, 1: Green, Others: Red/Orange/etc.
        mask_vis = np.zeros((*pseudo_mask.shape, 3), dtype=np.uint8)
        mask_vis[pseudo_mask == 1] = [0, 255, 0] # Healthy: Green
        mask_vis[pseudo_mask > 1] = [255, 128, 0] # Category: Orange
        
        axes[3].imshow(mask_vis)
        axes[3].set_title("Multi-Class Pseudo Mask")
        
        for ax in axes:
            ax.axis('off')
            
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        plt.close()
        
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
