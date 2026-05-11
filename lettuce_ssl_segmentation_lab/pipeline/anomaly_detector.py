"""
PaDiM (Patch Distribution Modeling) implementation for Anomaly Localization.
Models healthy patch features as multivariate Gaussian distributions.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Dict
from tqdm import tqdm
import pickle

class PaDiMDetector:
    """
    PaDiM: Patch Distribution Modeling for Anomaly Detection.
    Learns a multivariate Gaussian distribution for each patch position.
    """
    
    def __init__(
        self, 
        d_reduced: int = 128, 
        device: str = "cpu",
        regularization: float = 0.01
    ):
        self.d_reduced = d_reduced
        self.device = device
        self.regularization = regularization
        
        self.idx: Optional[torch.Tensor] = None
        self.mean: Optional[torch.Tensor] = None
        self.inv_covariance: Optional[torch.Tensor] = None
        self.feature_dim: Optional[int] = None
        self.h_p: Optional[int] = None
        self.w_p: Optional[int] = None

    def _select_random_dimensions(self, d_original: int):
        """Select random indices for dimensionality reduction."""
        if self.idx is None:
            torch.manual_seed(42)
            self.idx = torch.randperm(d_original)[:self.d_reduced].to(self.device)
            self.feature_dim = d_original

    def fit(self, dataloader, extractor, max_batches: Optional[int] = None):
        """
        Fit PaDiM on healthy images.
        
        Args:
            dataloader: DataLoader providing healthy images
            extractor: Feature extractor with extract_patch_features method
            max_batches: Limit batches for testing
        """
        extractor.model.to(self.device)
        extractor.model.eval()
        
        all_patch_features = []
        
        num_batches = len(dataloader) if max_batches is None else min(max_batches, len(dataloader))
        pbar = tqdm(dataloader, total=num_batches, desc="Fitting PaDiM (Extracting patches)")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                
                images = batch["image"].to(self.device)
                if images.shape[1] > 3:
                    images = images[:, 0:3, :, :]
                    
                # (B, D, H_p, W_p)
                features = extractor.extract_patch_features(images)
                
                # Dimensionality reduction via random selection
                if self.idx is None:
                    self._select_random_dimensions(features.shape[1])
                    self.h_p, self.w_p = features.shape[2], features.shape[3]
                
                features = torch.index_select(features, 1, self.idx)
                all_patch_features.append(features.cpu())
                
        # (N, D_red, H_p, W_p)
        all_patch_features = torch.cat(all_patch_features, dim=0)
        N, D, H, W = all_patch_features.shape
        
        print(f"Computing statistics for {N} healthy samples at {H}x{W} resolution...")
        
        # Reshape to (N, H*W, D)
        features_flat = all_patch_features.permute(0, 2, 3, 1).reshape(N, H * W, D)
        
        # Compute mean per patch position (H*W, D)
        self.mean = torch.mean(features_flat, dim=0).to(self.device)
        
        # Compute covariance per patch position (H*W, D, D)
        # We compute this position by position to save memory if needed, 
        # but with H*W=324 and D=128, (324, 128, 128) is only ~21MB.
        self.inv_covariance = torch.zeros((H * W, D, D)).to(self.device)
        
        # Add regularization to diagonal
        identity = torch.eye(D).to(self.device)
        
        for i in range(H * W):
            # (N, D)
            pos_features = features_flat[:, i, :].to(self.device)
            # Center features
            centered = pos_features - self.mean[i]
            # (D, D)
            cov = (centered.T @ centered) / (N - 1)
            # Regularize
            cov += self.regularization * identity
            # Invert
            self.inv_covariance[i] = torch.inverse(cov)
            
        print("[INFO] PaDiM model fitted successfully")

    def score(self, image_batch: torch.Tensor, extractor) -> torch.Tensor:
        """
        Compute anomaly scores for a batch of images.
        
        Args:
            image_batch: (B, 3, H, W)
            extractor: Feature extractor
            
        Returns:
            Anomaly heatmaps of shape (B, 1, H, W)
        """
        if self.mean is None or self.inv_covariance is None:
            raise RuntimeError("Model must be fitted before scoring")
            
        image_batch = image_batch.to(self.device)
        if image_batch.shape[1] > 3:
            image_batch = image_batch[:, 0:3, :, :]
            
        with torch.no_grad():
            # (B, D_orig, H_p, W_p)
            features = extractor.extract_patch_features(image_batch)
            B, _, H_p, W_p = features.shape
            
            # Dimensionality reduction
            features = torch.index_select(features, 1, self.idx)
            # (B, H_p*W_p, D_red)
            features_flat = features.permute(0, 2, 3, 1).reshape(B, H_p * W_p, -1)
            
            # Compute Mahalanobis distance
            # d = sqrt( (x-mu)^T @ Sigma^-1 @ (x-mu) )
            delta = features_flat - self.mean.unsqueeze(0) # (B, H*W, D)
            
            # We want (B, H*W) distances
            # distances = sum_j sum_k delta_ij * inv_cov_ijk * delta_ik
            # Using einsum for efficiency:
            # b: batch, p: patch position, i,j: feature dims
            distances = torch.einsum('bpi,pij,bpj->bp', delta, self.inv_covariance, delta)
            distances = torch.sqrt(torch.clamp(distances, min=0))
            
            # Reshape to (B, 1, H_p, W_p)
            anomaly_map = distances.reshape(B, 1, H_p, W_p)
            
            # Upsample to original image resolution
            h_orig, w_orig = image_batch.shape[2], image_batch.shape[3]
            anomaly_map = F.interpolate(
                anomaly_map, 
                size=(h_orig, w_orig), 
                mode='bilinear', 
                align_corners=False
            )
            
        return anomaly_map

    def save(self, path: Union[str, Path]):
        """Save model to disk."""
        data = {
            "d_reduced": self.d_reduced,
            "idx": self.idx.cpu() if self.idx is not None else None,
            "mean": self.mean.cpu() if self.mean is not None else None,
            "inv_covariance": self.inv_covariance.cpu() if self.inv_covariance is not None else None,
            "feature_dim": self.feature_dim,
            "h_p": self.h_p,
            "w_p": self.w_p,
            "regularization": self.regularization
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"[INFO] PaDiM model saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path], device: str = "cpu") -> PaDiMDetector:
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            
        instance = cls(
            d_reduced=data["d_reduced"], 
            device=device,
            regularization=data["regularization"]
        )
        instance.idx = data["idx"].to(device) if data["idx"] is not None else None
        instance.mean = data["mean"].to(device) if data["mean"] is not None else None
        instance.inv_covariance = data["inv_covariance"].to(device) if data["inv_covariance"] is not None else None
        instance.feature_dim = data["feature_dim"]
        instance.h_p = data["h_p"]
        instance.w_p = data["w_p"]
        
        return instance
