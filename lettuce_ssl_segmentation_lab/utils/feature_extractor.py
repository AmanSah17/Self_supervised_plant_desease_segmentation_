"""
Feature extraction utilities for SSL backbones.
Supports DINOv2 and MAE with proper device handling and batching.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from lettuce_ssl_segmentation_lab.config import LabConfig


class FeatureExtractor(ABC):
    """Abstract base class for feature extraction."""
    
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the backbone model."""
        raise NotImplementedError
    
    @abstractmethod
    def extract(self, image_batch: torch.Tensor) -> torch.Tensor:
        """Extract features from batch of images (B, C, H, W)."""
        raise NotImplementedError
    
    @abstractmethod
    def feature_dim(self) -> int:
        """Return feature dimension."""
        raise NotImplementedError


class DINOv2FeatureExtractor(FeatureExtractor):
    """DINOv2 feature extractor for dense visual features."""
    
    def __init__(self, model_name: str = "dinov2_vitb14", device: Optional[str] = None):
        super().__init__(device)
        self.model_name = model_name
        self._feature_dim = None
        self.load_model()
    
    def load_model(self) -> None:
        """Load DINOv2 model from torch hub."""
        print(f"Loading {self.model_name} on {self.device}...")
        try:
            self.model = torch.hub.load("facebookresearch/dinov2", self.model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"[OK] {self.model_name} loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load {self.model_name}: {str(e)}")
    
    def extract(self, image_batch: torch.Tensor) -> torch.Tensor:
        """
        Extract DINOv2 features from batch.
        
        Args:
            image_batch: Tensor of shape (B, 3, H, W) with values in [0, 1] or [0, 255]
        
        Returns:
            Features of shape (B, D) where D is feature dimension
        """
        if image_batch.shape[1] != 3:
            raise ValueError(f"Expected 3-channel RGB input, got {image_batch.shape[1]}")
        
        # Ensure float32 and on correct device
        image_batch = image_batch.float().to(self.device)
        
        # Normalize if in 0-255 range
        if image_batch.max() > 1.0:
            image_batch = image_batch / 255.0
            
        # DINOv2 ViT models usually have patch size 14
        # Input dimensions must be multiple of patch size
        patch_size = 14
        h, w = image_batch.shape[-2:]
        if h % patch_size != 0 or w % patch_size != 0:
            new_h = (h // patch_size) * patch_size
            new_w = (w // patch_size) * patch_size
            image_batch = F.interpolate(image_batch, size=(new_h, new_w), mode='bicubic', align_corners=False)
        
        with torch.no_grad():
            # DINOv2 expects specific normalization
            # ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
            image_batch = (image_batch - mean) / std
            
            # Extract features
            # DINOv2 forward_features returns a dict with keys like 'x_norm_clstoken', 'x_norm_patchtokens'
            output = self.model.forward_features(image_batch)
            
            # Handle dict output from DINOv2
            if isinstance(output, dict):
                if 'x_norm_clstoken' in output:
                    features = output['x_norm_clstoken']
                elif 'x' in output:
                    features = output['x']
                elif 'x_norm_patchtokens' in output:
                    features = output['x_norm_patchtokens']
                else:
                    # Fallback: get the first tensor value
                    features = next((v for v in output.values() if torch.is_tensor(v)), output)
            else:
                features = output
            
            # Ensure features are tensors
            if not torch.is_tensor(features):
                raise TypeError(f"Expected tensor output from model, got {type(features)}")
                
            # Ensure features are 2D (B, D)
            if features.ndim == 3:
                # (B, num_patches, D) -> (B, D) via mean pooling
                features = features.mean(dim=1)
            elif features.ndim > 2:
                # Flatten any higher dimensions
                batch_size = features.shape[0]
                features = features.view(batch_size, -1)
            
        return features
    
    def extract_patch_features(self, image_batch: torch.Tensor) -> torch.Tensor:
        """
        Extract DINOv2 patch-level features from batch.
        
        Args:
            image_batch: Tensor of shape (B, 3, H, W)
            
        Returns:
            Patch features of shape (B, D, H_p, W_p)
        """
        if image_batch.shape[1] != 3:
            raise ValueError(f"Expected 3-channel RGB input, got {image_batch.shape[1]}")
            
        # Ensure float32 and on correct device
        image_batch = image_batch.float().to(self.device)
        
        # Normalize if in 0-255 range
        if image_batch.max() > 1.0:
            image_batch = image_batch / 255.0
            
        # DINOv2 ViT models usually have patch size 14
        patch_size = 14
        h, w = image_batch.shape[-2:]
        
        # Ensure h, w are multiples of patch_size
        if h % patch_size != 0 or w % patch_size != 0:
            new_h = (h // patch_size) * patch_size
            new_w = (w // patch_size) * patch_size
            image_batch = F.interpolate(image_batch, size=(new_h, new_w), mode='bicubic', align_corners=False)
            h, w = new_h, new_w
            
        h_p, w_p = h // patch_size, w // patch_size
        
        with torch.no_grad():
            # ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
            image_batch = (image_batch - mean) / std
            
            # Extract features
            output = self.model.forward_features(image_batch)
            
            if isinstance(output, dict):
                if 'x_norm_patchtokens' in output:
                    features = output['x_norm_patchtokens']
                else:
                    # Try to find something that looks like patch tokens (B, N, D)
                    features = next((v for v in output.values() if torch.is_tensor(v) and v.ndim == 3), None)
                    if features is None:
                        raise RuntimeError(f"Could not find patch tokens in DINOv2 output: {output.keys()}")
            else:
                features = output
                
            # features is (B, N, D) or (B, 1+N, D)
            if features.ndim == 3:
                num_patches = h_p * w_p
                if features.shape[1] == num_patches + 1:
                    # Remove CLS token
                    features = features[:, 1:, :]
                elif features.shape[1] != num_patches:
                    raise ValueError(f"Unexpected number of patches: got {features.shape[1]}, expected {num_patches}")
                
                # Reshape to (B, H_p, W_p, D)
                features = features.view(-1, h_p, w_p, features.shape[-1])
                # Permute to (B, D, H_p, W_p)
                features = features.permute(0, 3, 1, 2)
            else:
                raise TypeError(f"Expected 3D tensor for patch features, got {features.ndim}D")
                
        return features

    def feature_dim(self) -> int:
        """Return DINOv2 feature dimension."""
        if self._feature_dim is None:
            # Test forward pass to determine dimension
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            with torch.no_grad():
                output = self.extract(dummy_input)
            
            # Output should be (1, D) after extraction
            if torch.is_tensor(output):
                if output.ndim == 2:
                    self._feature_dim = output.shape[1]
                else:
                    self._feature_dim = output.numel()
            else:
                raise TypeError(f"Expected tensor from extract(), got {type(output)}")
        
        return self._feature_dim


class FeatureStorage:
    """Store and retrieve extracted features efficiently."""
    
    def __init__(self, output_dir: Path, split: str = "train"):
        self.output_dir = output_dir
        self.split = split
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.features_path = self.output_dir / f"features_{split}.npy"
        self.indices_path = self.output_dir / f"indices_{split}.npy"
        self.metadata_path = self.output_dir / f"metadata_{split}.npy"
    
    def save_batch(
        self,
        features: np.ndarray,
        image_paths: list[str],
        class_names: list[str],
        label_kinds: list[str],
        offset: int = 0,
    ) -> None:
        """Save features and metadata in memory-efficient way."""
        batch_size = len(image_paths)
        
        # Save features
        if not self.features_path.exists():
            features_array = np.memmap(
                self.features_path, 
                dtype=np.float32, 
                mode='w+', 
                shape=((offset + batch_size),) + features.shape[1:]
            )
        else:
            features_array = np.memmap(
                self.features_path, 
                dtype=np.float32, 
                mode='r+', 
                shape=((offset + batch_size),) + features.shape[1:]
            )
        
        features_array[offset:offset+batch_size] = features
        features_array.flush()
        
        # Save metadata
        metadata = np.array([
            (path, cls, kind) 
            for path, cls, kind in zip(image_paths, class_names, label_kinds)
        ], dtype=object)
        
        if not self.metadata_path.exists():
            metadata_array = np.memmap(
                self.metadata_path,
                dtype=object,
                mode='w+',
                shape=(offset + batch_size,)
            )
        else:
            metadata_array = np.memmap(
                self.metadata_path,
                dtype=object,
                mode='r+',
                shape=(offset + batch_size,)
            )
        
        metadata_array[offset:offset+batch_size] = metadata
        metadata_array.flush()
    
    def load_all(self) -> tuple[np.ndarray, np.ndarray]:
        """Load all features and metadata."""
        if not self.features_path.exists():
            raise FileNotFoundError(f"Features not found at {self.features_path}")
        
        features = np.load(self.features_path)
        metadata = np.load(self.metadata_path, allow_pickle=True)
        
        return features, metadata


def extract_features_from_dataloader(
    dataloader,
    extractor: FeatureExtractor,
    output_dir: Path,
    split: str = "train",
    max_batches: Optional[int] = None,
) -> tuple[np.ndarray, dict]:
    """
    Extract features from all samples in dataloader.
    
    Args:
        dataloader: PyTorch DataLoader
        extractor: Feature extractor instance
        output_dir: Directory to save features
        split: Data split name
        max_batches: Max batches to process (for testing)
    
    Returns:
        Tuple of (features, statistics)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    storage = FeatureStorage(output_dir, split=split)
    
    all_features = []
    all_metadata = {
        "image_paths": [],
        "class_names": [],
        "label_kinds": [],
    }
    
    num_batches = len(dataloader) if max_batches is None else min(max_batches, len(dataloader))
    
    print(f"\nExtracting features from {split} split ({num_batches} batches)...")
    
    pbar = tqdm(dataloader, total=num_batches, desc=f"Feature extraction ({split})", unit="batch")
    
    for batch_idx, batch in enumerate(pbar):
        if max_batches is not None and batch_idx >= max_batches:
            break
        
        try:
            # Extract RGB from multi-channel input
            images = batch["image"]  # (B, 13, 256, 256)
            rgb_channel = images[:, 0:3, :, :]  # Take first 3 channels (RGB)
            
            # Extract features
            features = extractor.extract(rgb_channel)  # (B, D)
            
            # Features should already be 2D from extract() method
            # but ensure it's properly shaped
            if features.ndim != 2:
                batch_size = features.shape[0]
                features = features.view(batch_size, -1)
            
            # Convert to numpy
            features_np = features.cpu().numpy().astype(np.float32)
            
            # Collect metadata
            batch_paths = batch.get("image_path", [""] * len(features_np))
            batch_classes = batch.get("class_name", [""] * len(features_np))
            batch_kinds = batch.get("label_kind", [""] * len(features_np))
            
            all_features.append(features_np)
            all_metadata["image_paths"].extend(batch_paths)
            all_metadata["class_names"].extend(batch_classes)
            all_metadata["label_kinds"].extend(batch_kinds)
            
            pbar.update(1)
        
        except Exception as e:
            print(f"\n✗ Error processing batch {batch_idx}: {str(e)}")
            raise
    
    # Concatenate all features
    all_features = np.concatenate(all_features, axis=0)
    
    print(f"✓ Extracted {len(all_features)} feature vectors")
    print(f"  Shape: {all_features.shape}")
    print(f"  Dtype: {all_features.dtype}")
    
    # Save features
    features_path = output_dir / f"features_{split}.npy"
    np.save(features_path, all_features)
    print(f"✓ Features saved to {features_path}")
    
    # Save metadata
    metadata_path = output_dir / f"metadata_{split}.pkl"
    import pickle
    with open(metadata_path, 'wb') as f:
        pickle.dump(all_metadata, f)
    print(f"✓ Metadata saved to {metadata_path}")
    
    return all_features, all_metadata
