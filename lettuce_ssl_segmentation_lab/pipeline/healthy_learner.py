"""
Stage 2: Healthy-Only Representation Learning
Extracts features from healthy leaf images and builds feature statistics.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from lettuce_ssl_segmentation_lab.config import LabConfig
from lettuce_ssl_segmentation_lab.pipeline.orchestrator import SegmentationResearchOrchestrator
from lettuce_ssl_segmentation_lab.data.multichannel_dataset import MultiChannelLeafDataset
from lettuce_ssl_segmentation_lab.utils.feature_extractor import DINOv2FeatureExtractor
from lettuce_ssl_segmentation_lab.utils.checkpoint_manager import CheckpointManager
from lettuce_ssl_segmentation_lab.utils.logging_utils import ExperimentLogger


class HealthyRepresentationLearner:
    """Learn representation from healthy images using DINOv2."""
    
    def __init__(
        self,
        config: LabConfig,
        device: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ):
        self.config = config.resolve()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir or self.config.lab_root / "stage2_healthy_learning"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = ExperimentLogger(self.output_dir)
        self.checkpoint_manager = CheckpointManager(self.output_dir, "healthy_features")
        
        # Will be initialized later
        self.extractor = None
        self.manifest_df = None
        self.healthy_dataset = None
    
    def _log(self, msg: str, level: str = "info"):
        """Log message with timestamp."""
        print(f"[{level.upper()}] {msg}")
    
    def setup_manifest(self) -> pd.DataFrame:
        """Build and load manifest."""
        self._log("Building manifest...")
        orchestrator = SegmentationResearchOrchestrator(self.config)
        self.manifest_df, summary = orchestrator.build_manifest()
        
        self._log(f"Manifest built: {len(self.manifest_df)} total samples")
        self._log(f"  Healthy: {summary.healthy_samples}")
        self._log(f"  Diseased: {summary.diseased_samples}")
        
        return self.manifest_df
    
    def setup_feature_extractor(self, model_name: str = "dinov2_vitb14") -> DINOv2FeatureExtractor:
        """Initialize feature extractor."""
        self._log(f"Initializing {model_name} on {self.device}...")
        self.extractor = DINOv2FeatureExtractor(model_name=model_name, device=self.device)
        self._log(f"Feature dimension: {self.extractor.feature_dim()}")
        return self.extractor
    
    def get_healthy_dataset(self, split: str = "train") -> Subset:
        """Get dataset filtered to only healthy images."""
        self._log(f"Filtering {split} split to healthy images...")
        
        dataset = MultiChannelLeafDataset(self.manifest_df, self.config, split=split)
        
        # Filter to healthy indices
        healthy_indices = []
        for idx in range(len(dataset)):
            sample = dataset.manifest_df.iloc[idx]
            if sample["label_kind"] == "healthy":
                healthy_indices.append(idx)
        
        self._log(f"Found {len(healthy_indices)} healthy images in {split} split")
        
        if len(healthy_indices) == 0:
            self._log(f"[WARN] No healthy images in {split} split!", level="warning")
            return None
        
        self.healthy_dataset = Subset(dataset, healthy_indices)
        return self.healthy_dataset
    
    def extract_healthy_features(
        self,
        split: str = "train",
        batch_size: int = 4,
        num_workers: int = 0,
        max_batches: Optional[int] = None,
    ) -> tuple[np.ndarray, dict]:
        """Extract features from healthy images."""
        if self.healthy_dataset is None:
            self.get_healthy_dataset(split=split)
        
        if self.healthy_dataset is None:
            self._log(f"Cannot extract features: no healthy dataset", level="error")
            return None, None
        
        if self.extractor is None:
            self.setup_feature_extractor()
        
        # Create dataloader
        dataloader = DataLoader(
            self.healthy_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        
        self._log(f"\nExtracting features from {len(self.healthy_dataset)} healthy images...")
        self._log(f"Batch size: {batch_size}, Total batches: {len(dataloader)}")
        
        all_features = []
        all_metadata = {
            "image_paths": [],
            "class_names": [],
            "image_stems": [],
        }
        
        num_batches = len(dataloader) if max_batches is None else min(max_batches, len(dataloader))
        pbar = tqdm(dataloader, total=num_batches, desc="Feature extraction", unit="batch", ncols=100)
        
        total_samples = 0
        
        for batch_idx, batch in enumerate(pbar):
            if max_batches is not None and batch_idx >= max_batches:
                break
            
            try:
                # Extract RGB channels
                images = batch["image"]  # (B, 13, 256, 256)
                rgb = images[:, 0:3, :, :]  # (B, 3, 256, 256)
                
                # Extract features
                with torch.no_grad():
                    features = self.extractor.extract(rgb)  # (B, D)
                
                # Features should already be 2D from extract() method
                if features.ndim != 2:
                    batch_size_actual = features.shape[0]
                    features = features.view(batch_size_actual, -1)
                
                # Convert to numpy
                features_np = features.cpu().numpy().astype(np.float32)
                all_features.append(features_np)
                
                # Collect metadata
                paths = batch.get("image_path", [""] * len(features_np))
                classes = batch.get("class_name", [""] * len(features_np))
                stems = batch.get("image_stem", [""] * len(features_np))
                
                all_metadata["image_paths"].extend(paths)
                all_metadata["class_names"].extend(classes)
                all_metadata["image_stems"].extend(stems)
                
                total_samples += len(features_np)
                pbar.set_postfix({"total": total_samples})
                
                # Checkpoint every 10 batches
                if (batch_idx + 1) % 10 == 0:
                    checkpoint_data = {
                        "features": np.concatenate(all_features),
                        "metadata": all_metadata,
                    }
                    self.checkpoint_manager.save_checkpoint(
                        checkpoint_data,
                        epoch=0,
                        batch_idx=batch_idx,
                        total_samples=total_samples,
                        status="in_progress",
                    )
            
            except Exception as e:
                self._log(f"Error in batch {batch_idx}: {str(e)}", level="error")
                raise
        
        # Concatenate all features
        all_features = np.concatenate(all_features, axis=0)
        
        self._log(f"\n[OK] Extracted {len(all_features)} feature vectors")
        self._log(f"  Shape: {all_features.shape}")
        self._log(f"  Dtype: {all_features.dtype}")
        
        return all_features, all_metadata
    
    def compute_healthy_statistics(
        self,
        features: np.ndarray,
        metadata: dict,
    ) -> dict:
        """Compute statistics from healthy features."""
        self._log("\nComputing healthy feature statistics...")
        
        stats = {
            "num_samples": len(features),
            "feature_dim": features.shape[1],
            "mean": np.mean(features, axis=0),
            "std": np.std(features, axis=0),
            "min": np.min(features, axis=0),
            "max": np.max(features, axis=0),
            "median": np.median(features, axis=0),
            "percentile_25": np.percentile(features, 25, axis=0),
            "percentile_75": np.percentile(features, 75, axis=0),
        }
        
        # Per-class statistics
        class_names = metadata["class_names"]
        unique_classes = set(class_names)
        
        stats["per_class"] = {}
        for cls in sorted(unique_classes):
            cls_mask = np.array([c == cls for c in class_names])
            cls_features = features[cls_mask]
            
            stats["per_class"][cls] = {
                "num_samples": int(cls_mask.sum()),
                "mean": np.mean(cls_features, axis=0),
                "std": np.std(cls_features, axis=0),
            }
            
            self._log(f"  {cls}: {int(cls_mask.sum())} samples")
        
        # Feature statistics
        self._log(f"\nFeature Statistics:")
        self._log(f"  Num samples: {stats['num_samples']}")
        self._log(f"  Feature dim: {stats['feature_dim']}")
        self._log(f"  Mean range: [{stats['mean'].min():.6f}, {stats['mean'].max():.6f}]")
        self._log(f"  Std range: [{stats['std'].min():.6f}, {stats['std'].max():.6f}]")
        
        return stats
    
    def save_healthy_bank(
        self,
        features: np.ndarray,
        stats: dict,
        metadata: dict,
        split: str = "train",
    ) -> Path:
        """Save healthy feature bank and statistics."""
        self._log(f"\nSaving healthy feature bank...")
        
        # Save features
        features_path = self.output_dir / f"healthy_features_{split}.npy"
        np.save(features_path, features)
        self._log(f"[OK] Features saved: {features_path}")
        
        # Save statistics
        stats_path = self.output_dir / f"healthy_stats_{split}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        stats_serializable = {}
        for k, v in stats.items():
            if isinstance(v, dict):
                stats_serializable[k] = {}
                for k2, v2 in v.items():
                    if isinstance(v2, dict):
                        stats_serializable[k][k2] = {
                            k3: v3.tolist() if isinstance(v3, np.ndarray) else v3
                            for k3, v3 in v2.items()
                        }
                    else:
                        stats_serializable[k][k2] = v2
            elif isinstance(v, np.ndarray):
                stats_serializable[k] = v.tolist()
            else:
                stats_serializable[k] = v
        
        with open(stats_path, 'w') as f:
            json.dump(stats_serializable, f, indent=2)
        self._log(f"[OK] Statistics saved: {stats_path}")
        
        # Save metadata
        metadata_path = self.output_dir / f"healthy_metadata_{split}.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        self._log(f"[OK] Metadata saved: {metadata_path}")
        
        # Save summary
        summary = {
            "split": split,
            "num_samples": stats["num_samples"],
            "feature_dim": stats["feature_dim"],
            "per_class_counts": {
                cls: stats["per_class"][cls]["num_samples"]
                for cls in stats["per_class"]
            },
            "files": {
                "features": str(features_path),
                "stats": str(stats_path),
                "metadata": str(metadata_path),
            }
        }
        
        summary_path = self.output_dir / f"healthy_bank_summary_{split}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        self._log(f"[OK] Summary saved: {summary_path}")
        
        return features_path
    
    def run_pipeline(
        self,
        split: str = "train",
        batch_size: int = 4,
        num_workers: int = 0,
        model_name: str = "dinov2_vitb14",
    ) -> dict:
        """Run complete healthy learning pipeline."""
        self._log("="*80)
        self._log("STAGE 2: HEALTHY-ONLY REPRESENTATION LEARNING")
        self._log("="*80)
        
        try:
            # Step 1: Manifest
            self.setup_manifest()
            
            # Step 2: Feature extractor
            self.setup_feature_extractor(model_name=model_name)
            
            # Step 3: Get healthy dataset
            self.get_healthy_dataset(split=split)
            
            # Step 4: Extract features
            features, metadata = self.extract_healthy_features(
                split=split,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            
            if features is None:
                self._log("Pipeline failed: no features extracted", level="error")
                return {"status": "failed"}
            
            # Step 5: Compute statistics
            stats = self.compute_healthy_statistics(features, metadata)
            
            # Step 6: Save everything
            self.save_healthy_bank(features, stats, metadata, split=split)
            
            # Save final checkpoint
            self.checkpoint_manager.save_checkpoint(
                {
                    "features": features,
                    "stats": stats,
                    "metadata": metadata,
                },
                epoch=0,
                batch_idx=-1,
                total_samples=len(features),
                status="completed",
                metrics={"num_samples": len(features)},
            )
            
            self._log("\n" + "="*80)
            self._log("[OK] STAGE 2 COMPLETED SUCCESSFULLY")
            self._log("="*80)
            
            return {
                "status": "completed",
                "num_samples": len(features),
                "feature_dim": features.shape[1],
                "feature_shape": features.shape,
                "stats": stats,
            }
        
        except Exception as e:
            self._log(f"\n[ERROR] Pipeline failed: {str(e)}", level="error")
            import traceback
            traceback.print_exc()
            
            self.checkpoint_manager.save_checkpoint(
                {},
                epoch=0,
                batch_idx=-1,
                total_samples=0,
                status="failed",
                error_msg=str(e),
            )
            
            return {"status": "failed", "error": str(e)}
