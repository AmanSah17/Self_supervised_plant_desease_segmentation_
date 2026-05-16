"""
Stage 3 Execution Script: Anomaly Localization with PaDiM.
Fits a healthy distribution and generates anomaly heatmaps for diseased samples.
"""
from __future__ import annotations

import sys
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import json

from lettuce_ssl_segmentation_lab.config import LabConfig
from lettuce_ssl_segmentation_lab.data.multichannel_dataset import MultiChannelLeafDataset
from lettuce_ssl_segmentation_lab.utils.feature_extractor import DINOv2FeatureExtractor
from lettuce_ssl_segmentation_lab.pipeline.anomaly_detector import PaDiMDetector
from lettuce_ssl_segmentation_lab.pipeline.orchestrator import SegmentationResearchOrchestrator

def print_section(title: str, width: int = 80):
    print(f"\n{'='*width}")
    print(f"{title.center(width)}")
    print(f"{'='*width}\n")

def main():
    print_section("STAGE 3: ANOMALY LOCALIZATION (PaDiM)", 80)
    
    config = LabConfig().resolve()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = config.lab_root / "stage3_anomaly_localization"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Setup Manifest
    print(f"[INFO] Setting up manifest...")
    orchestrator = SegmentationResearchOrchestrator(config)
    manifest_df, summary = orchestrator.build_manifest()
    
    # 2. Initialize Feature Extractor
    print(f"[INFO] Initializing DINOv2 feature extractor...")
    extractor = DINOv2FeatureExtractor(model_name="dinov2_vitb14", device=device)
    
    # 3. Fit PaDiM on Healthy Training Images
    print_section("Fitting Healthy Distribution")
    
    train_dataset = MultiChannelLeafDataset(manifest_df, config, split="train")
    healthy_indices = [i for i, row in train_dataset.manifest_df.iterrows() if row["label_kind"] == "healthy"]
    
    if not healthy_indices:
        print("[ERROR] No healthy images found in train split!")
        return 1
        
    print(f"[INFO] Found {len(healthy_indices)} healthy images for fitting.")
    healthy_subset = Subset(train_dataset, healthy_indices)
    healthy_loader = DataLoader(healthy_subset, batch_size=8, shuffle=False, num_workers=0)
    
    detector = PaDiMDetector(d_reduced=128, device=device)
    detector.fit(healthy_loader, extractor)
    
    # Save model
    model_path = output_dir / "padim_model.pkl"
    detector.save(model_path)
    
    # 4. Generate Anomaly Maps for Diseased Images
    print_section("Generating Anomaly Maps")
    
    # Get diseased images from train and val
    all_datasets = {
        "train": MultiChannelLeafDataset(manifest_df, config, split="train"),
        "validation": MultiChannelLeafDataset(manifest_df, config, split="validation")
    }
    
    # Create output subdirectories
    maps_dir = output_dir / "anomaly_maps"
    maps_dir.mkdir(exist_ok=True)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    total_processed = 0
    
    for split, dataset in all_datasets.items():
        diseased_indices = [i for i, row in dataset.manifest_df.iterrows() if row["label_kind"] == "disease"]
        if not diseased_indices:
            print(f"[INFO] No diseased images in {split} split. Skipping.")
            continue
            
        print(f"[INFO] Processing {len(diseased_indices)} diseased images in {split} split...")
        diseased_subset = Subset(dataset, diseased_indices)
        diseased_loader = DataLoader(diseased_subset, batch_size=4, shuffle=False)
        
        for batch_idx, batch in enumerate(tqdm(diseased_loader, desc=f"Inference ({split})")):
            images = batch["image"]
            stems = batch["image_stem"]
            
            # Compute anomaly maps
            anomaly_maps = detector.score(images, extractor)
            
            # Save maps and visualizations
            for i in range(len(images)):
                stem = stems[i]
                amap = anomaly_maps[i, 0].cpu().numpy()
                
                # Save raw map
                np.save(maps_dir / f"{stem}_anomaly.npy", amap)
                
                # Save visualization (top 5 from each split for now to save space/time)
                if total_processed < 20:
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    
                    # Original RGB
                    rgb = images[i, 0:3, :, :].permute(1, 2, 0).cpu().numpy()
                    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
                    axes[0].imshow(rgb)
                    axes[0].set_title("Original RGB")
                    axes[0].axis("off")
                    
                    # Anomaly Heatmap
                    im = axes[1].imshow(amap, cmap='jet')
                    plt.colorbar(im, ax=axes[1])
                    axes[1].set_title("Anomaly Heatmap (PaDiM)")
                    axes[1].axis("off")
                    
                    plt.tight_layout()
                    plt.savefig(viz_dir / f"{stem}_viz.png")
                    plt.close()
                
                total_processed += 1

    print_section("Stage 3 Completed Successfully")
    print(f"[INFO] Total anomaly maps generated: {total_processed}")
    print(f"[INFO] Outputs saved to: {output_dir}")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
