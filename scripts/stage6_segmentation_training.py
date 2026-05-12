"""
Stage 6: Multi-Task Segmentation Training.
Uses pseudo-masks from Stage 4 to train a supervised segmenter (SegFormer or DeepLabV3+).
Features MLflow tracking for all metrics.
"""
from __future__ import annotations

import sys
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
import numpy as np

from lettuce_ssl_segmentation_lab.config import LabConfig
from lettuce_ssl_segmentation_lab.data.multichannel_dataset import MultiChannelLeafDataset
from lettuce_ssl_segmentation_lab.pipeline.orchestrator import SegmentationResearchOrchestrator
from lettuce_ssl_segmentation_lab.pipeline.models import SegmentationModelFactory
from lettuce_ssl_segmentation_lab.pipeline.losses import SegmentationLoss
from lettuce_ssl_segmentation_lab.pipeline.segmentation_trainer import SegmentationTrainer

def get_transforms(img_size: tuple[int, int]):
    # Note: Spatial transforms (flips) are handled manually in the dataset
    # This Compose should only contain pixel-level transforms (color, noise, etc.)
    train_transform = T.Compose([
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ])
    
    val_transform = None 
    
    return train_transform, val_transform

def main():
    print("\n" + "="*80)
    print("STAGE 6: MULTI-TASK SEGMENTATION TRAINING".center(80))
    print("="*80 + "\n")
    
    config = LabConfig().resolve()
    # config.selected_channels = ["rgb"] # Commented out to allow full stack
    
    is_smoke_test = os.environ.get("SMOKE_TEST", "false").lower() == "true"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = config.lab_root / "stage6_segmentation_training"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    orchestrator = SegmentationResearchOrchestrator(config)
    manifest_df, _ = orchestrator.build_manifest()
    
    # Filter manifest to only include samples that have Stage 5 masks
    print("[INFO] Filtering manifest for images with valid Stage 5 masks...")
    mask_dir = config.lab_root / "stage5_cam_attention_masks" / "masks"
    if not mask_dir.exists():
        print(f"[ERROR] Mask directory not found: {mask_dir}")
        return 1
        
    def has_mask(row):
        return (mask_dir / f"{row['image_stem']}_mask.png").exists()
        
    manifest_df = manifest_df[manifest_df.apply(has_mask, axis=1)].reset_index(drop=True)
    print(f"[INFO] Found {len(manifest_df)} images with valid masks.")
    
    if len(manifest_df) == 0:
        print("[ERROR] No masks found. Please run Stage 5 first.")
        return 1
        
    # Data Setup
    train_transform, val_transform = get_transforms(config.img_size)
    
    train_dataset = MultiChannelLeafDataset(manifest_df, config, split="train", transform=train_transform)
    val_dataset = MultiChannelLeafDataset(manifest_df, config, split="val", transform=val_transform)
    
    if is_smoke_test:
        from torch.utils.data import Subset
        train_count = min(100, len(train_dataset))
        val_count = min(50, len(val_dataset))
        train_dataset = Subset(train_dataset, range(train_count))
        val_dataset = Subset(val_dataset, range(val_count))
        print(f"[INFO] Smoke test: dataset limited to {len(train_dataset)} train, {len(val_dataset)} val samples.")
    
    batch_size = int(os.environ.get("BATCH_SIZE", "8"))
    num_workers = int(os.environ.get("NUM_WORKERS", "0"))
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    # 3. Model Setup
    # Mapping: 0: BG, 1: HLTY, 2-8: Diseases/Weeds -> Total 9 classes
    num_classes = 9
    model_name = os.environ.get("SEG_MODEL", "segformer-b3")
    if is_smoke_test:
        model_name = "segformer-b0" # Use smallest for smoke test
        
    # Correctly determine input channels by inspecting a sample
    sample_batch = train_dataset[0]
    input_channels = sample_batch["image"].shape[0]
    print(f"[INFO] Detected {input_channels} input channels from dataset.")
    
    model = SegmentationModelFactory.get_model(
        model_name, 
        num_classes=num_classes,
        pretrained=True,
        input_channels=input_channels
    )
    
    # Optimization
    criterion = SegmentationLoss(ce_weight=1.0, dice_weight=1.0, ignore_index=255)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    # Trainer
    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_classes=num_classes,
        experiment_name="Lettuce_Disease_Segmentation_v1",
        output_dir=output_dir
    )
    
    # Start Training
    num_epochs = int(os.environ.get("NUM_EPOCHS", "100"))
    patience = int(os.environ.get("PATIENCE", "15"))
    
    if os.environ.get("SMOKE_TEST", "false").lower() == "true":
        num_epochs = 1
        print("[INFO] Smoke test enabled: Running 1 epoch.")
        
    trainer.fit(num_epochs=num_epochs, patience=patience)
    
    print("\n" + "="*80)
    print("Stage 6 Completed Successfully".center(80))
    print("="*80 + "\n")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
