import os
import sys
from pathlib import Path

# Add project root to PYTHONPATH
root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

from lettuce_ssl_segmentation_lab.config import LabConfig
from lettuce_ssl_segmentation_lab.data.multichannel_dataset import MultiChannelLeafDataset
from lettuce_ssl_segmentation_lab.pipeline.models import SegmentationModelFactory
from lettuce_ssl_segmentation_lab.pipeline.segmentation_trainer import SegmentationTrainer
from lettuce_ssl_segmentation_lab.pipeline.metrics import SegmentationMetrics

def print_section(title: str, width: int = 80):
    print(f"\n{'='*width}")
    print(f"{title:^{width}}")
    print(f"{'='*width}\n")

class ManualLabelDataset(MultiChannelLeafDataset):
    """
    Overridden dataset to prioritize manual masks.
    """
    def __init__(self, manifest_df, config, manual_mask_dir):
        # We don't filter by split="validation" here because manual labels might be from any split
        # but the user said they are for validation. I'll just check all images in manifest.
        super().__init__(manifest_df, config, split="validation") 
        self.manual_mask_dir = Path(manual_mask_dir)
        
        # Filter to only images that have manual masks
        valid_indices = []
        for i in range(len(self.manifest_df)):
            stem = self.manifest_df.iloc[i]['image_stem']
            mask_path = self.manual_mask_dir / f"{stem}_mask.png"
            if mask_path.exists():
                valid_indices.append(i)
        
        self.manifest_df = self.manifest_df.iloc[valid_indices].reset_index(drop=True)
        print(f"[INFO] ManualLabelDataset: Found {len(self)} images with manual masks.")

    def __getitem__(self, index: int):
        res = super().__getitem__(index)
        stem = res['image_stem']
        manual_mask = self._get_manual_mask(stem)
        res['mask'] = manual_mask
        return res

    def _get_manual_mask(self, stem):
        mask_path = self.manual_mask_dir / f"{stem}_mask.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        # Resize to match model input (Config is (256, 256))
        mask = cv2.resize(mask, (self.config.img_size[1], self.config.img_size[0]), interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(mask.astype(np.int64))

def main():
    config = LabConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print_section("STAGE 8: SUPERVISED FINE-TUNING")
    
    # 1. Load Data
    manifest_path = Path(config.logs_dir) / "manifests" / "multirepresentation_manifest.csv"
    manifest_df = pd.read_csv(manifest_path)
    
    manual_mask_dir = Path(r"d:\gemma4\segmentation_lattuce-desease\lettuce_ssl_segmentation_lab\manual_validation_masks")
    
    dataset = ManualLabelDataset(manifest_df, config, manual_mask_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 2. Load Base Model (Epoch 15)
    stage6_dir = config.lab_root / "stage6_segmentation_training"
    model_path = stage6_dir / "epoch_15.pth"
    if not model_path.exists():
        print(f"[ERROR] Base model not found at {model_path}")
        return

    print(f"[INFO] Loading model from {model_path}...")
    # Initialize model first
    num_classes = len(config.class_names) + 1
    model = SegmentationModelFactory.get_model(
        name="segformer",
        num_classes=num_classes,
        input_channels=14, 
    )
    
    state_dict = torch.load(model_path, map_location=device)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # 3. Baseline Evaluation (Before Fine-tuning)
    print("[INFO] Computing baseline metrics on manual labels...")
    metrics = SegmentationMetrics(num_classes=num_classes)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Baseline Eval"):
            images = batch["image"].to(device)
            targets = batch["mask"].to(device)
            outputs = model(images).logits
            outputs = torch.nn.functional.interpolate(outputs, size=targets.shape[-2:], mode='bilinear', align_corners=False)
            preds = torch.argmax(outputs, dim=1)
            metrics.update(preds, targets)
            
    baseline_results = metrics.compute()
    print("\n--- Baseline Metrics (Prior to SFT) ---")
    for k, v in baseline_results.items():
        if isinstance(v, (int, float)):
            print(f"{k}: {v:.4f}")
        elif isinstance(v, (list, np.ndarray, torch.Tensor)):
            pass
    
    # 4. Fine-tuning
    print_section("STARTING FINE-TUNING (10 Epochs)")
    
    lr = 5e-6
    num_epochs = 10
    output_dir = stage6_dir / "supervised_finetune"
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    trainer = SegmentationTrainer(
        model=model,
        train_loader=dataloader,
        val_loader=dataloader,
        optimizer=optimizer,
        criterion=criterion,
        device=str(device),
        num_classes=num_classes,
        experiment_name="Lettuce_SFT",
        output_dir=output_dir
    )
    
    trainer.fit(num_epochs=num_epochs, patience=5)
    
    # 5. Final Evaluation
    print_section("FINAL EVALUATION")
    model.eval()
    metrics.reset()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Final Eval"):
            images = batch["image"].to(device)
            targets = batch["mask"].to(device)
            outputs = model(images).logits
            outputs = torch.nn.functional.interpolate(outputs, size=targets.shape[-2:], mode='bilinear', align_corners=False)
            preds = torch.argmax(outputs, dim=1)
            metrics.update(preds, targets)
            
    final_results = metrics.compute()
    print("\n--- Final Metrics (Post-SFT) ---")
    for k, v in final_results.items():
        if isinstance(v, (int, float)):
            print(f"{k}: {v:.4f}")
        elif isinstance(v, (list, np.ndarray, torch.Tensor)):
            pass
        
    print_section("SFT COMPLETE")
    print(f"Baseline mIoU: {baseline_results['miou']:.4f}")
    print(f"Final mIoU:    {final_results['miou']:.4f}")
    print(f"Improvement:   {final_results['miou'] - baseline_results['miou']:.4f}")

if __name__ == "__main__":
    main()
