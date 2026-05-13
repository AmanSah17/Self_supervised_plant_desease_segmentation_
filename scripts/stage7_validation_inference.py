"""
Stage 7: Validation Inference and Analytics.
Generates segmentation masks, class probabilities, prediction scores, and integrates PaDiM anomaly scores.
"""
from __future__ import annotations

import sys
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from datetime import datetime

from lettuce_ssl_segmentation_lab.config import LabConfig
from lettuce_ssl_segmentation_lab.data.multichannel_dataset import MultiChannelLeafDataset
from lettuce_ssl_segmentation_lab.pipeline.models import SegmentationModelFactory
from lettuce_ssl_segmentation_lab.pipeline.anomaly_detector import PaDiMDetector
from lettuce_ssl_segmentation_lab.utils.feature_extractor import DINOv2FeatureExtractor
from lettuce_ssl_segmentation_lab.pipeline.orchestrator import SegmentationResearchOrchestrator

def colorize_mask(mask, num_classes=9):
    """Apply a colormap to a categorical mask."""
    # Create a distinct color palette
    colors = plt.cm.get_cmap('tab10', num_classes).colors
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.float32)
    for i in range(num_classes):
        colored[mask == i] = colors[i][:3]
    return colored

def main():
    print("\n" + "="*80)
    print("STAGE 7: VALIDATION INFERENCE AND ANALYTICS".center(80))
    print("="*80 + "\n")
    
    config = LabConfig().resolve()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Output Directory
    output_dir = config.lab_root / "stage7_validation_inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # 2. Setup Dataset
    print("[INFO] Building manifest and loading validation dataset...")
    orchestrator = SegmentationResearchOrchestrator(config)
    manifest_df, _ = orchestrator.build_manifest()
    
    # Use 'validation' split as found in manifest
    val_dataset = MultiChannelLeafDataset(manifest_df, config, split="validation")
    if len(val_dataset) == 0:
        print("[WARNING] 'validation' split empty, trying 'val'...")
        val_dataset = MultiChannelLeafDataset(manifest_df, config, split="val")
        
    print(f"[INFO] Validation dataset size: {len(val_dataset)}")
    
    loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    # 3. Model Loading
    # A. Segmentation Model
    stage6_dir = config.lab_root / "stage6_segmentation_training"
    model_path = stage6_dir / "best_model.pth"
    if not model_path.exists():
        print(f"[WARNING] best_model.pth not found. Looking for latest epoch checkpoint...")
        checkpoints = list(stage6_dir.glob("epoch_*.pth"))
        if checkpoints:
            # Sort by epoch number
            checkpoints.sort(key=lambda x: int(x.stem.split("_")[1]), reverse=True)
            model_path = checkpoints[0]
            print(f"[INFO] Using latest checkpoint: {model_path}")
        else:
            print(f"[ERROR] No checkpoints found in {stage6_dir}")
            return 1
    
    # Detect input channels (should match Stage 6)
    sample_batch = val_dataset[0]
    input_channels = sample_batch["image"].shape[0]
    num_classes = 9
    
    print(f"[INFO] Loading SegFormer (b3) from {model_path}...")
    model = SegmentationModelFactory.get_model("segformer-b3", num_classes=num_classes, input_channels=input_channels)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    
    # B. PaDiM Anomaly Detector
    stage3_dir = config.lab_root / "stage3_anomaly_localization"
    padim_path = stage3_dir / "padim_model.pkl"
    detector = None
    extractor = None
    
    if padim_path.exists():
        print(f"[INFO] Loading PaDiM model from {padim_path}...")
        detector = PaDiMDetector.load(padim_path, device=device)
        extractor = DINOv2FeatureExtractor(model_name="dinov2_vitb14", device=device)
    else:
        print("[WARNING] PaDiM model not found. Anomaly scores will be 0.")
        
    # Mapping for display names
    class_names = ["BG", "HLTY", "BACT", "DML", "PML", "SBL", "SPW", "VIRL", "WLBL"]
    
    # 4. Inference Loop
    results = []
    print("[INFO] Starting inference...")
    
    total_images = len(val_dataset)
    # limit visualizations to save time/space if needed, but user said "utilize the entire validation dataset"
    # for saving masks with original images.
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Validation Inference")):
            images = batch["image"].to(device)
            stems = batch["image_stem"]
            true_classes = batch["class_name"]
            
            # A. Segmentation Output
            outputs = model(images)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            # Upsample logits to match input image size
            if logits.shape[-2:] != images.shape[-2:]:
                logits = F.interpolate(logits, size=images.shape[-2:], mode='bilinear', align_corners=False)
                
            probs = F.softmax(logits, dim=1)
            pred_masks = torch.argmax(probs, dim=1)
            conf_scores = torch.max(probs, dim=1).values
            
            # B. Anomaly Scores
            anomaly_maps = None
            if detector and extractor:
                # PaDiM expects 3-channel RGB for feature extraction
                rgb_images = images[:, 0:3, :, :]
                anomaly_maps = detector.score(rgb_images, extractor)
            
            # C. Batch Visualization (4 images at once)
            fig, axes = plt.subplots(len(stems), 4, figsize=(24, 6 * len(stems)))
            if len(stems) == 1:
                axes = axes[None, :] # Ensure 2D for consistency
                
            for i in range(len(stems)):
                stem = stems[i]
                pred_mask_np = pred_masks[i].cpu().numpy()
                prob_map_np = probs[i].cpu().numpy()
                conf_score_np = conf_scores[i].cpu().numpy()
                
                # Anomaly Calculation
                mean_img_anomaly = 0.0
                if anomaly_maps is not None:
                    amap = anomaly_maps[i, 0].cpu().numpy()
                    mean_img_anomaly = float(amap.mean())
                    disease_mask = (pred_mask_np >= 2)
                    mean_disease_anomaly = float(amap[disease_mask].mean()) if disease_mask.any() else 0.0
                else:
                    amap = np.zeros_like(pred_mask_np, dtype=np.float32)
                    mean_disease_anomaly = 0.0

                # Top predicted class
                disease_pixels = pred_mask_np[pred_mask_np >= 2]
                if len(disease_pixels) > 0:
                    dominant_disease_id = int(np.bincount(disease_pixels).argmax())
                else:
                    dominant_disease_id = 1 if (pred_mask_np == 1).any() else 0
                
                results.append({
                    "image_stem": stem,
                    "true_class": true_classes[i],
                    "pred_class": class_names[dominant_disease_id],
                    "mean_confidence": float(conf_score_np.mean()),
                    "max_confidence": float(conf_score_np.max()),
                    "mean_anomaly_score": mean_img_anomaly,
                    "mean_disease_anomaly": mean_disease_anomaly,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Plot Row i
                # 1. Original RGB
                rgb = images[i, 0:3, :, :].permute(1, 2, 0).cpu().numpy()
                rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
                axes[i, 0].imshow(rgb)
                axes[i, 0].set_title(f"Original RGB\n({stem} | True: {true_classes[i]})")
                axes[i, 0].axis("off")
                
                # 2. Predicted Mask
                colored_mask = colorize_mask(pred_mask_np, num_classes=num_classes)
                axes[i, 1].imshow(colored_mask)
                axes[i, 1].set_title(f"Predicted Mask\n(Pred: {class_names[dominant_disease_id]})")
                axes[i, 1].axis("off")
                
                # 3. Class Probability Heatmap
                prob_to_show = prob_map_np[dominant_disease_id]
                im2 = axes[i, 2].imshow(prob_to_show, cmap='inferno')
                plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)
                axes[i, 2].set_title(f"Prob: {class_names[dominant_disease_id]}")
                axes[i, 2].axis("off")
                
                # 4. Anomaly Map
                im3 = axes[i, 3].imshow(amap, cmap='jet')
                plt.colorbar(im3, ax=axes[i, 3], fraction=0.046, pad=0.04)
                axes[i, 3].set_title(f"Anomaly Map\n(Score: {mean_disease_anomaly:.4f})")
                axes[i, 3].axis("off")
                
            plt.tight_layout()
            plt.savefig(viz_dir / f"batch_{batch_idx:03d}_validation_results.png", dpi=720)
            plt.close(fig)
                
    # 6. Save Summary
    results_df = pd.DataFrame(results)
    csv_path = output_dir / "validation_inference_results.csv"
    results_df.to_csv(csv_path, index=False)
    
    print("\n" + "="*80)
    print(f"Validation Inference Completed: {len(results)} images processed".center(80))
    print(f"Results saved to: {output_dir}".center(80))
    print("="*80 + "\n")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
