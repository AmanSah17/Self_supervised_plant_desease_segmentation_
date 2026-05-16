import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import pandas as pd
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(r'd:\gemma4\segmentation_lattuce-desease')
sys.path.append(str(project_root))

from lettuce_ssl_segmentation_lab.data.multichannel_dataset import MultiChannelLeafDataset
from lettuce_ssl_segmentation_lab.config import LabConfig
from lettuce_ssl_segmentation_lab.pipeline.models import SegmentationModelFactory
from lettuce_ssl_segmentation_lab.pipeline.metrics import SegmentationMetrics

def main():
    config = LabConfig().resolve()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = project_root / 'Report' / 'stage78_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    manual_mask_dir = project_root / 'lettuce_ssl_segmentation_lab' / 'manual_validation_masks'
    stage6_dir = config.lab_root / "stage6_segmentation_training"
    sft_dir = stage6_dir / "supervised_finetune"
    
    # 1. Load Models
    num_classes = 9
    input_channels = 14
    
    print("[INFO] Loading SSL model (Epoch 15)...")
    ssl_model = SegmentationModelFactory.get_model("segformer-b3", num_classes=num_classes, input_channels=input_channels)
    ssl_ckpt = torch.load(stage6_dir / "epoch_15.pth", map_location=device)
    ssl_model.load_state_dict(ssl_ckpt['model_state_dict'] if 'model_state_dict' in ssl_ckpt else ssl_ckpt)
    ssl_model.to(device).eval()
    
    print("[INFO] Loading SFT model (Best)...")
    sft_model = SegmentationModelFactory.get_model("segformer-b3", num_classes=num_classes, input_channels=input_channels)
    sft_path = sft_dir / "best_model.pth"
    if sft_path.exists():
        sft_ckpt = torch.load(sft_path, map_location=device)
        sft_model.load_state_dict(sft_ckpt['model_state_dict'] if 'model_state_dict' in sft_ckpt else sft_ckpt)
        sft_model.to(device).eval()
    else:
        print("[WARNING] SFT best_model not found. Using Epoch 10 if available.")
        sft_path = sft_dir / "epoch_10.pth"
        if sft_path.exists():
            sft_ckpt = torch.load(sft_path, map_location=device)
            sft_model.load_state_dict(sft_ckpt['model_state_dict'] if 'model_state_dict' in sft_ckpt else sft_ckpt)
            sft_model.to(device).eval()

    # 2. Setup Dataset
    manifest_path = config.logs_dir / "manifests" / "multirepresentation_manifest.csv"
    manifest_df = pd.read_csv(manifest_path)
    dataset = MultiChannelLeafDataset(manifest_df, config, split="validation")

    # 3. Select 3 representative samples
    test_stems = [
        "PML_0_aug123",
        "BACT_0_aug2_aug4_aug157",
        "DML_19_aug4_aug5_aug16"
    ]
    
    fig, axes = plt.subplots(3, 6, figsize=(24, 12))
    metrics_log = []

    for i, stem in enumerate(test_stems):
        # Find index in dataset's filtered manifest
        matches = dataset.manifest_df[dataset.manifest_df['image_stem'] == stem]
        if matches.empty: 
            print(f"[WARNING] Stem {stem} not found in validation split.")
            continue
        idx = matches.index[0]
        
        batch = dataset[idx]
        image_tensor = batch["image"].unsqueeze(0).to(device)
        rgb_np = image_tensor[0, 0:3].permute(1, 2, 0).cpu().numpy()
        rgb_np = (rgb_np - rgb_np.min()) / (rgb_np.max() - rgb_np.min() + 1e-8)
        
        # Manual GT
        gt_path = manual_mask_dir / f"{stem}_mask.png"
        gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.resize(gt_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        # Predictions
        with torch.no_grad():
            ssl_out = ssl_model(image_tensor).logits
            ssl_out = torch.nn.functional.interpolate(ssl_out, size=(256, 256), mode='bilinear', align_corners=False)
            ssl_pred = torch.argmax(ssl_out, dim=1).squeeze().cpu().numpy()
            
            sft_out = sft_model(image_tensor).logits
            sft_out = torch.nn.functional.interpolate(sft_out, size=(256, 256), mode='bilinear', align_corners=False)
            sft_pred = torch.argmax(sft_out, dim=1).squeeze().cpu().numpy()
            
        # Visualization
        axes[i, 0].imshow(rgb_np)
        axes[i, 0].set_title(f"RGB ({stem})")
        axes[i, 1].imshow(gt_mask, cmap='tab10', vmin=0, vmax=9)
        axes[i, 1].set_title("Manual GT (Labelled)")
        axes[i, 2].imshow(ssl_pred, cmap='tab10', vmin=0, vmax=9)
        axes[i, 2].set_title("SSL Only (S6)")
        axes[i, 3].imshow(sft_pred, cmap='tab10', vmin=0, vmax=9)
        axes[i, 3].set_title("SSL + SFT (S8)")
        
        # Error maps (Prediction vs GT)
        ssl_error = (ssl_pred != gt_mask).astype(np.float32)
        sft_error = (sft_pred != gt_mask).astype(np.float32)
        
        axes[i, 4].imshow(ssl_error, cmap='Reds')
        axes[i, 4].set_title("SSL Error Map")
        axes[i, 5].imshow(sft_error, cmap='Reds')
        axes[i, 5].set_title("SFT Error Map")
        
        for j in range(6): axes[i, j].axis('off')
        
        # Compute individual IoU for this image
        m_ssl = SegmentationMetrics(9)
        m_ssl.update(torch.from_numpy(ssl_pred), torch.from_numpy(gt_mask))
        r_ssl = m_ssl.compute()
        
        m_sft = SegmentationMetrics(9)
        m_sft.update(torch.from_numpy(sft_pred), torch.from_numpy(gt_mask))
        r_sft = m_sft.compute()
        
        metrics_log.append({
            "stem": stem,
            "ssl_miou": r_ssl['miou'],
            "sft_miou": r_sft['miou']
        })

    plt.tight_layout()
    plt.savefig(output_dir / "ssl_vs_sft_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save Metrics CSV
    pd.DataFrame(metrics_log).to_csv(output_dir / "sft_improvement_metrics.csv", index=False)
    print(f"Generated comparison visualization and metrics in {output_dir}")

if __name__ == "__main__":
    main()
