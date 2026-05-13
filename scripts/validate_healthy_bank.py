import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import random
from tqdm import tqdm

# Add project root to path
project_root = Path(r'd:\gemma4\segmentation_lattuce-desease')
sys.path.append(str(project_root))

from lettuce_ssl_segmentation_lab.utils.feature_extractor import DINOv2FeatureExtractor
from lettuce_ssl_segmentation_lab.pipeline.anomaly_detector import PaDiMDetector

def get_random_images(base_path, class_name, count=2):
    class_dir = base_path / class_name
    all_images = list(class_dir.glob('*.jpg'))
    return random.sample(all_images, min(count, len(all_images)))

def main():
    # 1. Setup paths
    model_path = project_root / 'lettuce_ssl_segmentation_lab' / 'stage3_anomaly_localization' / 'padim_model.pkl'
    dataset_base = project_root / 'Lettuce_disease_datasets_split' / 'train'
    output_dir = project_root / 'Report' / 'validation_assets'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load Model and Extractor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading PaDiM model from {model_path} on {device}...")
    padim = PaDiMDetector.load(model_path, device=device)
    extractor = DINOv2FeatureExtractor(model_name="dinov2_vitb14", device=device)

    # 3. Select Images
    # 4 Healthy
    healthy_images = get_random_images(dataset_base, 'HLTY', count=4)
    
    # 2 from each diseased class
    diseased_classes = ['BACT', 'DML', 'PML', 'SBL', 'SPW', 'VIRL', 'WLBL']
    diseased_images = {}
    for cls in diseased_classes:
        diseased_images[cls] = get_random_images(dataset_base, cls, count=2)

    # 4. Process and Visualize
    all_samples = [('Healthy', p) for p in healthy_images]
    for cls in diseased_classes:
        for p in diseased_images[cls]:
            all_samples.append((cls, p))

    print(f"Processing {len(all_samples)} samples...")
    
    # We will create a combined plot for each class
    for cls_name in ['Healthy'] + diseased_classes:
        samples = [s for s in all_samples if s[0] == cls_name]
        if not samples: continue
        
        fig, axes = plt.subplots(len(samples), 3, figsize=(15, 5 * len(samples)))
        if len(samples) == 1: axes = [axes]
        
        for i, (label, img_path) in enumerate(samples):
            # Load and preprocess
            img_bgr = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_rgb_256 = cv2.resize(img_rgb, (256, 256))
            
            # (B, 3, 256, 256)
            input_tensor = torch.from_numpy(img_rgb_256).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            
            # Compute anomaly map
            # Returns (B, 1, 256, 256)
            anomaly_map = padim.score(input_tensor, extractor)
            anomaly_map_np = anomaly_map.squeeze().cpu().numpy()
            
            # Normalize for visualization
            am_norm = (anomaly_map_np - anomaly_map_np.min()) / (anomaly_map_np.max() - anomaly_map_np.min() + 1e-8)
            
            # Subplot 1: Original
            axes[i][0].imshow(img_rgb_256)
            axes[i][0].set_title(f"{label} Original\n{img_path.name}")
            axes[i][0].axis('off')
            
            # Subplot 2: Anomaly Heatmap
            im2 = axes[i][1].imshow(anomaly_map_np, cmap='jet')
            axes[i][1].set_title(f"Mahalanobis Distance\n(Deviation from Healthy)")
            axes[i][1].axis('off')
            plt.colorbar(im2, ax=axes[i][1], fraction=0.046, pad=0.04)
            
            # Subplot 3: Overlay
            heatmap_color = cv2.applyColorMap((am_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(img_rgb_256, 0.6, heatmap_color, 0.4, 0)
            axes[i][2].imshow(overlay)
            axes[i][2].set_title("Anomaly Overlay")
            axes[i][2].axis('off')

        plt.tight_layout()
        save_path = output_dir / f"validation_{cls_name.lower()}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved {save_path}")

    # 5. Create a Summary Figure (One from each class)
    print("Creating summary figure...")
    summary_samples = []
    summary_samples.append(('Healthy', healthy_images[0]))
    for cls in diseased_classes:
        summary_samples.append((cls, diseased_images[cls][0]))
        
    fig, axes = plt.subplots(len(summary_samples), 2, figsize=(10, 3 * len(summary_samples)))
    for i, (label, img_path) in enumerate(summary_samples):
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb_256 = cv2.resize(img_rgb, (256, 256))
        input_tensor = torch.from_numpy(img_rgb_256).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        anomaly_map = padim.score(input_tensor, extractor)
        anomaly_map_np = anomaly_map.squeeze().cpu().numpy()
        
        axes[i][0].imshow(img_rgb_256)
        axes[i][0].set_title(f"Class: {label}")
        axes[i][0].axis('off')
        
        im = axes[i][1].imshow(anomaly_map_np, cmap='jet')
        axes[i][1].set_title("Anomaly Score")
        axes[i][1].axis('off')
        plt.colorbar(im, ax=axes[i][1])

    plt.tight_layout()
    plt.savefig(output_dir / "00_comparison_summary.png", bbox_inches='tight', dpi=200)
    plt.close()
    print("Saved 00_comparison_summary.png")

if __name__ == '__main__':
    main()
