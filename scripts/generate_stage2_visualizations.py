import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
project_root = Path(r'd:\gemma4\segmentation_lattuce-desease')
sys.path.append(str(project_root))

from lettuce_ssl_segmentation_lab.utils.feature_extractor import DINOv2FeatureExtractor

def main():
    image_path = Path(r'D:\gemma4\segmentation_lattuce-desease\Lettuce_disease_datasets_split\train\HLTY\HLTY_0.jpg')
    output_dir = project_root / 'Report' / 'stage2_assets'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize extractor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = DINOv2FeatureExtractor(model_name="dinov2_vitb14", device=device)

    # 2. Load and preprocess image
    img_bgr = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb_256 = cv2.resize(img_rgb, (256, 256))
    
    # (B, 3, 256, 256)
    input_tensor = torch.from_numpy(img_rgb_256).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # 3. Extract patch features
    # Shape: (B, D, H_p, W_p)
    with torch.no_grad():
        patch_features = extractor.extract_patch_features(input_tensor)
    
    features_np = patch_features.squeeze(0).cpu().numpy() # (D, H_p, W_p)
    D, Hp, Wp = features_np.shape
    print(f"Extracted patch features: {features_np.shape}")

    # 4. Visualize Feature Activation Map (L2 Norm of features)
    feat_norm = np.linalg.norm(features_np, axis=0) # (H_p, W_p)
    feat_norm_resized = cv2.resize(feat_norm, (256, 256), interpolation=cv2.INTER_LINEAR)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb_256)
    plt.title("Original Healthy Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(feat_norm_resized, cmap='viridis')
    plt.colorbar(label='Feature Norm Intensity')
    plt.title("DINOv2 Feature Activation (Healthy)")
    plt.axis('off')
    
    plt.savefig(output_dir / '01_healthy_activation.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved 01_healthy_activation.png")

    # 5. Visualize some individual feature channels (first 4)
    plt.figure(figsize=(12, 3))
    for i in range(4):
        plt.subplot(1, 4, i+1)
        chan = features_np[i]
        chan_resized = cv2.resize(chan, (256, 256), interpolation=cv2.INTER_LINEAR)
        plt.imshow(chan_resized, cmap='inferno')
        plt.title(f"Channel {i}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / '02_feature_channels.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved 02_feature_channels.png")

    # 6. Illustrate Patch Grid
    grid_img = img_rgb_256.copy()
    patch_size = 14
    for x in range(0, 252, patch_size):
        cv2.line(grid_img, (x, 0), (x, 252), (255, 255, 255), 1)
    for y in range(0, 252, patch_size):
        cv2.line(grid_img, (0, y), (252, y), (255, 255, 255), 1)
        
    cv2.imwrite(str(output_dir / '03_patch_grid.png'), cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR))
    print("Saved 03_patch_grid.png")

if __name__ == '__main__':
    main()
