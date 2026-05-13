import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import random

# Add project root to path
project_root = Path(r'd:\gemma4\segmentation_lattuce-desease')
sys.path.append(str(project_root))

from lettuce_ssl_segmentation_lab.utils.feature_extractor import DINOv2FeatureExtractor
from lettuce_ssl_segmentation_lab.pipeline.anomaly_detector import PaDiMDetector
from lettuce_ssl_segmentation_lab.pipeline.classifier_head import DiseaseClassifierHead
from lettuce_ssl_segmentation_lab.pipeline.pseudo_mask_generator import PseudoMaskGenerator
from lettuce_ssl_segmentation_lab.utils.background_segmenter import DINOv2BackgroundSegmenter

def get_random_image(base_path, class_name):
    class_dir = base_path / class_name
    all_images = list(class_dir.glob('*.jpg'))
    return random.choice(all_images)

def main():
    # 1. Setup
    config_root = project_root / 'lettuce_ssl_segmentation_lab'
    model_dir = config_root / 'stage3_anomaly_localization'
    head_path = config_root / 'stage4_pseudo_masks' / 'classifier_head_multiclass.pth'
    dataset_base = project_root / 'Lettuce_disease_datasets_split' / 'train'
    output_dir = project_root / 'Report' / 'stage4_assets'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. Load Models
    padim = PaDiMDetector.load(model_dir / "padim_model.pkl", device=device)
    extractor = DINOv2FeatureExtractor(model_name="dinov2_vitb14", device=device)
    head = DiseaseClassifierHead(num_classes=8)
    head.load(head_path, device=device)
    bg_segmenter = DINOv2BackgroundSegmenter(device=device)
    generator = PseudoMaskGenerator(anomaly_weight=0.6, cam_weight=0.4)

    # 3. Define Classes and Mapping
    classes = sorted(['HLTY', 'BACT', 'DML', 'PML', 'SBL', 'SPW', 'VIRL', 'WLBL'])
    class_to_id = {cls: i for i, cls in enumerate(classes)}
    
    # Select samples
    samples = []
    for cls in classes:
        samples.append((cls, get_random_image(dataset_base, cls)))

    print(f"Processing {len(samples)} samples for Stage 4 visualization...")

    for cls_name, img_path in samples:
        stem = img_path.stem
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb_256 = cv2.resize(img_rgb, (256, 256))
        
        input_tensor = torch.from_numpy(img_rgb_256).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        # A. Anomaly Map
        anomaly_map = padim.score(input_tensor, extractor)
        am_np = anomaly_map.squeeze().cpu().numpy()
        
        # B. CAM Map
        with torch.no_grad():
            patch_feats = extractor.extract_patch_features(input_tensor.to(device))
            cam = head.generate_cam(patch_feats, class_to_id[cls_name])
            cam_np = torch.nn.functional.interpolate(
                cam, size=(256, 256), mode='bilinear', align_corners=False
            ).squeeze().cpu().numpy()
            
        # C. Fusion Evidence
        evidence = 0.6 * am_np + 0.4 * cam_np
        
        # D. Foreground Segment
        fg_map = bg_segmenter.segment(img_rgb_256.astype(np.float32) / 255.0, patch_feats)
        
        # E. Pseudo Mask (Mocking segments for now, or loading if available)
        # For viz, we'll just show the refined mask using a simple threshold if segments missing
        # But we want to show the REAL one from generator if possible.
        # We'll use a dummy segment map (all zeros or grid) just to satisfy the method if needed
        segments = np.zeros((256, 256), dtype=int)
        pseudo_mask = generator.refine_multi_class(
            fg_map, evidence, segments, cat_label=class_to_id[cls_name]+1, 
            is_healthy_image=(cls_name == 'HLTY')
        )

        # Plot 5-way comparison
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        
        axes[0].imshow(img_rgb_256)
        axes[0].set_title(f"RGB ({cls_name})")
        axes[0].axis('off')
        
        axes[1].imshow(am_np, cmap='jet')
        axes[1].set_title("Anomaly Map (S3)")
        axes[1].axis('off')
        
        axes[2].imshow(cam_np, cmap='magma')
        axes[2].set_title("CAM Map (S4)")
        axes[2].axis('off')
        
        axes[3].imshow(evidence, cmap='hot')
        axes[3].set_title("Fused Evidence")
        axes[3].axis('off')
        
        axes[4].imshow(pseudo_mask, cmap='tab10')
        axes[4].set_title("Final Pseudo Mask")
        axes[4].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"fusion_step_{cls_name.lower()}.png", bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved fusion_step_{cls_name.lower()}.png")

    # 4. Create "How CAM works" detailed plot
    # We'll take one diseased sample (BACT) and show different class CAMs
    bact_sample = samples[0][1] if samples[0][0] == 'BACT' else samples[1][1]
    img_bgr = cv2.imread(str(bact_sample))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb_256 = cv2.resize(img_rgb, (256, 256))
    input_tensor = torch.from_numpy(img_rgb_256).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    
    with torch.no_grad():
        patch_feats = extractor.extract_patch_features(input_tensor.to(device))
        
    plt.figure(figsize=(12, 12))
    plt.subplot(3, 3, 1)
    plt.imshow(img_rgb_256)
    plt.title("Original BACT Image")
    plt.axis('off')
    
    # Show CAMs for 8 classes
    for i, cls in enumerate(classes):
        with torch.no_grad():
            cam = head.generate_cam(patch_feats, i)
            cam_np = torch.nn.functional.interpolate(
                cam, size=(256, 256), mode='bilinear', align_corners=False
            ).squeeze().cpu().numpy()
            
        plt.subplot(3, 3, i+2)
        plt.imshow(cam_np, cmap='viridis')
        plt.title(f"CAM for: {cls}")
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(output_dir / "cam_concept_explanation.png", bbox_inches='tight', dpi=200)
    plt.close()
    print("Saved cam_concept_explanation.png")

if __name__ == '__main__':
    main()
