"""
Revised Stage 4 Execution Script: Multi-Class Pseudo Mask Generation.
Trains an 8-class classifier head and fuses evidence to create 3-way pseudo-labels.
Mapping: 0: Background, 1: Healthy Leaf, 2-8: Specific Disease/Weed category.
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
import cv2

from lettuce_ssl_segmentation_lab.config import LabConfig
from lettuce_ssl_segmentation_lab.data.multichannel_dataset import MultiChannelLeafDataset
from lettuce_ssl_segmentation_lab.utils.feature_extractor import DINOv2FeatureExtractor
from lettuce_ssl_segmentation_lab.utils.background_segmenter import DINOv2BackgroundSegmenter
from lettuce_ssl_segmentation_lab.pipeline.classifier_head import DiseaseClassifierHead
from lettuce_ssl_segmentation_lab.pipeline.pseudo_mask_generator import PseudoMaskGenerator
from lettuce_ssl_segmentation_lab.pipeline.orchestrator import SegmentationResearchOrchestrator

def print_section(title: str, width: int = 80):
    print(f"\n{'='*width}")
    print(f"{title.center(width)}")
    print(f"{'='*width}\n")

def main():
    print_section("REVISED STAGE 4: MULTI-CLASS PSEUDO MASK GENERATION", 80)
    
    config = LabConfig().resolve()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = config.lab_root / "stage4_pseudo_masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    is_smoke_test = os.environ.get("SMOKE_TEST", "false").lower() == "true"
    
    orchestrator = SegmentationResearchOrchestrator(config)
    manifest_df, _ = orchestrator.build_manifest()
    
    # 0. Setup Class Mapping
    unique_classes = sorted(manifest_df['class_name'].unique())
    class_to_id = {cls: i for i, cls in enumerate(unique_classes)}
    # Final segmentation mapping: 0: BG, 1: HLTY, 2+: Others
    # We want HLTY to be 1.
    id_to_seg_label = {}
    for cls, idx in class_to_id.items():
        if cls == 'HLTY':
            id_to_seg_label[idx] = 1
        else:
            # Shift others to 2, 3, ... based on their order
            other_classes = [c for c in unique_classes if c != 'HLTY']
            id_to_seg_label[idx] = other_classes.index(cls) + 2
            
    print(f"[INFO] Class to Classifier ID: {class_to_id}")
    print(f"[INFO] Classifier ID to Seg Label: {id_to_seg_label}")
    
    # 1. Stage 4a: Train Multi-Class Classifier Head
    print_section("Stage 4a: Training Multi-Class Head")
    
    extractor = DINOv2FeatureExtractor(model_name="dinov2_vitb14", device=device)
    
    # Check for cached features
    all_features = []
    all_labels = []
    
    # Try to use Stage 2 healthy features if available
    stage2_dir = config.lab_root / "stage2_healthy_learning"
    healthy_features_path = stage2_dir / "healthy_features_train.npy"
    if healthy_features_path.exists():
        print("[INFO] Loading Stage 2 healthy features...")
        h_feats = np.load(healthy_features_path)
        all_features.append(h_feats)
        all_labels.append(np.full(len(h_feats), class_to_id['HLTY']))
    
    # Extract features for all other classes
    print("[INFO] Extracting features for all classes...")
    train_dataset = MultiChannelLeafDataset(manifest_df, config, split="train")
    
    # Limit for smoke test
    max_cls_samples = 100 if is_smoke_test else 1000
    
    for cls_name in unique_classes:
        if cls_name == 'HLTY' and healthy_features_path.exists():
            continue # already loaded
            
        cls_indices = [i for i, row in train_dataset.manifest_df.iterrows() if row["class_name"] == cls_name]
        subset_indices = cls_indices[:max_cls_samples]
        subset = Subset(train_dataset, subset_indices)
        loader = DataLoader(subset, batch_size=32, shuffle=False)
        
        cls_feats = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Features ({cls_name})"):
                images = batch["image"][:, 0:3, :, :].to(device)
                feats = extractor.extract(images)
                cls_feats.append(feats.cpu().numpy())
        
        if cls_feats:
            cls_feats = np.concatenate(cls_feats, axis=0)
            all_features.append(cls_feats)
            all_labels.append(np.full(len(cls_feats), class_to_id[cls_name]))
            
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Train 8-class head
    head = DiseaseClassifierHead(input_dim=extractor.feature_dim(), num_classes=len(unique_classes))
    head.train_head(all_features, all_labels, epochs=20)
    head.save(output_dir / "classifier_head_multiclass.pth")
    
    # 2. Stage 4b: Generate Multi-Class Pseudo Masks
    print_section("Stage 4b: Generating Multi-Class Pseudo Masks")
    
    bg_segmenter = DINOv2BackgroundSegmenter(device=device)
    generator = PseudoMaskGenerator(anomaly_weight=0.9, cam_weight=0.1, fg_threshold=0.35, cat_threshold=0.7)
    
    # Path to Stage 3 anomaly maps
    stage3_dir = config.lab_root / "stage3_anomaly_localization"
    maps_dir = stage3_dir / "anomaly_maps"
    
    # Output dirs
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(exist_ok=True)
    viz_dir = output_dir / "fusion_viz"
    viz_dir.mkdir(exist_ok=True)
    
    # Path to manual labels (Anchors)
    manual_labels_root = config.dataset_base / "Manual_labels"
    
    # Process all images in train and val
    all_datasets = {
        "train": MultiChannelLeafDataset(manifest_df, config, split="train"),
        "validation": MultiChannelLeafDataset(manifest_df, config, split="validation")
    }
    
    total_processed = 0
    max_gen_batches = 2 if is_smoke_test else None
    
    for split, dataset in all_datasets.items():
        print(f"[INFO] Generating pseudo masks for {split} split...")
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Generating masks ({split})")):
            if max_gen_batches is not None and batch_idx >= max_gen_batches:
                break
                
            images = batch["image"]
            stems = batch["image_stem"]
            class_names = batch["class_name"]
            segments_batch = batch["segments"].numpy()
            
            with torch.no_grad():
                patch_features = extractor.extract_patch_features(images[:, 0:3, :, :].to(device))
                # Generate CAMs for each class in the batch
                cams_batch = []
                for i in range(len(images)):
                    cls_id = class_to_id[class_names[i]]
                    cam = head.generate_cam(patch_features[i:i+1], cls_id)
                    cam_full = torch.nn.functional.interpolate(
                        cam, size=images.shape[2:], mode='bilinear', align_corners=False
                    ).cpu().numpy()[0, 0, :, :]
                    cams_batch.append(cam_full)
            
            for i in range(len(images)):
                stem = stems[i]
                cls_name = class_names[i]
                cls_id = class_to_id[cls_name]
                seg_label = id_to_seg_label[cls_id]
                
                # Foreground mask
                fg_map = bg_segmenter.segment(
                    images[i, 0:3, :, :].permute(1, 2, 0).numpy(), 
                    patch_features[i:i+1]
                )
                
                # Anomaly map (from Stage 3)
                map_path = maps_dir / f"{stem}_anomaly.npy"
                if map_path.exists():
                    anomaly_map = np.load(map_path)
                else:
                    # If healthy or map missing, use zeros
                    anomaly_map = np.zeros(images.shape[2:])
                
                cam_map = cams_batch[i]
                segments = segments_batch[i]
                
                # Multi-class refinement
                pseudo_mask_ssl = generator.refine_multi_class(
                    fg_map, 
                    (0.6 * anomaly_map + 0.4 * cam_map), # Simple fusion for evidence
                    segments,
                    cat_label=seg_label,
                    is_healthy_image=(cls_name == 'HLTY')
                )
                
                # ANCHORING LOGIC: Check for manual ground truth mask
                manual_mask_path = manual_labels_root / split / f"{stem}_mask.png"
                if manual_mask_path.exists():
                    # Load manual mask and use it as anchor
                    manual_mask = cv2.imread(str(manual_mask_path), cv2.IMREAD_GRAYSCALE)
                    manual_mask = cv2.resize(manual_mask, images.shape[2:][::-1], interpolation=cv2.INTER_NEAREST)
                    # For diseased images in Roboflow, the mask is 1 (Disease). 
                    # We map it to our seg_label.
                    pseudo_mask = np.where(manual_mask > 0, seg_label, pseudo_mask_ssl)
                    # Ensure background is still background if manual label is zero but SSL said category?
                    # Actually, if we have GT, GT is the source of truth.
                    # But Roboflow GT only masks the disease. The rest is leaf (1) or BG (0).
                    # Our pseudo_mask_ssl already separates leaf from BG.
                    # So we use manual_mask for category, and pseudo_mask_ssl for leaf/BG structure.
                    
                    # If manual_mask is 1 (disease), set it to seg_label.
                    # If manual_mask is 0, we trust pseudo_mask_ssl (which could be 0:BG or 1:HLTY).
                    is_disease = (manual_mask > 0)
                    pseudo_mask = pseudo_mask_ssl.copy()
                    pseudo_mask[is_disease] = seg_label
                    anchored = True
                else:
                    pseudo_mask = pseudo_mask_ssl
                    anchored = False
                
                # Save mask
                mask_path = masks_dir / f"{stem}_mask.png"
                cv2.imwrite(str(mask_path), pseudo_mask) # Save raw values for training
                
                # Visualization
                if total_processed < 40:
                    viz = generator.visualize_multi_class(
                        images[i, 0:3, :, :].permute(1, 2, 0).numpy(),
                        fg_map,
                        (0.6 * anomaly_map + 0.4 * cam_map),
                        pseudo_mask,
                        f"{cls_name} (Anchored)" if anchored else cls_name
                    )
                    plt.imsave(viz_dir / f"{stem}_fusion.png", viz)
                
                total_processed += 1

    print_section("Stage 4 Completed Successfully")
    print(f"[INFO] Total multi-class pseudo masks generated: {total_processed}")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
