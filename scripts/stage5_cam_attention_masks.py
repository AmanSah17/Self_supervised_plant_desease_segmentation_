"""
Stage 5: CAM-Guided Pseudo Mask Generation.
Orchestrates the fusion of anomaly maps and CAMs using spatial attention.
Outputs refined masks (.png) and metadata (.parquet).
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
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2

from lettuce_ssl_segmentation_lab.config import LabConfig
from lettuce_ssl_segmentation_lab.data.multichannel_dataset import MultiChannelLeafDataset
from lettuce_ssl_segmentation_lab.utils.feature_extractor import DINOv2FeatureExtractor
from lettuce_ssl_segmentation_lab.utils.background_segmenter import DINOv2BackgroundSegmenter
from lettuce_ssl_segmentation_lab.pipeline.classifier_head import DiseaseClassifierHead
from lettuce_ssl_segmentation_lab.pipeline.cam_attention_fusion import CAMAttentionRefiner
from lettuce_ssl_segmentation_lab.pipeline.orchestrator import SegmentationResearchOrchestrator

def main():
    print("\n" + "="*80)
    print("STAGE 5: CAM ATTENTION MASK GENERATION".center(80))
    print("="*80 + "\n")
    
    config = LabConfig().resolve()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = config.lab_root / "stage5_cam_attention_masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(exist_ok=True)
    
    orchestrator = SegmentationResearchOrchestrator(config)
    manifest_df, _ = orchestrator.build_manifest()
    
    # 0. Load Prerequisites
    extractor = DINOv2FeatureExtractor(model_name="dinov2_vitb14", device=device)
    bg_segmenter = DINOv2BackgroundSegmenter(device=device)
    refiner = CAMAttentionRefiner(device=device)
    
    # Load Stage 4 Head
    head_path = config.lab_root / "stage4_pseudo_masks" / "classifier_head_multiclass.pth"
    unique_classes = sorted(manifest_df['class_name'].unique())
    class_to_id = {cls: i for i, cls in enumerate(unique_classes)}
    
    head = DiseaseClassifierHead(input_dim=extractor.feature_dim(), num_classes=len(unique_classes))
    if head_path.exists():
        print(f"[INFO] Loading Stage 4 Classifier Head from {head_path}")
        head.load_state_dict(torch.load(head_path))
    else:
        print("[ERROR] Stage 4 Head not found. Please run Stage 4 first.")
        return 1
    head.to(device).eval()
    
    # Final segmentation mapping (same as Stage 4 for consistency)
    id_to_seg_label = {}
    for cls, idx in class_to_id.items():
        if cls == 'HLTY':
            id_to_seg_label[idx] = 1
        else:
            other_classes = [c for c in unique_classes if c != 'HLTY']
            id_to_seg_label[idx] = other_classes.index(cls) + 2
            
    # 1. Processing Loop
    print("[INFO] Starting mask generation with CAM Attention Fusion...")
    
    # Paths to Stage 3 anomaly maps
    stage3_dir = config.lab_root / "stage3_anomaly_localization"
    maps_dir = stage3_dir / "anomaly_maps"
    
    # Path to manual labels (Anchors)
    manual_labels_root = config.dataset_base / "Manual_labels"
    
    # Process all images in train and validation
    all_datasets = {
        "train": MultiChannelLeafDataset(manifest_df, config, split="train"),
        "validation": MultiChannelLeafDataset(manifest_df, config, split="validation")
    }
    
    metadata_records = []
    total_processed = 0
    is_smoke_test = os.environ.get("SMOKE_TEST", "false").lower() == "true"
    max_batches = 5 if is_smoke_test else None
    
    for split, dataset in all_datasets.items():
        print(f"[INFO] Generating refined masks for {split} split...")
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Stage 5 Fusion ({split})")):
            if max_batches is not None and batch_idx >= max_batches:
                break
                
            images = batch["image"]
            stems = batch["image_stem"]
            class_names = batch["class_name"]
            segments_batch = batch["segments"].numpy()
            
            with torch.no_grad():
                patch_features = extractor.extract_patch_features(images[:, 0:3, :, :].to(device))
                
                for i in range(len(images)):
                    stem = stems[i]
                    cls_name = class_names[i]
                    cls_id = class_to_id[cls_name]
                    seg_label = id_to_seg_label[cls_id]
                    
                    # A. CAM Generation
                    cam = head.generate_cam(patch_features[i:i+1], cls_id)
                    cam_np = torch.nn.functional.interpolate(
                        cam, size=images.shape[2:], mode='bilinear', align_corners=False
                    ).cpu().numpy()[0, 0, :, :]
                    
                    # B. Foreground Segmentation
                    fg_map = bg_segmenter.segment(
                        images[i, 0:3, :, :].permute(1, 2, 0).numpy(), 
                        patch_features[i:i+1]
                    )
                    
                    # C. Anomaly Retrieval
                    map_path = maps_dir / f"{stem}_anomaly.npy"
                    anomaly_map = np.load(map_path) if map_path.exists() else np.zeros(images.shape[2:])
                    # Normalize anomaly map to [0, 1]
                    if anomaly_map.max() > anomaly_map.min():
                        anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
                    
                    # D. Attention Fusion Refinement
                    pseudo_mask_refined = refiner.refine(
                        images[i, 0:3, :, :].permute(1, 2, 0).numpy(),
                        anomaly_map,
                        cam_np,
                        fg_map,
                        segments_batch[i],
                        cat_label=seg_label,
                        threshold=0.55 # Slightly stricter for higher quality
                    )
                    
                    # ANCHORING LOGIC: Check for manual ground truth mask
                    manual_mask_path = manual_labels_root / split / f"{stem}_mask.png"
                    if manual_mask_path.exists():
                        # Load manual mask and use it as anchor
                        manual_mask = cv2.imread(str(manual_mask_path), cv2.IMREAD_GRAYSCALE)
                        manual_mask = cv2.resize(manual_mask, images.shape[2:][::-1], interpolation=cv2.INTER_NEAREST)
                        
                        # If manual_mask is 1 (disease), set it to seg_label.
                        # Otherwise, use the refined SSL mask (which handles leaf/BG).
                        is_disease = (manual_mask > 0)
                        pseudo_mask = pseudo_mask_refined.copy()
                        pseudo_mask[is_disease] = seg_label
                        anchored = True
                    else:
                        pseudo_mask = pseudo_mask_refined
                        anchored = False
                    
                    # E. Storage
                    mask_path = masks_dir / f"{stem}_mask.png"
                    cv2.imwrite(str(mask_path), pseudo_mask)
                    
                    metadata_records.append({
                        "image_stem": stem,
                        "class_name": cls_name,
                        "seg_label": seg_label,
                        "mask_path": str(mask_path),
                        "mean_anomaly": float(anomaly_map.mean()),
                        "mean_cam": float(cam_np.mean()),
                        "fg_coverage": float((fg_map > 0.35).mean()),
                        "anchored": anchored
                    })
                    total_processed += 1
                
    # 2. Save Metadata as Parquet (Arque)
    if metadata_records:
        metadata_df = pd.DataFrame(metadata_records)
        parquet_path = output_dir / "mask_generation_metadata.parquet"
        metadata_df.to_parquet(parquet_path, engine='pyarrow', index=False)
        print(f"[OK] Metadata saved to {parquet_path}")

    print("\n" + "="*80)
    print(f"Stage 5 Completed: {total_processed} masks generated".center(80))
    print("="*80 + "\n")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
