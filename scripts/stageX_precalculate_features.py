"""
Stage X: Offline Pre-calculation of 14-Channel Representation Stack.
Eliminates CPU bottlenecks by pre-computing all channels (ExG, Watershed, Edges, etc.) 
and saving them as optimized uint8 tensors.
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
from tqdm import tqdm
import concurrent.futures
from lettuce_ssl_segmentation_lab.config import LabConfig
from lettuce_ssl_segmentation_lab.data.multichannel_dataset import MultiChannelLeafDataset
from lettuce_ssl_segmentation_lab.pipeline.orchestrator import SegmentationResearchOrchestrator





def process_and_save(row, dataset, output_dir: Path):
    try:
        # We reuse the logic from the dataset but without transforms
        image_stem = row['image_stem']
        feat_path = output_dir / f"{image_stem}_feat.npy"
        
        if feat_path.exists():
            return True
            
        # Get index in dataset
        idx = row.name # assuming row is from df.iterrows() or similar
        
        # We need a way to get the raw 14-channel stack
        # I'll add a method to the dataset for this or just call __getitem__ 
        # and extract the tensor before any random augmentations.
        
        sample = dataset[idx]
        tensor = sample["image"] # (14, H, W) float32
        segments = sample["segments"] # (H, W) int64
        
        # Save features as uint8
        feat_uint8 = (tensor.numpy() * 255).astype(np.uint8)
        np.save(feat_path, feat_uint8)
        
        # Save segments as int32
        seg_path = output_dir / f"{image_stem}_seg.npy"
        np.save(seg_path, segments.numpy().astype(np.int32))
        
        return True
    except Exception as e:
        print(f"Error processing {row.get('image_stem')}: {e}")
        return False



def main():
    config = LabConfig().resolve()
    orchestrator = SegmentationResearchOrchestrator(config)
    manifest_df, _ = orchestrator.build_manifest()
    
    # Filter to only include samples that have Stage 5 masks (training set)
    mask_dir = config.lab_root / "stage5_cam_attention_masks" / "masks"
    def has_mask(row):
        return (mask_dir / f"{row['image_stem']}_mask.png").exists()
    
    manifest_df = manifest_df[manifest_df.apply(has_mask, axis=1)].reset_index(drop=True)
    print(f"[INFO] Found {len(manifest_df)} images to pre-calculate.")
    
    output_dir = config.lab_root / "stageX_precalculated_features"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize dataset WITHOUT transforms
    dataset = MultiChannelLeafDataset(manifest_df, config, transform=None)
    
    print(f"[INFO] Starting pre-calculation using {os.cpu_count()} workers...")
    
    # Process in parallel using ThreadPoolExecutor for I/O bound or ProcessPoolExecutor for CPU bound
    # Since compute_edge, exg, etc. are CPU intensive, ProcessPoolExecutor is better.
    # But wait, numba and cv2 usually release GIL. ThreadPool might be enough and safer on Windows.
    # Actually, ProcessPool is better for CPU bound tasks.
    
    with tqdm(total=len(manifest_df)) as pbar:
        # We'll do it in chunks to avoid memory issues if using ProcessPool
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for i, row in manifest_df.iterrows():
                # We need to pass the dataset carefully. 
                # Dataset[i] loads files, which is fine in threads.
                futures.append(executor.submit(process_and_save, row, dataset, output_dir))
            
            for future in concurrent.futures.as_completed(futures):
                future.result()
                pbar.update(1)

    print(f"\n[OK] Pre-calculation complete. Saved to {output_dir}")

if __name__ == "__main__":
    main()
