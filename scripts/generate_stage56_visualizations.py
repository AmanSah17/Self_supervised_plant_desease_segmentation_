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

from lettuce_ssl_segmentation_lab.data.multichannel_dataset import MultiChannelLeafDataset
from lettuce_ssl_segmentation_lab.config import LabConfig

def main():
    config = LabConfig().resolve()
    output_dir = project_root / 'Report' / 'stage56_assets'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Visualize 14-Channel Stack Sample
    # We'll use the dataset logic to get one sample
    # We need a manifest
    from lettuce_ssl_segmentation_lab.pipeline.orchestrator import SegmentationResearchOrchestrator
    orchestrator = SegmentationResearchOrchestrator(config)
    manifest_df, _ = orchestrator.build_manifest()
    
    dataset = MultiChannelLeafDataset(manifest_df, config, split="train")
    sample = dataset[10] # Pick a diseased sample
    tensor = sample["image"].numpy() # (14, 256, 256)
    
    channel_names = [
        "R", "G", "B", 
        "CLAHE_R", "CLAHE_G", "CLAHE_B",
        "ExG", "Edges", "Watershed", 
        "H", "S", "V", 
        "ExG_N", "Sobel_N"
    ]
    
    plt.figure(figsize=(15, 10))
    for i in range(14):
        plt.subplot(3, 5, i+1)
        plt.imshow(tensor[i], cmap='gray' if i not in [0,1,2] else None)
        plt.title(channel_names[i])
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "14_channel_visualization.png", bbox_inches='tight', dpi=200)
    plt.close()
    print("Saved 14_channel_visualization.png")

    # 2. Stage 5: Refined Mask Comparison
    # Compare Stage 4 mask vs Stage 5 mask if available
    s4_dir = config.lab_root / "stage4_pseudo_masks" / "masks"
    s5_dir = config.lab_root / "stage5_cam_attention_masks" / "masks"
    
    sample_stem = "BACT_10"
    s4_path = s4_dir / f"{sample_stem}_mask.png"
    s5_path = s5_dir / f"{sample_stem}_mask.png"
    
    if s4_path.exists() and s5_path.exists():
        s4_mask = cv2.imread(str(s4_path), cv2.IMREAD_GRAYSCALE)
        s5_mask = cv2.imread(str(s5_path), cv2.IMREAD_GRAYSCALE)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(s4_mask, cmap='tab10')
        plt.title("Stage 4: Base Pseudo Mask")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(s5_mask, cmap='tab10')
        plt.title("Stage 5: Refined Attention Mask")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / "mask_refinement_comparison.png", bbox_inches='tight', dpi=200)
        plt.close()
        print("Saved mask_refinement_comparison.png")

if __name__ == '__main__':
    main()
