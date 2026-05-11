#!/usr/bin/env python3
"""
FELZENSZWALB SEGMENTATION - QUICK START GUIDE
==============================================

This file provides quick reference for using the Felzenszwalb segmentation pipeline.
"""

# ============================================================================
# OPTION 1: QUICK START WITH DEFAULT PARAMETERS (5 minutes)
# ============================================================================

"""
If you want to generate masks immediately with reasonable default parameters:

1. Open terminal in the workspace directory
2. Run:
   
   python felzenszwalb_segmentation_gpu.py

3. Wait for processing to complete
4. Results saved to: felzenszwalb_masks_output/
"""


# ============================================================================
# OPTION 2: OPTIMIZE PARAMETERS FIRST (30-45 minutes)
# ============================================================================

"""
For better results, optimize hyperparameters first:

Step 1: Run hyperparameter tuning (on validation set)
   python felzenszwalb_hyperparameter_tuning.py
   
   This will:
   - Test 5×5×5 = 125 parameter combinations
   - Evaluate on 3 images per class (3 seconds each)
   - Provide recommendations in JSON format
   
Step 2: Review recommendations
   - Open: felzenszwalb_hyperparameter_tuning/recommendations.json
   - Note the "best_uniformity" parameters for consistent segmentation
   - Or use "best_boundary" for tight boundary following
   
Step 3: Update main script
   - Edit: felzenszwalb_segmentation_gpu.py
   - Find: FelzenszwalbConfig()
   - Update:
       config.felz_scale = 120.0      # From recommendations
       config.felz_sigma = 0.6        # From recommendations
       config.felz_min_size = 12      # From recommendations
   
Step 4: Run full dataset processing
   python felzenszwalb_segmentation_gpu.py
"""


# ============================================================================
# OPTION 3: BATCH PROCESSING MULTIPLE CONFIGURATIONS (varies)
# ============================================================================

"""
Test multiple configurations simultaneously:

Step 1: Use batch processor for pre-defined configurations
   python felzenszwalb_batch_processor.py
   
   Menu options:
   1 - Lettuce Disease Defaults (fine to coarse)
   2 - Edge-Focused Configs (for disease boundaries)
   3 - Multi-Scale Analysis (compare scales)
   4 - Custom Configuration
   5 - Compare Results
   
Step 2: Each configuration prepares output directory structure

Step 3: Update felzenszwalb_segmentation_gpu.py for each config
   or modify the script to accept config files
"""


# ============================================================================
# OUTPUT STRUCTURE
# ============================================================================

"""
After running, you'll have:

felzenszwalb_masks_output/
├── train/
│   ├── BACT/
│   │   ├── image_name_s120_0_sg0_6_ms12_raw.png          ← Use for training
│   │   ├── image_name_s120_0_sg0_6_ms12_colored.png      ← Visual check
│   │   ├── image_name_s120_0_sg0_6_ms12_boundary.png     ← Boundary visualization
│   │   └── image_name_s120_0_sg0_6_ms12_info.txt         ← Metadata
│   ├── DML/
│   └── [other classes]
├── validation/
├── test/
└── processing_summary.csv                                  ← Quality report

Files per image:
- *_raw.png: Grayscale segment labels (0-255) → USE THIS FOR TRAINING
- *_colored.png: RGB visualization
- *_boundary.png: Segment boundaries
- *_info.txt: Parameters and metadata
"""


# ============================================================================
# FILE FORMAT FOR TRAINING
# ============================================================================

"""
The *_raw.png files are ready for semantic segmentation training.

Each pixel value (0-255) represents a different segment/label.

Example PyTorch usage:

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class LetuceSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None):
        self.image_paths = sorted(image_dir.glob("*.jpg"))
        self.mask_paths = sorted(mask_dir.glob("*_raw.png"))
        self.transforms = transforms
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx])
        
        if self.transforms:
            image = self.transforms(image)
        
        mask = torch.tensor(np.array(mask), dtype=torch.long)
        image = torch.tensor(np.array(image) / 255.0, dtype=torch.float32)
        
        return image, mask

# Usage
train_dataset = LetuceSegmentationDataset(
    image_dir=Path("Lettuce_disease_datasets_split/train"),
    mask_dir=Path("felzenszwalb_masks_output/train"),
    transforms=my_transforms
)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
"""


# ============================================================================
# PERFORMANCE OPTIMIZATION
# ============================================================================

"""
To speed up processing:

1. Check GPU is being used:
   - Open new terminal
   - Run: nvidia-smi -l 1
   - Look for Python process using GPU
   
2. Disable unnecessary outputs:
   config.save_colored_masks = False      # Skip RGB visualization
   config.save_boundary_masks = False     # Skip boundary masks
   
3. Increase min_size to reduce tiny segments:
   config.felz_min_size = 20      # Default 12 → 20
   
4. Disable edge enhancement if not needed:
   config.apply_edge_enhancement = False

5. Process fewer images for testing:
   config.samples_per_class = 10  # Test on 10 images per class
"""


# ============================================================================
# PARAMETER EXPLANATION
# ============================================================================

"""
Understanding Felzenszwalb parameters:

scale (default 120.0):
  - Controls segment size
  - Range: 50-250
  - Lower (50-80) = Many small segments (detail)
  - Higher (150-200) = Fewer large segments (regions)
  - For leaf diseases: 100-150 usually good

sigma (default 0.6):
  - Gaussian blur amount
  - Range: 0.1-2.0
  - Lower = More sensitive to fine edges
  - Higher = Smoother, more uniform segments
  - 0.4-0.8 typical for most applications

min_size (default 12):
  - Minimum pixels in a segment
  - Range: 5-30
  - Prevents noise from creating tiny segments
  - 8-16 typical for 512x512 images
  - 15-25 typical for 1024x1024 images
"""


# ============================================================================
# TROUBLESHOOTING
# ============================================================================

"""
Problem: CUDA out of memory
Solution: 
  - Reduce batch processing
  - Script already handles this, but check nvidia-smi
  - If needed, process smaller image sizes first

Problem: Segmentation missing fine details
Solution:
  - Decrease scale (80-100)
  - Decrease sigma (0.3-0.4)
  - Enable edge enhancement

Problem: Over-segmentation (too many small pieces)
Solution:
  - Increase scale (150+)
  - Increase min_size (20+)

Problem: Very slow processing
Solution:
  - Check GPU with nvidia-smi
  - Set use_gpu=True in config
  - Disable colored masks
  - Increase min_size

Problem: Segment boundaries crossing disease regions
Solution:
  - Decrease scale
  - Enable edge enhancement
  - Run hyperparameter tuning for recommendations
"""


# ============================================================================
# NEXT STEPS AFTER MASK GENERATION
# ============================================================================

"""
1. Verify mask quality:
   - Check processing_summary.csv for errors
   - View *_colored.png files for visual inspection
   - Spot-check *_raw.png files

2. Choose masks for training:
   - Use *_raw.png files directly
   - These are ground truth segmentation maps

3. Implement training pipeline:
   - Create dataset loader (see example above)
   - Set up semantic segmentation model (U-Net, DeepLab, etc.)
   - Train on these masks

4. Optional: Create ensemble
   - Run multiple configurations with different scales
   - Combine masks from different scales
   - Train on ensemble predictions

5. Evaluate results:
   - Compare with manually annotated masks
   - Check IoU (Intersection over Union)
   - Adjust parameters if needed and re-run
"""


# ============================================================================
# RECOMMENDED CONFIGURATIONS BY USE CASE
# ============================================================================

"""
For general leaf disease segmentation:
  scale=120.0, sigma=0.6, min_size=12

For detailed lesion boundaries:
  scale=80.0, sigma=0.4, min_size=8

For coarse regional analysis:
  scale=150.0, sigma=0.7, min_size=16

For fast processing:
  scale=150.0, sigma=0.8, min_size=20

For maximum detail (slower):
  scale=70.0, sigma=0.4, min_size=5
"""


# ============================================================================
# COMMAND LINE QUICK REFERENCE
# ============================================================================

"""
# Run with defaults
python felzenszwalb_segmentation_gpu.py

# Run hyperparameter tuning
python felzenszwalb_hyperparameter_tuning.py

# Run batch processor (interactive menu)
python felzenszwalb_batch_processor.py

# Setup specific configuration
python felzenszwalb_batch_processor.py setup my_config 120.0 0.6 12
"""


# ============================================================================
# EXPECTED RUNTIME
# ============================================================================

"""
RTX 3090 GPU:
  - Per image (512x512): 0.5-1.0 seconds
  - Per image (1024x1024): 1.0-2.0 seconds
  - Entire dataset (~1000 images): 10-20 minutes

Hyperparameter tuning:
  - 125 combinations × 3 images × 1 second = ~2-3 minutes
  - Analysis: < 1 minute

Full batch processing of 4 configs:
  - ~40-80 minutes (1000 images)
"""


# ============================================================================
# FILES CREATED BY THIS PIPELINE
# ============================================================================

"""
Script files:
  ✓ felzenszwalb_segmentation_gpu.py         (Main segmentation)
  ✓ felzenszwalb_hyperparameter_tuning.py    (Parameter optimization)
  ✓ felzenszwalb_batch_processor.py          (Batch processing utility)
  ✓ FELZENSZWALB_PIPELINE_GUIDE.md           (Detailed documentation)
  ✓ felzenszwalb_quickstart.py               (This file)

Output directories created:
  - felzenszwalb_masks_output/               (Main masks)
  - felzenszwalb_hyperparameter_tuning/      (Tuning results)
  - felzenszwalb_batch_results/              (Batch processing)
"""


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*70)
    print("QUICK START:")
    print("="*70)
    print("1. For quick results: python felzenszwalb_segmentation_gpu.py")
    print("2. For optimized: python felzenszwalb_hyperparameter_tuning.py")
    print("3. For batch processing: python felzenszwalb_batch_processor.py")
    print("="*70 + "\n")
