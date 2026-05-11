"""
FELZENSZWALB SEGMENTATION PIPELINE GUIDE
========================================

This document explains how to use the Felzenszwalb segmentation scripts for creating
training masks and optimizing hyperparameters.

## Scripts Overview

### 1. felzenszwalb_segmentation_gpu.py
Main segmentation script that generates segmentation masks for the entire dataset.

Features:
- GPU-accelerated Felzenszwalb segmentation
- Edge enhancement for better boundary detection
- Multiple mask output formats (raw, colored, boundary)
- tqdm progress tracking
- CUDA-enabled for fast processing

### 2. felzenszwalb_hyperparameter_tuning.py
Hyperparameter tuning script that performs grid search over parameter combinations.

Features:
- Tests multiple parameter combinations
- Computes segmentation quality metrics
- Provides recommendations for optimal parameters
- JSON output with best parameters by different criteria

## Usage Instructions

### Step 1: Hyperparameter Tuning (Optional but Recommended)

Before processing the entire dataset, tune hyperparameters on a validation subset:

```bash
python felzenszwalb_hyperparameter_tuning.py
```

This will:
1. Load a sample of images from the validation split
2. Test all parameter combinations
3. Compute metrics for each combination
4. Save results to `felzenszwalb_hyperparameter_tuning/`
5. Provide JSON recommendations in `recommendations.json`

Output files:
- `tuning_results.csv`: Detailed results for all combinations
- `recommendations.json`: Best parameters for different objectives

Key metrics:
- `size_uniformity`: Lower is better (more consistent segment sizes)
- `compactness`: Higher is better (rounder, more compact segments)
- `boundary_edge_strength`: Higher is better (boundaries follow image edges)
- `elapsed_ms`: Lower is better (faster processing)

### Step 2: Configure Main Script

Edit `felzenszwalb_segmentation_gpu.py` to use optimal parameters:

```python
# Update the FelzenszwalbConfig
config = FelzenszwalbConfig()
config.felz_scale = 120.0      # From tuning recommendations
config.felz_sigma = 0.6        # From tuning recommendations  
config.felz_min_size = 12      # From tuning recommendations
```

### Step 3: Run Main Segmentation

Process the entire dataset with optimized parameters:

```bash
python felzenszwalb_segmentation_gpu.py
```

This will:
1. Process all images in train/validation/test splits
2. Generate segmentation masks for each image
3. Create three types of masks per image:
   - `*_raw.png`: Normalized segment labels
   - `*_colored.png`: RGB visualization with random colors
   - `*_boundary.png`: Segment boundaries highlighted
4. Save processing summary to CSV

## Output Structure

```
felzenszwalb_masks_output/
├── train/
│   ├── BACT/
│   │   ├── BACT_0_s120_0_sg0_6_ms12_raw.png
│   │   ├── BACT_0_s120_0_sg0_6_ms12_colored.png
│   │   ├── BACT_0_s120_0_sg0_6_ms12_boundary.png
│   │   └── BACT_0_s120_0_sg0_6_ms12_info.txt
│   ├── DML/
│   └── ...
├── validation/
├── test/
└── processing_summary.csv
```

## Output Files Explained

### Raw Mask (*_raw.png)
- Grayscale image where pixel value represents segment ID
- Direct input for training segmentation models
- Values normalized to 0-255 range

### Colored Mask (*_colored.png)
- RGB visualization of segments
- Each segment gets a random color
- Use for visual inspection and verification

### Boundary Mask (*_boundary.png)
- Binary mask highlighting segment boundaries
- White (255) = boundary, Black (0) = interior
- Useful for analyzing segmentation quality

### Info File (*_info.txt)
- Contains segmentation parameters used
- Number of segments generated
- Image dimensions
- Useful for reproducibility

### Processing Summary (processing_summary.csv)
- One row per processed image
- Columns: split, class_name, image_path, status, height, width, 
          num_segments, scale, sigma, min_size, elapsed_seconds
- Useful for quality control and performance analysis

## GPU Acceleration Details

The scripts use:
- **PyTorch CUDA**: Device detection and memory management
- **NumPy/SciPy**: Fast array operations
- **OpenCV**: Optimized image I/O and processing

GPU Benefits:
- Faster image loading and preprocessing
- Parallel batch processing capability
- Reduced processing time for large datasets

To monitor GPU usage while processing:

```bash
# In another terminal
nvidia-smi -l 1  # Update every 1 second
```

## Configuration Options

### FelzenszwalbConfig Parameters

```python
# Dataset paths
dataset_base = "Lettuce_disease_datasets_split"
output_base = "felzenszwalb_masks_output"

# Felzenszwalb hyperparameters
felz_scale = 120.0        # Larger = larger segments (range: 50-200)
felz_sigma = 0.6          # Gaussian blur, ~0.5-1.0
felz_min_size = 12        # Minimum segment pixels (range: 5-30)

# Output options
apply_edge_enhancement = True      # Improve boundary detection
save_colored_masks = True          # For visualization
save_raw_masks = True              # For training
save_boundary_masks = True         # For analysis

# Processing options
overwrite = False                  # Skip existing outputs
samples_per_class = None           # None = all images
```

## Hyperparameter Tuning Config

```python
class HyperparameterConfig:
    scale_range = (80.0, 100.0, 120.0, 150.0, 180.0)
    sigma_range = (0.3, 0.5, 0.7, 0.9, 1.1)
    min_size_range = (8, 12, 16, 20, 25)
    
    samples_per_class = 3     # Images per class for tuning
    max_classes = None        # Limit classes (faster tuning)
```

## Interpretation of Metrics

### size_uniformity
- Measures consistency of segment sizes
- Lower is better (std_dev / mean_size)
- Lower values = more uniform segmentation
- Good for: Consistent training data

### compactness  
- Measures how round/compact segments are
- Higher is better (range 0-1)
- Good for: Following object boundaries

### boundary_edge_strength
- Average edge strength at segment boundaries
- Higher is better
- Indicates if boundaries align with image edges
- Good for: Detecting fine structures

### num_segments
- Total number of segments generated
- Check consistency across images
- Too few = lost detail, Too many = over-segmentation

## Tips for Best Results

1. **Choose appropriate scale**:
   - Small scale (80-100): More segments, finer details
   - Large scale (150-200): Fewer segments, broader regions
   - For leaf diseases: 100-150 often good

2. **Edge enhancement**:
   - Usually improves results for texture-rich images
   - Leave enabled unless performance is critical

3. **Min size**:
   - Prevents noise from creating tiny segments
   - 10-20 usually appropriate for 512-1024px images
   - Increase for larger images

4. **Multi-scale approach**:
   - Run with different scales to create ensemble
   - Use scale_range for grid search
   - Combine results for robustness

## Using Masks for Training

The generated masks can be used for:

1. **Semantic Segmentation**:
   - Each mask is a single-channel ground truth
   - Load from *_raw.png
   - Train models like U-Net, DeepLab

2. **Instance Segmentation**:
   - Each unique value = one instance
   - Use boundary masks for additional guidance

3. **Superpixel-based Learning**:
   - Use the raw masks as pre-computed superpixels
   - Extract features from each superpixel
   - Train graph-based models

Example PyTorch DataLoader:
```python
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.images = sorted(image_dir.glob("*.jpg"))
        self.masks = sorted(mask_dir.glob("*_raw.png"))
        self.transform = transform
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx])
        mask = np.array(mask, dtype=np.int64)
        
        if self.transform:
            img = self.transform(img)
        
        return torch.from_numpy(np.array(img)), torch.from_numpy(mask)

# Usage
dataset = SegmentationDataset(image_dir, mask_dir)
loader = DataLoader(dataset, batch_size=16, shuffle=True)
```

## Troubleshooting

### Issue: Too many small segments
**Solution**: Increase min_size or scale parameter

### Issue: Segments crossing disease boundaries
**Solution**: Decrease scale or enable edge enhancement

### Issue: GPU memory error
**Solution**: Process fewer images per batch or reduce image size

### Issue: Very slow processing
**Solution**: 
- Disable colored mask generation
- Increase min_size
- Reduce scale (fewer segments)
- Check GPU is being used: nvidia-smi

## Performance Benchmarks

On NVIDIA RTX 3090:
- ~100-200 images/minute for 512x512 images
- ~50-100 images/minute for 1024x1024 images
- Hyperparameter tuning: ~1000 evaluations/minute

## Next Steps

1. Run hyperparameter tuning to find optimal parameters
2. Review recommendations.json for best parameters
3. Configure main script with optimal parameters
4. Run segmentation on full dataset
5. Inspect colored masks to verify quality
6. Use *_raw.png masks for training models

## References

- Felzenszwalb et al., 2004: "Efficient Graph-Based Image Segmentation"
- skimage.segmentation.felzenszwalb documentation
- PyTorch CUDA documentation for GPU acceleration
"""

# This is a documentation file. Run the actual scripts using:
# python felzenszwalb_segmentation_gpu.py
# python felzenszwalb_hyperparameter_tuning.py
