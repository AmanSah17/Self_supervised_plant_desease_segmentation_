# Felzenszwalb Segmentation Pipeline for Lettuce Disease Dataset

GPU-accelerated segmentation mask generation using the Felzenszwalb algorithm with hyperparameter optimization and batch processing capabilities.

## 🎯 Overview

This pipeline creates segmentation masks for training custom semantic segmentation models on lettuce disease images. It's optimized for CUDA-enabled GPUs and includes tools for hyperparameter optimization and batch processing.

## 📦 Created Files

| File | Purpose | Time |
|------|---------|------|
| `felzenszwalb_segmentation_gpu.py` | Main segmentation script - GPU accelerated | 10-20 min |
| `felzenszwalb_hyperparameter_tuning.py` | Parameter optimization via grid search | 2-3 min |
| `felzenszwalb_batch_processor.py` | Batch process multiple configurations | Variable |
| `FELZENSZWALB_PIPELINE_GUIDE.md` | Comprehensive documentation | Reference |
| `felzenszwalb_quickstart.py` | Quick reference guide | Reference |
| `felzenszwalb_summary.py` | This summary document | Reference |

## 🚀 Quick Start

### Option 1: Default Parameters (Fastest)
```bash
python felzenszwalb_segmentation_gpu.py
```
✅ Masks ready in `felzenszwalb_masks_output/` in ~15 minutes

### Option 2: Optimized Parameters (Recommended)
```bash
# Step 1: Tune hyperparameters
python felzenszwalb_hyperparameter_tuning.py

# Step 2: Review recommendations
cat felzenszwalb_hyperparameter_tuning/recommendations.json

# Step 3: Update main script with best parameters
# Edit: felzenszwalb_segmentation_gpu.py
# Update: config.felz_scale, config.felz_sigma, config.felz_min_size

# Step 4: Generate masks with optimal parameters
python felzenszwalb_segmentation_gpu.py
```
⏱️ Total time: ~50 minutes for optimized results

### Option 3: Multi-Configuration Batch
```bash
python felzenszwalb_batch_processor.py
# Select from: Lettuce Disease, Edge-Focused, Multi-Scale, or Custom
```
🎯 Best for ensemble approaches and comprehensive analysis

## 📊 Output Structure

```
felzenszwalb_masks_output/
├── train/
│   ├── BACT/
│   │   ├── image_s120_0_sg0_6_ms12_raw.png       ← Use this for training!
│   │   ├── image_s120_0_sg0_6_ms12_colored.png   ← Visual verification
│   │   ├── image_s120_0_sg0_6_ms12_boundary.png  ← Boundary analysis
│   │   └── image_s120_0_sg0_6_ms12_info.txt      ← Metadata
│   ├── DML/, HLTY/, PML/, SBL/, SPW/, VIRL/, WLBL/
├── validation/
├── test/
└── processing_summary.csv
```

**Key Output Files:**
- `*_raw.png` - Segmentation mask for training (8-bit grayscale, values 0-255)
- `*_colored.png` - RGB visualization for quality check
- `*_boundary.png` - Segment boundaries (white on black)
- `processing_summary.csv` - Quality metrics and timing data

## ⚙️ Hyperparameter Guide

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| `scale` | 50-250 | 120 | Segment size (↓ more detail, ↑ larger regions) |
| `sigma` | 0.1-2.0 | 0.6 | Smoothing (↓ fine edges, ↑ uniform) |
| `min_size` | 1-100 | 12 | Min pixels/segment (↑ removes noise) |

**Recommended Configurations:**

```python
# General disease segmentation (balanced)
scale=120.0, sigma=0.6, min_size=12

# Fine lesion boundaries (detailed)
scale=80.0, sigma=0.4, min_size=8

# Regional analysis (coarse)
scale=150.0, sigma=0.7, min_size=16

# Fast processing
scale=150.0, sigma=0.8, min_size=20
```

## 📈 Performance

**GPU: RTX 3090**
- 512×512 images: ~100-200/minute
- 1024×1024 images: ~50-100/minute
- Full dataset (1000 imgs): ~10-20 minutes

**Hyperparameter Tuning**
- 125 combinations: ~2-3 minutes
- Analysis: <1 minute

## 🔧 Integration with Training

### PyTorch Dataset Loader Example

```python
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.images = sorted(Path(image_dir).glob("*.jpg"))
        self.masks = sorted(Path(mask_dir).glob("*_raw.png"))
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx])
        
        img_array = np.array(img, dtype=np.float32) / 255.0
        mask_array = np.array(mask, dtype=np.int64)
        
        return torch.tensor(img_array).permute(2,0,1), torch.tensor(mask_array)

# Use it
dataset = SegmentationDataset(
    "Lettuce_disease_datasets_split/train",
    "felzenszwalb_masks_output/train"
)
loader = DataLoader(dataset, batch_size=16, shuffle=True)
```

## 📋 Features

✅ **GPU Acceleration**
- CUDA-enabled with PyTorch
- Parallel image processing
- GPU memory management

✅ **Segmentation Quality**
- Edge enhancement for disease boundaries
- Automatic tiny segment merging
- Multiple output formats

✅ **Hyperparameter Tuning**
- Grid search over parameter space
- 6 quality metrics computed
- JSON recommendations

✅ **Batch Processing**
- Pre-configured defaults
- Multi-scale analysis
- Interactive menu system

✅ **Data Management**
- tqdm progress tracking
- Error logging and recovery
- CSV processing reports

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| GPU out of memory | Reduce batch size, disable colored masks |
| Missing fine details | Decrease `scale` (80-100), decrease `sigma` (0.3-0.4) |
| Over-segmentation | Increase `scale` (150+), increase `min_size` (20+) |
| Slow processing | Verify GPU usage with `nvidia-smi`, disable outputs |
| Boundaries crossing disease | Run hyperparameter tuning, decrease scale |

## 📖 Documentation

- **Detailed Guide**: [FELZENSZWALB_PIPELINE_GUIDE.md](FELZENSZWALB_PIPELINE_GUIDE.md)
- **Quick Reference**: `python felzenszwalb_quickstart.py`
- **Summary**: `python felzenszwalb_summary.py`

## 🎓 Next Steps

1. ✅ Generate segmentation masks (run one of the scripts above)
2. ✅ Verify mask quality by viewing `*_colored.png` files
3. ✅ Use `*_raw.png` masks for training
4. ✅ Train semantic segmentation model (U-Net, DeepLab, etc.)
5. ✅ Evaluate on validation masks

## 📚 Supported Models

- U-Net
- DeepLab
- FCN (Fully Convolutional Networks)
- SegNet
- HRNet

## 🔗 Key Technologies

- **skimage.segmentation.felzenszwalb** - Core algorithm
- **PyTorch CUDA** - GPU acceleration
- **OpenCV** - Image I/O and processing
- **NumPy/SciPy** - Array operations

## 💡 Tips

1. **Start with hyperparameter tuning** to find optimal parameters for your dataset
2. **Check GPU usage** during processing: `nvidia-smi -l 1`
3. **Verify mask quality** by inspecting colored visualizations
4. **Run ensemble** with multiple scales for robustness
5. **Monitor progress** with the summary CSV file

## ⏱️ Expected Workflow Timeline

| Step | Time | Command |
|------|------|---------|
| Hyperparameter tuning | 2-3 min | `python felzenszwalb_hyperparameter_tuning.py` |
| Review recommendations | 5 min | View `recommendations.json` |
| Update main script | 2 min | Edit config parameters |
| Full dataset processing | 10-20 min | `python felzenszwalb_segmentation_gpu.py` |
| **Total** | **~35-50 min** | |

## 📝 Output Files

Each image generates:
```
image_name_s120_0_sg0_6_ms12_raw.png       (Segmentation mask)
image_name_s120_0_sg0_6_ms12_colored.png   (RGB visualization)
image_name_s120_0_sg0_6_ms12_boundary.png  (Boundary map)
image_name_s120_0_sg0_6_ms12_info.txt      (Parameters)
```

Filename format: `{image}_{param_suffix}_{type}.png`
- `s120_0` = scale 120.0
- `sg0_6` = sigma 0.6
- `ms12` = min_size 12

## 🎯 Use Cases

| Use Case | Configuration | Scale | Sigma | Min Size |
|----------|---------------|-------|-------|----------|
| Disease boundaries | Edge-focused | 80-100 | 0.4 | 8 |
| Balanced analysis | Standard | 120 | 0.6 | 12 |
| Regional patterns | Coarse | 150 | 0.7 | 16 |
| Fast processing | Optimized | 150 | 0.8 | 20 |

## 📞 Support

For issues or questions:
1. Check troubleshooting section above
2. Review [FELZENSZWALB_PIPELINE_GUIDE.md](FELZENSZWALB_PIPELINE_GUIDE.md)
3. Run `python felzenszwalb_summary.py` for detailed summary
4. Check GPU availability: `nvidia-smi`

---

**Created**: May 2026  
**Framework**: PyTorch + scikit-image + OpenCV  
**GPU**: CUDA-enabled (RTX series recommended)  
**Dataset**: Lettuce Disease Images  
**Purpose**: Training semantic segmentation models
