# Felzenszwalb Segmentation Pipeline - Getting Started Checklist

## ✅ Pre-Flight Checklist

Before running the scripts, verify:

- [ ] CUDA toolkit installed: `nvidia-smi` (should show your GPU)
- [ ] PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] scikit-image installed: `python -c "from skimage.segmentation import felzenszwalb"`
- [ ] Dataset available at: `Lettuce_disease_datasets_split/`
- [ ] Workspace directory: `d:\gemma4\segmentation_lattuce-desease`

## 📋 Step-by-Step Getting Started

### Scenario 1: I want masks NOW (15 minutes) ⚡

```
1. [ ] Open terminal in workspace
2. [ ] Run: python felzenszwalb_segmentation_gpu.py
3. [ ] Wait for processing (10-20 minutes)
4. [ ] Check: felzenszwalb_masks_output/ directory
5. [ ] Success: You have segmentation masks!
```

**Results Location:** `felzenszwalb_masks_output/train/BACT/*_raw.png` (etc.)

### Scenario 2: I want optimal parameters (50 minutes) 🎯

```
1. [ ] Open terminal
2. [ ] Run: python felzenszwalb_hyperparameter_tuning.py
3. [ ] Wait for tuning (2-3 minutes)
4. [ ] Open: felzenszwalb_hyperparameter_tuning/recommendations.json
5. [ ] Copy best parameters (e.g., scale, sigma, min_size)
6. [ ] Edit: felzenszwalb_segmentation_gpu.py
   [ ] Find: config = FelzenszwalbConfig()
   [ ] Change: config.felz_scale = (your_scale)
   [ ] Change: config.felz_sigma = (your_sigma)
   [ ] Change: config.felz_min_size = (your_min_size)
7. [ ] Save the file
8. [ ] Run: python felzenszwalb_segmentation_gpu.py
9. [ ] Wait for processing
10. [ ] Success: Optimized masks ready!
```

**Results Location:** `felzenszwalb_masks_output/`

### Scenario 3: I want to compare multiple configurations (60-120 minutes) 📊

```
1. [ ] Open terminal
2. [ ] Run: python felzenszwalb_batch_processor.py
3. [ ] Select: 1 (Lettuce Disease Defaults)
4. [ ] Run: python felzenszwalb_segmentation_gpu.py
5. [ ] Edit config for next set
6. [ ] Repeat steps 4-5 for other configurations
7. [ ] Compare: felzenszwalb_batch_results/

Alternative: Modify scripts to accept config files and batch process
```

## 🔧 Configuration Editing Guide

### To change Felzenszwalb parameters:

**File:** `felzenszwalb_segmentation_gpu.py`

**Find this section** (around line 40-45):
```python
@dataclass
class FelzenszwalbConfig:
    """Configuration for Felzenszwalb segmentation"""
    dataset_base: str = "Lettuce_disease_datasets_split"
    output_base: str = "felzenszwalb_masks_output"
    splits: Tuple[str, ...] = ("train", "validation", "test")
    
    # Felzenszwalb hyperparameters
    felz_scale: float = 120.0      # ← CHANGE THIS
    felz_sigma: float = 0.6        # ← CHANGE THIS
    felz_min_size: int = 12        # ← CHANGE THIS
```

**Change values like:**
```python
# For fine details
felz_scale = 80.0
felz_sigma = 0.4
felz_min_size = 8

# For balanced
felz_scale = 120.0
felz_sigma = 0.6
felz_min_size = 12

# For coarse regions
felz_scale = 150.0
felz_sigma = 0.7
felz_min_size = 16
```

## 📊 Understanding Output

### Files Generated Per Image

For each image `BACT_0.jpg`, you get:

| File | Purpose | Use |
|------|---------|-----|
| `BACT_0_s120_0_sg0_6_ms12_raw.png` | **Segmentation mask** | ✅ **USE FOR TRAINING** |
| `BACT_0_s120_0_sg0_6_ms12_colored.png` | Color visualization | Visual QA |
| `BACT_0_s120_0_sg0_6_ms12_boundary.png` | Boundary map | Analysis |
| `BACT_0_s120_0_sg0_6_ms12_info.txt` | Parameters | Metadata |

### Directory Structure

```
felzenszwalb_masks_output/
├── train/
│   ├── BACT/
│   │   ├── BACT_0_*.png (multiple files)
│   │   ├── BACT_1_*.png
│   │   └── ... (all BACT images)
│   ├── DML/
│   ├── HLTY/
│   └── ... (other disease classes)
├── validation/ (same structure)
├── test/ (same structure)
└── processing_summary.csv (quality report)
```

## 🎨 Visual Verification Steps

After running segmentation:

1. **Check colored masks**
   ```
   # Open in image viewer
   felzenszwalb_masks_output/train/BACT/BACT_0_*_colored.png
   ```
   ✓ Should see different colors for different segments
   ✓ Colors should change at disease boundaries

2. **Check boundary masks**
   ```
   # View boundaries
   felzenszwalb_masks_output/train/BACT/BACT_0_*_boundary.png
   ```
   ✓ Should see white lines where segments change
   ✓ Boundaries should follow disease edges

3. **Check processing summary**
   ```
   # View CSV with a spreadsheet application
   felzenszwalb_masks_output/processing_summary.csv
   ```
   ✓ Check for any "error" status
   ✓ Verify num_segments makes sense
   ✓ Check elapsed_seconds for timing

## 🐛 Quick Troubleshooting

### GPU Not Being Used
```bash
# Check in another terminal
nvidia-smi -l 1
```
Should show Python using GPU memory. If not:
- Verify CUDA installation
- Check PyTorch CUDA availability
- May need to reinstall PyTorch with CUDA support

### Running Very Slowly
- [ ] Verify GPU usage: `nvidia-smi`
- [ ] Disable colored masks: Set `save_colored_masks = False`
- [ ] Increase min_size to reduce segments
- [ ] Check system resources: `htop` or Task Manager

### Segmentation Quality Issues

**Too many tiny segments?**
```python
felz_min_size = 20  # Increase this (was 12)
```

**Missing fine details?**
```python
felz_scale = 80.0   # Decrease this (was 120)
felz_sigma = 0.4    # Decrease this (was 0.6)
```

**Boundaries crossing disease?**
```python
felz_scale = 100.0  # Decrease this
# Run hyperparameter tuning for recommendations
```

## 📚 Documentation Quick Links

| Need | Command | File |
|------|---------|------|
| Quick overview | `cat README_FELZENSZWALB.md` | README_FELZENSZWALB.md |
| Detailed guide | `cat FELZENSZWALB_PIPELINE_GUIDE.md` | FELZENSZWALB_PIPELINE_GUIDE.md |
| Quick reference | `python felzenszwalb_quickstart.py` | felzenszwalb_quickstart.py |
| Complete summary | `python felzenszwalb_summary.py` | felzenszwalb_summary.py |
| File index | `cat FILE_INDEX.md` | FILE_INDEX.md |

## 💻 System Requirements

| Requirement | Minimum | Recommended |
|------------|---------|-------------|
| GPU | Any NVIDIA | RTX 3090 / RTX 4090 |
| RAM | 8 GB | 16 GB+ |
| Disk | 10 GB free | 50 GB free |
| Python | 3.8+ | 3.10+ |
| CUDA | 11.0+ | 12.0+ |

## ⏱️ Time Estimates

| Task | Time |
|------|------|
| Hyperparameter tuning | 2-3 minutes |
| Full dataset (1000 images) | 10-20 minutes |
| Total optimized workflow | ~50 minutes |

## 🚀 Next Steps After Mask Generation

### 1. Verify Quality (5 minutes)
- [ ] Open colored masks in image viewer
- [ ] Check for visual quality
- [ ] Scan processing_summary.csv for errors

### 2. Prepare for Training (10 minutes)
- [ ] Note directory structure
- [ ] Plan dataset splits (train/val/test)
- [ ] Prepare PyTorch DataLoader

### 3. Train Segmentation Model (days/weeks)
- [ ] Choose model architecture (U-Net, DeepLab, etc.)
- [ ] Setup training pipeline
- [ ] Train on generated masks
- [ ] Evaluate on validation set

## ✅ Success Criteria

You've successfully generated masks if:

- [ ] `felzenszwalb_masks_output/` directory exists
- [ ] Subdirectories exist: `train/`, `validation/`, `test/`
- [ ] Each disease class has a subdirectory (BACT/, DML/, etc.)
- [ ] Each class has multiple `*_raw.png` files
- [ ] `processing_summary.csv` shows "success" status
- [ ] Colored masks look reasonable visually

## 🎯 Recommended Default Starting Point

**If unsure what to do, start here:**

```bash
# Step 1: Tune parameters (2 minutes)
python felzenszwalb_hyperparameter_tuning.py

# Step 2: Check recommendations
cat felzenszwalb_hyperparameter_tuning/recommendations.json

# Step 3: Use best_uniformity parameters in main script

# Step 4: Generate masks (15 minutes)
python felzenszwalb_segmentation_gpu.py

# Step 5: Verify outputs
# - Check colored masks visually
# - Review processing_summary.csv
```

**Total Time: ~50 minutes for production-ready masks**

## 🎓 Learning Resources

1. **Quick Start** (5 min read)
   - This checklist
   - README_FELZENSZWALB.md

2. **Understanding Parameters** (10 min read)
   - See "Hyperparameter Guide" in README
   - Check recommendations.json output

3. **Advanced Usage** (30 min read)
   - Full FELZENSZWALB_PIPELINE_GUIDE.md
   - Source code docstrings
   - Tuning metrics explanation

## 📞 Common Questions

**Q: Which mask file should I use for training?**  
A: Use the `*_raw.png` files. These are the segmentation maps.

**Q: How do I change parameters?**  
A: Edit `FelzenszwalbConfig` in `felzenszwalb_segmentation_gpu.py`

**Q: Why is processing slow?**  
A: Check GPU usage with `nvidia-smi`. May need to reinstall PyTorch with CUDA.

**Q: Can I process in batches?**  
A: Yes, use `felzenszwalb_batch_processor.py` for multiple configurations.

**Q: What do the numbers in filenames mean?**  
A: `s120_0_sg0_6_ms12` = scale 120.0, sigma 0.6, min_size 12

## ✨ Tips for Success

1. **Start small** - Use just a few images first to test
2. **Visual inspection** - Always check colored masks
3. **Keep parameters** - Note what parameters worked well
4. **Automate** - Use batch processor for multiple configs
5. **Compare** - Run multiple parameter sets and compare results

---

**Ready to start? Pick your scenario above and follow the checklist! 🚀**
