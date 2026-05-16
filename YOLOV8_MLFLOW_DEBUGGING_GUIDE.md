# YOLOv8 Training Pipeline - Debugging & MLflow Integration Guide

## 📋 Overview

This guide documents the comprehensive debugging improvements made to the YOLOv8 segmentation training pipeline, now featuring:
- ✅ **MLflow Experiment Tracking** - Track all experiments with automatic metric logging
- ✅ **Per-Epoch Loss Metrics** - Capture training/validation loss at each epoch
- ✅ **Multi-Model Benchmarking** - Train and compare nano/small/medium models
- ✅ **Robust Error Handling** - Comprehensive try-catch blocks and validation
- ✅ **Data Validation** - Pre-training dataset integrity checks
- ✅ **GPU Monitoring** - Automatic GPU availability detection and logging
- ✅ **Results Visualization** - Training metrics plots and comparison tables

---

## 🔍 Issues Identified in Original Pipeline

### 1. **No Experiment Tracking**
   - **Issue**: No record of training runs, hyperparameters, or results
   - **Fix**: Integrated MLflow with automatic parameter logging

### 2. **Missing Per-Epoch Metrics**
   - **Issue**: Unable to track loss evolution or identify training issues
   - **Fix**: Custom `MetricsExtractor` class parses results.csv and logs to MLflow

### 3. **No Data Validation**
   - **Issue**: Training could fail mid-process due to missing/corrupted data
   - **Fix**: Added `DatasetValidator` class with comprehensive checks

### 4. **Hardcoded Paths**
   - **Issue**: Not portable across systems or project structures
   - **Fix**: Implemented Path-based configuration management

### 5. **Limited Error Handling**
   - **Issue**: Cryptic errors without clear diagnostics
   - **Fix**: Try-catch blocks with detailed logging throughout

### 6. **No Model Benchmarking**
   - **Issue**: Unable to compare performance across model sizes
   - **Fix**: Added `BenchmarkManager` for multi-model comparison

---

## 📂 File Structure

```
d:\gemma4\segmentation_lattuce-desease\
├── 01_yolov8_segmentation_mlflow.ipynb          [NEW] Main training notebook
├── YOLOV8_MLFLOW_DEBUGGING_GUIDE.md             [NEW] This guide
├── Leaf Disease Segmentation.v1i.coco-segmentation/
│   └── Notebooks/
│       ├── 01_ultalytics_yolov8.ipynb           [OLD] Original notebook
│       ├── dataset_yolo/
│       │   ├── data.yaml                         YOLO dataset config
│       │   ├── images/
│       │   │   ├── train/
│       │   │   ├── val/
│       │   │   └── test/
│       │   └── labels/
│       └── runs/
│           └── segment/
│               └── plant_disease_segmentation/
└── mlruns/                                       MLflow artifacts storage
    └── [experiment_id]/
        └── [run_id]/
```

---

## 🚀 Quick Start

### Step 1: Install Required Dependencies
```bash
pip install mlflow>=2.0.0
pip install torch torchvision ultralytics
pip install pandas numpy matplotlib seaborn
```

### Step 2: Run the Notebook
```python
# Open notebook: 01_yolov8_segmentation_mlflow.ipynb
# Run cells sequentially (automatic execution shows progress)
```

### Step 3: View MLflow Results
```bash
# Start MLflow UI (optional, for web-based dashboard)
mlflow ui --backend-store-uri "file:///d:/gemma4/segmentation_lattuce-desease/mlruns"
# Open: http://localhost:5000
```

---

## 📊 MLflow Integration Details

### Tracked Parameters
- `model`: Model type (yolov8n-seg, yolov8s-seg, etc.)
- `epochs`: Total training epochs
- `batch_size`: Batch size
- `image_size`: Input image resolution
- `optimizer`: Optimization algorithm
- `learning_rate`: Initial learning rate
- `device`: GPU/CPU

### Logged Metrics (Per-Epoch)
```
epoch/train/box_loss        - Bounding box loss
epoch/train/cls_loss        - Classification loss
epoch/train/dfl_loss        - Distribution focal loss
epoch/train/seg_loss        - Segmentation mask loss
epoch/val/box_loss          - Validation box loss
epoch/val/cls_loss          - Validation classification loss
epoch/val/seg_loss          - Validation segmentation loss
epoch/metrics/precision     - Precision (IoU threshold 0.5)
epoch/metrics/recall        - Recall (IoU threshold 0.5)
epoch/metrics/mAP50         - Mean Average Precision @ IoU=0.50
epoch/metrics/mAP50_95      - Mean Average Precision @ IoU=0.50:0.95
evaluation/val_map50        - Final validation mAP50
evaluation/val_map50_95     - Final validation mAP50-95
evaluation/val_precision    - Final validation precision
evaluation/val_recall       - Final validation recall
```

### Saved Artifacts
```
runs/segment/plant_disease_segmentation/[run_name]/
├── weights/
│   ├── best.pt              ← Best model (lowest val loss)
│   └── last.pt              ← Last epoch model
├── results.csv              ← Per-epoch metrics table
├── training_metrics_plot.png ← Training curves visualization
└── benchmark_config.json    ← Model metadata
```

---

## 🔧 Key Classes & Functions

### `DatasetValidator`
Validates dataset structure before training:
```python
validator = DatasetValidator(dataset_root, data_yaml_path)
is_valid = validator.validate()
```

**Checks:**
- ✅ data.yaml exists and is valid YAML
- ✅ Train/val directory structure
- ✅ Image and label files present
- ✅ Class definitions in YAML

### `MLflowCallback`
Custom callback for per-epoch logging:
```python
class MLflowCallback:
    def on_train_epoch_end(self, trainer):
        # Extracts metrics from trainer and logs to MLflow
        # Runs every 5 epochs to reduce API calls
```

### `MetricsExtractor`
Parses training results CSV and logs to MLflow:
```python
extractor = MetricsExtractor(results_csv_path)
metrics_df = extractor.load_and_parse()
extractor.log_to_mlflow()
```

**Output:** DataFrame with all per-epoch metrics

### `ModelEvaluator`
Evaluates trained models on validation sets:
```python
evaluator = ModelEvaluator(model_path, data_yaml)
evaluator.load_model()
eval_metrics = evaluator.evaluate()
evaluator.log_evaluation_to_mlflow()
```

### `BenchmarkManager`
Manages and compares multiple trained models:
```python
manager = BenchmarkManager(experiment_name)
benchmark = manager.create_benchmark(model_path, metadata)
comparison_df = manager.compare_benchmarks()
```

---

## 📈 Training Hyperparameters

Default configuration in `TRAIN_PARAMS`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| epochs | 100 | Maximum training epochs |
| imgsz | 512 | Input image size |
| batch | 4 | Batch size per iteration |
| patience | 20 | Early stopping patience |
| optimizer | SGD | Optimization algorithm |
| lr0 | 0.01 | Initial learning rate |
| device | 0 | GPU device ID (0 for first GPU) |
| amp | True | Automatic Mixed Precision |
| cache | False | Cache images in RAM |

**Adjust these based on your GPU memory:**
- **RTX 3090**: batch=16, imgsz=640
- **RTX 3080**: batch=8, imgsz=512
- **RTX 2080Ti**: batch=4, imgsz=512
- **CPU**: batch=2, imgsz=320, workers=0

---

## 🐛 Debugging Workflow

### Issue: Training starts but stops abruptly

1. **Check GPU memory**
   ```python
   import torch
   print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
   print(f"Available: {torch.cuda.memory_available(0) / 1e9:.1f}GB")
   ```

2. **Check dataset validation**
   - Look for errors in cell 2 (Dataset Validation)
   - Ensure images exist in train/val directories

3. **Check MLflow logs**
   - Review MLflow artifacts for partial results
   - Check training_metrics_plot.png for convergence

### Issue: Poor model performance

1. **Review per-epoch metrics**
   ```python
   print(metrics_df[['train/loss', 'val/loss']])
   ```

2. **Check for overfitting**
   - If val/loss >> train/loss → increase regularization
   - If flat loss curve → increase learning rate

3. **Verify data quality**
   - Check image counts in each split
   - Verify annotation format

### Issue: CUDA out of memory

1. **Reduce batch size**
   ```python
   TRAIN_PARAMS['batch'] = 2  # Reduce from 4
   ```

2. **Reduce image size**
   ```python
   TRAIN_PARAMS['imgsz'] = 320  # Reduce from 512
   ```

3. **Enable gradient accumulation**
   ```python
   TRAIN_PARAMS['accumulate'] = 4
   ```

---

## 📊 Analyzing Results

### View Training Curves
```python
# Already generated in notebook
# Check: runs/segment/plant_disease_segmentation/[run_name]/training_metrics_plot.png
```

### Extract Best Model Metrics
```python
best_run = mlflow.search_runs(
    experiment_names=["lettuce_disease_segmentation"],
    order_by=["metrics.evaluation/val_map50_95 DESC"],
    max_results=1
)
print(best_run[['start_time', 'metrics.evaluation/val_map50_95']])
```

### Compare Multiple Models
```python
runs = mlflow.search_runs(
    experiment_names=["lettuce_disease_segmentation"]
)
comparison = runs[['params.model', 'metrics.evaluation/val_map50', 'metrics.evaluation/val_map50_95']]
print(comparison.sort_values('metrics.evaluation/val_map50_95', ascending=False))
```

---

## 🎯 Multi-Model Benchmarking

To train multiple models for comparison:

```python
# Uncomment in final cell of notebook:
benchmark_models = ['yolov8n-seg', 'yolov8s-seg', 'yolov8m-seg']
results = train_multiple_models(benchmark_models, TRAIN_PARAMS)
```

**Expected Results:**
- yolov8n: Fastest, smallest (~3.2MB), lower accuracy
- yolov8s: Balanced, medium (~22MB)
- yolov8m: Most accurate, largest (~49MB), slowest

---

## 🔄 Resuming Training

To resume an interrupted training:

```python
# Load last checkpoint
model = YOLO('path/to/last.pt')

# Resume training
TRAIN_PARAMS['resume'] = True
TRAIN_PARAMS['patience'] = 50  # Allow more epochs
model.train(**TRAIN_PARAMS)
```

---

## 📝 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| FileNotFoundError: data.yaml | Verify path in `TRAIN_PARAMS['data']` |
| CUDA out of memory | Reduce `batch` size or `imgsz` |
| No improvements in loss | Check data quality, increase `lr0` |
| Model evaluation fails | Verify data.yaml structure and paths |
| MLflow not logging metrics | Check experiment creation in Section 3 |
| Training very slow | Enable AMP: `TRAIN_PARAMS['amp'] = True` |

---

## 📚 Additional Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [PyTorch CUDA Documentation](https://pytorch.org/docs/stable/cuda.html)

---

## ✅ Verification Checklist

Before starting production training:

- [ ] Dataset validation passes (Cell 2)
- [ ] GPU/CUDA available or CPU mode confirmed (Cell 4)
- [ ] MLflow experiment created successfully (Cell 3)
- [ ] Training hyperparameters reviewed and adjusted for your GPU
- [ ] Backup of best model location noted
- [ ] MLflow tracking URI accessible
- [ ] Sufficient disk space for artifacts (estimate 500MB per model)

---

## 🎓 Next Steps

1. **Train initial model**: Run notebook with yolov8n-seg (fastest)
2. **Review results**: Check metrics in MLflow UI
3. **Compare models**: Train yolov8s-seg and yolov8m-seg
4. **Fine-tune**: Adjust hyperparameters based on performance
5. **Deploy**: Use best model in production pipeline

---

**Created**: 2026-05-16
**Status**: ✅ Production Ready
**Version**: 1.0
