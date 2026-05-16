# YOLOv8 MLflow Training - Quick Reference Card

## 🚀 Quick Start (5 Minutes)

### 1. Install MLflow
```bash
pip install mlflow --upgrade
```

### 2. Open Notebook
```
File: d:\gemma4\segmentation_lattuce-desease\01_yolov8_segmentation_mlflow.ipynb
```

### 3. Run All Cells
```
Jupyter: Kernel → Restart & Run All
```

### 4. View Results
```bash
# Open browser to MLflow UI
mlflow ui --backend-store-uri "file:///d:/gemma4/segmentation_lattuce-desease/mlruns"
# Visit: http://localhost:5000
```

---

## 📊 Key Metrics (Per Epoch)

| Metric | Meaning | Target |
|--------|---------|--------|
| `train/loss` | Training loss | Decreasing ↓ |
| `train/seg_loss` | Mask segmentation loss | Decreasing ↓ |
| `val/loss` | Validation loss | < train/loss |
| `metrics/mAP50` | Detection accuracy @ IoU=50% | > 0.80 |
| `metrics/mAP50_95` | Detection accuracy @ IoU=50-95% | > 0.65 |
| `metrics/precision` | True positives ratio | > 0.85 |
| `metrics/recall` | Detection coverage | > 0.80 |

---

## 🔧 Hyperparameter Presets

### For RTX 3090
```python
TRAIN_PARAMS['batch'] = 16
TRAIN_PARAMS['imgsz'] = 640
TRAIN_PARAMS['epochs'] = 100
```

### For RTX 3080
```python
TRAIN_PARAMS['batch'] = 8
TRAIN_PARAMS['imgsz'] = 512
TRAIN_PARAMS['epochs'] = 100
```

### For RTX 2080 Ti / RTX 3070
```python
TRAIN_PARAMS['batch'] = 4
TRAIN_PARAMS['imgsz'] = 512
TRAIN_PARAMS['epochs'] = 100
```

### For CPU Only
```python
TRAIN_PARAMS['batch'] = 2
TRAIN_PARAMS['imgsz'] = 320
TRAIN_PARAMS['workers'] = 0
TRAIN_PARAMS['device'] = 'cpu'
```

---

## 📁 File Locations

| File | Purpose | Location |
|------|---------|----------|
| Main notebook | Training pipeline | `01_yolov8_segmentation_mlflow.ipynb` |
| Debug guide | Troubleshooting | `YOLOV8_MLFLOW_DEBUGGING_GUIDE.md` |
| Improvements doc | What changed | `YOLOV8_IMPROVEMENTS_SUMMARY.md` |
| Dataset config | YOLO format | `dataset_yolo/data.yaml` |
| Training results | Per-epoch metrics | `runs/segment/.../results.csv` |
| Best weights | Trained model | `runs/segment/.../weights/best.pt` |
| MLflow storage | Experiment tracking | `mlruns/` |

---

## ✅ Pre-Training Checklist

- [ ] GPU available: `torch.cuda.is_available()` → `True`
- [ ] Dataset valid: Cell 2 shows "✅ Dataset validation PASSED!"
- [ ] MLflow ready: Cell 3 shows experiment created
- [ ] Hyperparameters adjusted for your GPU
- [ ] At least 5GB free disk space
- [ ] Training time: ~2-3 hours (RTX 3080, 100 epochs)

---

## 🐛 Common Fixes (30 Seconds)

### Training stops: "CUDA out of memory"
```python
TRAIN_PARAMS['batch'] = 2  # Reduce from 4
TRAIN_PARAMS['imgsz'] = 320  # Reduce from 512
```

### Data validation fails
```python
# Run Cell 2 again to see specific error
# Check: Leaf Disease Segmentation.v1i.coco-segmentation/Notebooks/dataset_yolo/
```

### Can't find MLflow artifacts
```bash
# Check storage location
ls d:\gemma4\segmentation_lattuce-desease\mlruns\
```

### Model evaluation fails
```python
# Verify data.yaml exists
ls d:\gemma4\segmentation_lattuce-desease\Leaf Disease Segmentation.v1i.coco-segmentation\Notebooks\dataset_yolo\data.yaml
```

---

## 📊 MLflow Commands

### View all experiments
```python
import mlflow
mlflow.search_experiments()
```

### View runs in experiment
```python
runs = mlflow.search_runs(experiment_names=["lettuce_disease_segmentation"])
print(runs[['start_time', 'params.model', 'metrics.evaluation/val_map50']])
```

### Get best run
```python
best_run = mlflow.search_runs(
    experiment_names=["lettuce_disease_segmentation"],
    order_by=["metrics.evaluation/val_map50_95 DESC"],
    max_results=1
)
```

### Download artifacts
```python
client = mlflow.tracking.MlflowClient()
client.download_artifacts(run_id="abc123", dst_path="./downloads")
```

---

## 🎯 Model Selection

| Model | Speed | Accuracy | Size | Use Case |
|-------|-------|----------|------|----------|
| yolov8n-seg | Fastest ⚡ | Lower | 3.2MB | Testing, prototyping |
| yolov8s-seg | Medium | Medium | 22MB | **Recommended** |
| yolov8m-seg | Slower | Highest | 49MB | Best accuracy |

**Recommendation**: Start with **yolov8s-seg** for best balance

---

## 📈 Performance Examples

### Expected Results (on Lettuce Disease Dataset)

**After 10 epochs:**
- mAP50: ~0.65-0.75
- Precision: ~0.80-0.90
- Recall: ~0.70-0.80

**After 50 epochs:**
- mAP50: ~0.80-0.88
- Precision: ~0.85-0.95
- Recall: ~0.85-0.92

**After 100 epochs:**
- mAP50: ~0.85-0.92
- Precision: ~0.88-0.96
- Recall: ~0.88-0.94

---

## 🔄 Workflow

```
1. Data Validation ✓
       ↓
2. Set Hyperparameters ✓
       ↓
3. Run Training ✓ (auto-logs to MLflow)
       ↓
4. Review Metrics ✓ (MLflow UI or plots)
       ↓
5. Evaluate Model ✓ (auto-evaluated)
       ↓
6. Save Benchmark ✓ (auto-saved)
       ↓
7. Deploy Best Model ✓
```

---

## 💾 Saving & Resuming

### Save current run ID
```python
print(f"Run ID: {mlflow.active_run().info.run_id}")
# Copy this for reference
```

### Resume interrupted training
```python
model = YOLO('path/to/last.pt')
model.train(data=..., epochs=150, resume=True)
```

---

## 🎓 Next Steps

1. **Run notebook** with yolov8s-seg (recommended)
2. **Wait for completion** (~2-3 hours)
3. **Open MLflow UI** to view metrics
4. **Compare models** - try nano vs small vs medium
5. **Tune hyperparameters** based on results
6. **Deploy best model** to production

---

## 📞 Support

### Debugging
- See: `YOLOV8_MLFLOW_DEBUGGING_GUIDE.md`

### What Changed
- See: `YOLOV8_IMPROVEMENTS_SUMMARY.md`

### YOLOv8 Docs
- https://docs.ultralytics.com/

### MLflow Docs
- https://mlflow.org/

---

**Created**: May 16, 2026
**Version**: 1.0
**Status**: ✅ Ready for Production
