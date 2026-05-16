# QUICK START GUIDE - Lettuce Disease Segmentation

## 🚀 Getting Started in 5 Minutes

### Windows (Recommended)

```batch
# 1. Navigate to project
cd d:\gemma4\segmentation_lattuce-desease

# 2. Activate CUDA environment
d:\gemma4\Scripta\activate.bat

# 3. Run startup script
startup.bat
```

✅ Everything starts automatically!

---

## 📍 Where to Go

After startup:

| Component | URL/Location | Purpose |
|-----------|--------------|---------|
| **Frontend** | `file:///d:/gemma4/segmentation_lattuce-desease/frontend/index.html` | Upload images & view results |
| **Backend API** | `http://localhost:8000` | Inference engine |
| **API Docs** | `http://localhost:8000/docs` | Interactive documentation |
| **Health Check** | `http://localhost:8000/health` | Verify server is running |

---

## 💡 What Each Tool Does

### Test Inference (Stage 9)
```bash
python scripts/stage9_test_inference.py
```
**→ Runs inference on all test images, generates metrics**

### Generate Dashboard
```bash
python scripts/dashboard_report_generator.py
```
**→ Creates HTML dashboard with visualizations**

### Compile Model to TensorRT
```bash
python scripts/tensorrt_compiler.py
```
**→ Optimizes model for 2-4x faster inference**

### Start Backend Server
```bash
python -m uvicorn backend.backend_server:app --host 0.0.0.0 --port 8000
```
**→ Starts API server for real-time inference**

### Docker (Production)
```bash
docker-compose up -d
```
**→ Starts everything in containers (Redis + Backend + Frontend)**

---

## 📊 Understanding Results

### Disease Spread Score
```
What: Percentage of image covered by disease
Range: 0-100%
Example: 45% = Half the plant is diseased
```

### Confidence Score
```
What: Model's certainty in the prediction
Range: 0-1 (0% to 100%)
Example: 0.92 = 92% confident
```

### Dominant Disease
```
What: Most prevalent disease class detected
Classes: BACT, DML, HLTY, PML, SBL, SPW, VIRL, WLBL
Example: "DML" = Downy Mildew
```

### Per-Class Distribution
```
What: Breakdown of pixels per disease
Shows: Total pixels + percentage for each class
```

---

## 🎯 Disease Classes Reference

| Code | Name | Color | Risk Level |
|------|------|-------|-----------|
| BACT | Bacterial Spot | 🟢 | Medium |
| DML | Downy Mildew | 🔴 | HIGH |
| HLTY | Healthy | 🔵 | Reference |
| PML | Powdery Mildew | 🟡 | HIGH |
| SBL | Septoria Leaf Blotch | 🟠 | Medium |
| SPW | Spotted Wilt | 🟣 | Medium |
| VIRL | Viral | 🔷 | HIGH |
| WLBL | Webbing/Leaf Blotch | 🩷 | Low |

---

## 📁 Important Locations

```
Model:
  ✓ Best: lettuce_ssl_segmentation_lab/stage6_segmentation_training/supervised_finetune/best_model.pth

Test Results:
  ✓ Masks: lettuce_ssl_segmentation_lab/stage9_test_inference/segmentation_masks/
  ✓ Report: lettuce_ssl_segmentation_lab/stage9_test_inference/inference_results.json
  ✓ Dashboard: lettuce_ssl_segmentation_lab/stage9_test_inference/dashboard_report/dashboard.html

Compiled Models:
  ✓ Location: lettuce_ssl_segmentation_lab/compiled_models/
```

---

## 🔧 API Examples

### Upload Image for Inference
```bash
curl -X POST -F "file=@image.jpg" http://localhost:8000/infer

# Returns:
# {"job_id": "abc-123", "status": "queued"}
```

### Check Inference Status
```bash
curl http://localhost:8000/result/abc-123

# Returns:
# {
#   "job_id": "abc-123",
#   "status": "completed",
#   "result": {
#     "dominant_disease": "DML",
#     "confidence_score": 0.92,
#     "disease_spread_score": 45.23,
#     ...
#   }
# }
```

### Download Segmentation Mask
```bash
curl http://localhost:8000/mask/abc-123 -o mask.png
```

### Server Statistics
```bash
curl http://localhost:8000/stats

# Returns job counts, queue size, device info
```

---

## ✅ Verification Checklist

After startup, verify:

- [ ] Redis running: `redis-cli ping` → should return `PONG`
- [ ] Backend running: `http://localhost:8000/health` → should return `{"status": "healthy"}`
- [ ] GPU available: `nvidia-smi` → should show GPU info
- [ ] Model loaded: Check backend console logs
- [ ] Frontend accessible: Open HTML file in browser

---

## 🆘 Quick Troubleshooting

### Backend won't start
```bash
# Check Python path
python --version

# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check port 8000 not in use
netstat -an | find ":8000"
```

### Redis connection failed
```bash
# Start Redis manually
redis-server --port 6379

# Or disable Redis (queue will work in-memory)
```

### GPU memory issues
```bash
# Check GPU memory
nvidia-smi

# Reduce batch size in backend_server.py (line ~150)
```

---

## 📈 Performance Tips

1. **Use FP16 precision** for 2x speed + minimal accuracy loss
2. **Batch processing** for multiple images
3. **Use compiled TensorRT** models for 2-4x speedup
4. **Monitor GPU** with: `nvidia-smi -l 1`
5. **Check logs** for bottlenecks: `docker-compose logs -f`

---

## 🎯 Next Steps

1. ✅ Start the system (`startup.bat`)
2. ✅ Open frontend (`frontend/index.html`)
3. ✅ Upload a test image
4. ✅ View results in real-time
5. ✅ Check dashboard HTML for batch analysis

---

## 📚 Full Documentation

See `README_DEPLOYMENT.md` for:
- Detailed architecture
- API reference
- Docker deployment
- Security notes
- Advanced configuration

---

## 🔗 Useful Links

- **Backend Docs**: http://localhost:8000/docs
- **Backend ReDoc**: http://localhost:8000/redoc  
- **Model**: stage6_segmentation_training/supervised_finetune/best_model.pth
- **Dashboard**: stage9_test_inference/dashboard_report/dashboard.html

---

**You're all set! 🎉 Upload an image and see the magic happen!**
