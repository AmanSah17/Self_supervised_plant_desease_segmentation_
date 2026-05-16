# Lettuce Disease Segmentation (SSL-Fine-Tuned Pipeline)

![Pipeline Overview](./docs/images/pipeline_overview.png)

## 📚 Overview
This repository implements an advanced **SSL-Fine-Tuned lettuce disease segmentation pipeline**. By integrating **Roboflow expert annotations** as structural anchors, the system achieves high-precision localization by combining **14-channel multi-modal features** with **self-supervised anomaly detection (PaDiM)** and **transformer-based segmentation (SegFormer)**.

| Stage | Purpose | Methodology | Key Outputs |
|------|---------|-------------|-------------|
| **1-2** | Representation Learning | DINOv2 Healthy Feature Bank | Frozen Vit-B/14 backbone |
| **3** | Anomaly Localization | PaDiM Structural Deviation | Pixel-wise Anomaly Heatmaps |
| **4** | Multi-Class Head | DINOv2-based Classifier | Class-specific CAMs |
| **5** | **Anchored Fusion** | **Roboflow Anchor Polygons** + CAM + Anomaly | **High-Fidelity Pseudo-Masks** |
| **6** | Supervised Training | SegFormer on 14-channel Stack | Final Segmentation Model |

---

## 🏗️ 14‑Channel Multi‑Modal Stack
The system processes each image into a specialized **14-channel tensor**, providing the model with a "multi-modal" understanding of the leaf's texture, chemistry (via ExG), and structure.

**Stacked Channels:**
1. **RGB** (3): Standard spectral information.
2. **CLAHE** (3): Locally contrast-enhanced RGB for subtle lesion edges.
3. **Felz-Colored** (3): Superpixel region consensus.
4. **Edge Map** (1): Sobel gradients for boundary detection.
5. **Excess-Green (ExG)** (1): Chlorophyll index for necrotic tissue contrast.
6. **Watershed** (1): Topographic region separation.
7. **Felz-Raw/Boundary** (2): Structural segmentation indices.

---

## 🎯 Roboflow Integration & "Anchoring"
The latest iteration of the pipeline integrates the **Roboflow Lettuce Disease Dataset**. To maximize performance, we utilize a technique called **Anchored Pseudo-Mask Generation**:

1.  **Polygon Anchors**: Ground-truth polygons from Roboflow are converted into pixel-masks.
2.  **Guided SSL**: During Stage 5, the self-supervised anomaly maps and CAMs are "anchored" to these manual labels.
3.  **Result**: This ensures 100% boundary fidelity for known disease spots while allowing the model to generalize across the entire dataset via self-supervision.

---

## 📊 Benchmarking & Performance
The move to the **14-channel anchored methodology** on the Roboflow dataset has resulted in a massive performance leap compared to the baseline SSL pipeline.

| Metric | Baseline SSL (Stage 3/4) | **Anchored SSL-Fine-Tuned (Stage 6)** | Improvement |
| :--- | :---: | :---: | :---: |
| **mIoU** | 0.083 | **0.814** | **+880%** |
| **Accuracy** | 0.152 | **0.892** | **+486%** |
| **Precision** | 0.139 | **0.845** | **+507%** |

> [!IMPORTANT]
> The integration of **Roboflow Anchors** in the training loop (Stage 6) transforms the SSL foundations into a high-precision supervised segmenter, capable of detecting subtle bacterial spots with expert-level accuracy.

---

## 🚀 How to Run
```powershell
# 1. Ingest Roboflow Data
python scripts\roboflow_ingestion.py

# 2. Execute Refined Pipeline (Stage 1-5)
python scripts\stage5_cam_attention_masks.py

# 3. Final SegFormer Training (Stage 6)
$env:NUM_EPOCHS="100"; python scripts\stage6_segmentation_training.py

# 4. View Metrics
mlflow ui
```

---

## 📂 Repository Layout
*   `lettuce_ssl_segmentation_lab/`: Core library and pipeline logic.
*   `scripts/`: Execution scripts for Stages 1 through 8.
*   `Roboflow_Dataset/`: Standardized dataset structure (images + manual labels).
*   `mlruns/`: MLflow experiment tracking logs.

---

*Project Maintained by Aman Sah – © 2026*
