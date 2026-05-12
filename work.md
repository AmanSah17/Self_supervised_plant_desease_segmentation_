# Lettuce Disease Segmentation: Self-Supervised Learning Pipeline

This repository contains an end-to-end research pipeline for the high-fidelity segmentation of lettuce diseases and weeds using a self-supervised learning (SSL) approach combined with CAM-aware attention fusion.

## 🚀 Research Pipeline Overview

The project is structured into 6 distinct stages, moving from unsupervised feature learning to supervised semantic segmentation.

### Stage 1 & 2: Representation Learning
*   **Healthy Learner**: Extracts high-dimensional features from "healthy" lettuce samples using a DINOv2 backbone.
*   **Feature Statistics**: Computes global statistics of healthy tissue to establish a baseline for "normal" leaf appearance.

### Stage 3: Anomaly Localization (Structural Evidence)
*   **Unsupervised Detection**: Identifies regions that deviate from the healthy feature distribution.
*   **Output**: High-resolution Anomaly Maps that highlight physical irregularities (lesions, necrosis, etc.) without knowing the specific disease class.

### Stage 4: Multi-Class Head & Semantic Evidence
*   **Classifier Training**: A multi-class head is trained on top of frozen DINOv2 features using image-level labels.
*   **CAM Generation**: Utilizes Class Activation Maps (CAM) to identify *which* disease is present and *where* the model's semantic evidence is strongest.

### Stage 5: CAM-Aware Attention Fusion
*   **Core Innovation**: Fuses the **Structural Anomaly Map** (from Stage 3) with the **Semantic CAM** (from Stage 4).
*   **Superpixel Consensus**: Uses Felzenszwalb superpixels to "snap" the labels to physical leaf boundaries, creating high-fidelity multi-class pseudo-masks.
*   **Refinement**: Filters out background noise using an HSV-based greenery mask (Vegetation Index).

### Stage 6: Multi-Task Segmentation Training
*   **Architecture**: Supports **SegFormer (MiT-B3)** and **DeepLabV3+**.
*   **Multi-Channel Stack**: Instead of just RGB, the model learns from a **14-channel representation stack**:
    *   RGB (3 channels)
    *   CLAHE Enhanced RGB (3 channels)
    *   Excess Green Index (1 channel)
    *   Canny Edge Maps (1 channel)
    *   Felzenszwalb Superpixels (Raw, Boundary, Colored)
    *   Watershed Segments (1 channel)
*   **Tracking**: Full integration with **MLflow** for mIoU, Precision, and Recall tracking.

## 🛠️ Technical Implementation Details

*   **Adaptive Models**: Custom model factory that dynamically adapts pretrained Transformer/CNN backbones to handle arbitrary input channel counts (e.g., 14-channel input).
*   **Numba Optimization**: High-performance vegetation indexing and superpixel remapping using Numba JIT.
*   **Modular Design**: Decoupled modules for losses, metrics, datasets, and trainers.

## 📈 Current Status
*   **Dataset Size**: 13,118 images with generated pseudo-masks.
*   **Optimization**: Transitioning to offline pre-calculation of the 14-channel stack to eliminate CPU bottlenecks during training.

---
*Created by Antigravity AI Assistant.*
