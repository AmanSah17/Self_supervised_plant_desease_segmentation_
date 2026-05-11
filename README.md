# Lettuce Disease SSL Segmentation Pipeline

This repository implements a 6-stage Self-Supervised Learning (SSL) pipeline for high-fidelity segmentation and classification of lettuce leaf diseases and weeds.

## 🚀 Research Goal
To develop a robust segmentation system that can learn from unlabelled or weakly labelled data (folder-level labels) using foundational vision models (DINOv2) and statistical anomaly detection.

## 🏗️ 6-Stage Architecture

### Stage 1: Multi-Representation Indexing
Builds a comprehensive manifest of the dataset, indexing RGB channels, segments (Felzenszwalb), and class metadata.

### Stage 2: Healthy-Only Representation Learning
Uses DINOv2 to extract deep features from "healthy" leaves to establish a baseline of "normal" leaf texture and structure.

### Stage 3: Anomaly Localization (PaDiM)
Models the distribution of healthy leaf patches using multivariate Gaussians. Diseased images are then scored against this distribution to produce **Anomaly Heatmaps**.

### Stage 4: CAM Fusion & Pseudo-Mask Generation
Trains a multi-class linear probe on top of frozen DINOv2 features to generate **Class Activation Maps (CAM)**. These are fused with anomaly maps and superpixel boundaries to create multi-class training labels.

### Stage 5: Mask Refinement (DenseCRF)
(In Progress) Uses Conditional Random Fields to refine the pseudo-mask boundaries using high-resolution edge and color information from the original images.

### Stage 6: Multi-Task Segmentation Training
(Pending) Final training of a supervised model (e.g., SegFormer) using the generated pseudo-masks as ground truth.

## 🛠️ Technology Stack
- **Backbone**: DINOv2 (ViT-B/14)
- **Anomaly Detection**: PaDiM (Patch Distribution Modeling)
- **Segmentation**: Felzenszwalb Superpixels, DenseCRF
- **Deep Learning**: PyTorch, TorchVision
- **Data Handling**: Pandas, OpenCV, Albumentations

## 📁 Project Structure
- `lettuce_ssl_segmentation_lab/`: Core library package.
    - `pipeline/`: Implementation of the 6 stages.
    - `data/`: Dataset loaders and manifest builders.
    - `utils/`: Feature extractors and background segmenters.
- `scripts/`: Execution scripts for each stage.
- `configs/`: YAML configuration files.

## 📈 Current Progress
- [x] Stage 1: Indexing Complete
- [x] Stage 2: Healthy Learning Complete
- [x] Stage 3: Anomaly Localization Complete (PaDiM implemented)
- [x] Stage 4: Multi-Class Pseudo Masks (8-class support implemented)
- [ ] Stage 5: DenseCRF Refinement
- [ ] Stage 6: SegFormer Training
