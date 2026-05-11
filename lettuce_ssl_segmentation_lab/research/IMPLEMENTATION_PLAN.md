# Research-Backed Implementation Plan

## Goal

Learn the distribution of **healthy lettuce leaves** from all available transform layers, then use that healthy prior to localize disease regions, generate pseudo masks, and finally train a multi-class transformer segmentation model.

## Why this structure fits your dataset

Your data already contains:

- folder-wise disease classes
- a healthy class (`HLTY`)
- classical transform outputs from Felzenszwalb
- the ability to compute CLAHE, Watershed, and edge maps

That makes your project a strong fit for a **healthy-reference anomaly localization** pipeline before full segmentation supervision exists.

## Recommended model order

### Phase A: Build a multi-representation manifest

Per image, align these channels:

- RGB
- CLAHE RGB
- Felzenszwalb raw superpixel map
- Felzenszwalb boundary map
- Felzenszwalb colored visualization
- Watershed map
- Sobel/Canny-style edge map

Why first:

- it keeps the pipeline deterministic
- it exposes missing transform coverage early
- it lets every later stage consume exactly the same aligned channels

### Phase B: Learn healthy feature distributions

Recommended first method: **frozen DINOv2 features**

Why:

- DINOv2 is strong for patch-level and dense transfer tasks
- it avoids the engineering cost of pretraining from scratch on day one
- it can be used as a feature extractor before segmentation fine-tuning

Recommended second method: **domain-adapted MAE**

Why later:

- MAE is attractive when you want lettuce-specific self-supervised pretraining
- but it requires more training and checkpoint management than DINOv2-first

### Phase C: Healthy-only anomaly localization

Recommended first method: **PaDiM**

Why first:

- simple healthy-only statistical modeling
- good localization baseline
- easier to debug than memory-bank methods

Recommended second method: **PatchCore**

Why second:

- often very strong for anomaly localization
- benefits from stable embeddings and careful memory-bank management
- more moving parts than PaDiM

### Phase D: CAM fusion

Use a classification or weakly supervised branch to create CAMs for:

- healthy vs disease
- or healthy vs each disease class

Fuse:

- anomaly heatmap
- CAM heatmap
- Felzenszwalb boundaries
- Watershed regions
- edge strength

Purpose:

- anomaly maps tell us "unusual relative to healthy"
- CAMs tell us "important for disease discrimination"
- classical transforms give boundary and region priors

### Phase E: Pseudo mask refinement

Apply:

- DenseCRF refinement
- morphological opening/closing
- connected-component filtering
- optional superpixel voting

Purpose:

- remove noisy isolated pixels
- keep lesion regions structurally consistent
- improve pseudo-mask supervision before segmentation training

### Phase F: Transformer segmentation training

Recommended order:

1. **SegFormer**
2. **Mask2Former**

Why SegFormer first:

- lighter and easier on a GTX 1650 class GPU
- very good first semantic segmentation baseline
- simpler decoder and easier experimentation

Why Mask2Former later:

- more flexible and powerful
- stronger universal segmentation design
- higher engineering and compute overhead

## Practical execution roadmap for this repo

### Milestone 1

Generate and inspect the manifest and healthy/disease coverage report.

### Milestone 2

Build a frozen DINOv2 feature extraction pipeline over the aligned multichannel inputs.

### Milestone 3

Fit PaDiM on healthy embeddings and score all disease images.

### Milestone 4

Train a disease classifier head and produce CAMs.

### Milestone 5

Fuse anomaly heatmaps + CAM + classical priors into pseudo masks.

### Milestone 6

Refine pseudo masks and export a segmentation training dataset.

### Milestone 7

Train SegFormer on pseudo labels.

### Milestone 8

If pseudo labels are strong enough, upgrade to Mask2Former or add iterative self-training.

## Important modeling guidance

- Use `HLTY` as the canonical healthy distribution source.
- Keep disease-class separation in metadata even before mask generation.
- Avoid augmentations that erase fine texture, veins, or subtle lesions.
- Do not force all transforms to be supervised targets; use many of them as feature channels or priors.
- Start frozen, then fine-tune only after healthy-vs-disease localization is trustworthy.

## What is implemented in this folder now

- OOP manifest builder for all image-layer alignment
- parallel-channel dataset loader
- staged research orchestrator
- backbone adapter registry for DINOv2 and MAE planning
- logging outputs for manifests and stage summaries

## What should be implemented next

1. DINOv2 feature extractor wrapper with batch inference
2. healthy embedding bank writer
3. PaDiM training and inference module
4. PatchCore module
5. CAM training branch
6. CRF and morphology refinement utilities
7. SegFormer training package for pseudo-labeled masks
