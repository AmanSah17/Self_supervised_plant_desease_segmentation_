# DRSA-Net Overview

`drsa_net/` is the newly generated disease-region superpixel-aware training stack for the lettuce disease project.

## What the package contains

- `drsa_net/config.py`
  Central configuration for dataset paths, representation settings, transformer depth, CAM propagation, training hyperparameters, checkpointing, and MLflow.
- `drsa_net/data/dataset.py`
  Builds the multi-representation dataset, matches RGB images with Felzenszwalb outputs, derives CLAHE and Watershed streams on the fly, reconstructs superpixel labels, and exposes train/validation dataloaders.
- `drsa_net/data/transforms.py`
  Applies synchronized resize, flips, rotation, and color jitter across all representation streams so branch alignment is preserved.
- `drsa_net/model/multistream_encoder.py`
  Encodes four parallel branches: RGB, CLAHE, combined Felzenszwalb masks, and Watershed. Their features are concatenated and projected to the shared embedding space.
- `drsa_net/model/superpixel_tokenizer.py`
  Converts dense feature maps into superpixel tokens with scatter-mean pooling and builds local / k-hop adjacency masks for graph-constrained attention.
- `drsa_net/model/region_aware_transformer.py`
  Runs adjacency-constrained transformer blocks with learned gating between local and semi-global attention neighborhoods, then produces disease logits from the CLS token.
- `drsa_net/model/cam_generator.py`
  Turns transformer rollout into dense CAMs, fuses it with Watershed, Felzenszwalb boundaries, and CLAHE saliency, then refines activations with region growing and graph propagation.
- `drsa_net/training/losses.py`
  Combines weakly supervised classification, CAM consistency, contrastive, compactness, and graph smoothness objectives using adaptive multi-task weighting.
- `drsa_net/training/trainer.py`
  Provides the two-view trainer, checkpointing, LR scheduling, gradient scaling, and MLflow logging.

## End-to-end flow

1. Load RGB image and its paired Felzenszwalb artifacts.
2. Generate CLAHE and Watershed branches.
3. Apply synchronized augmentation across all branches.
4. Encode all branches in parallel.
5. Pool encoder features into superpixel tokens.
6. Run region-aware transformer attention over the superpixel graph.
7. Produce disease logits and transformer attention rollout.
8. Generate and refine disease-aware CAMs.
9. Optimize with the combined DRSA loss stack.

## Entry points

- `train_drsa.py`
  Main training entrypoint for weakly supervised or self-supervised runs.
- `testing_various/smoke_tests/`
  Organized smoke and verification scripts for environment checks, dataset discovery, transforms, dataloaders, model assembly, and full-shape validation.
