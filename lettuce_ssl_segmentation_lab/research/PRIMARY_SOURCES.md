# Primary Sources

These are the main primary sources used to shape the implementation plan.

## Self-supervised backbones

- DINOv2 paper: https://arxiv.org/abs/2304.07193
- DINOv2 official repository: https://github.com/facebookresearch/dinov2
- DINOv2 semantic segmentation notebook: https://github.com/facebookresearch/dinov2/blob/main/notebooks/semantic_segmentation.ipynb
- MAE paper: https://arxiv.org/abs/2111.06377
- MAE official repository: https://github.com/facebookresearch/mae

## Anomaly localization

- PaDiM paper: https://arxiv.org/abs/2011.08785
- PatchCore paper: https://arxiv.org/abs/2106.08265
- PatchCore official repository: https://github.com/amazon-science/patchcore-inspection
- Anomalib repository with Padim and Patchcore support: https://github.com/open-edge-platform/anomalib

## Segmentation

- SegFormer paper: https://arxiv.org/abs/2105.15203
- SegFormer official repository: https://github.com/NVlabs/SegFormer
- Mask2Former CVPR paper: https://openaccess.thecvf.com/content/CVPR2022/html/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.html
- Mask2Former official repository: https://github.com/facebookresearch/mask2former

## Refinement and weak localization

- Fully connected CRF paper: https://papers.nips.cc/paper/4296-efficient
- CREAM weakly supervised activation mapping: https://openaccess.thecvf.com/content/CVPR2022/html/Xu_CREAM_Weakly_Supervised_Object_Localization_via_Class_RE-Activation_Mapping_CVPR_2022_paper.html
- Extracting CAMs from non-discriminative features: https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Extracting_Class_Activation_Maps_From_Non-Discriminative_Features_As_Well_CVPR_2023_paper.html

## Domain-relevant agriculture references

- Weakly supervised activation mapping for citrus pest localization: https://arxiv.org/abs/2004.11252
- Self-supervised learning of plant image representations: https://arxiv.org/abs/2604.27538

## How the source code is typically executed

- DINOv2:
  The official repo provides pretrained backbones and a semantic segmentation notebook. This is best used here as a frozen feature extractor before fine-tuning.
- MAE:
  The official repo is centered on pretraining and finetuning recipes, which makes it better as a second-phase domain-adaptation option.
- PatchCore:
  The official repo uses a memory-bank workflow and CLI-style scripts for training and evaluation.
- Anomalib:
  The library exposes `Padim` and `Patchcore` through a consistent training and prediction interface.
- SegFormer:
  The official implementation is tied to MMSegmentation and is well suited for semantic segmentation fine-tuning once pseudo masks are available.
- Mask2Former:
  The official implementation is built on Detectron2 and is the stronger but heavier second-stage segmentation option.
