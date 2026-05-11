# Lettuce SSL Segmentation Lab

This folder is the research-to-implementation workspace for the next pipeline:

1. Multi-representation input assembly
2. Healthy-only feature distribution learning
3. Anomaly localization for diseased leaves
4. CAM-guided pseudo mask generation
5. Pseudo-mask refinement
6. Transformer segmentation training

It is intentionally separate from `drsa_net/` so we can iterate on the new self-supervised and anomaly-driven segmentation workflow without destabilizing the existing training stack.

## Folder layout

- `research/`
  Research-backed implementation notes, method comparison, and source registry.
- `logs/`
  Generated manifests, summaries, and future experiment outputs.
- `data/`
  Multi-representation manifest building and parallel-channel dataset loading.
- `pipeline/`
  OOP orchestration, backbone adapters, and stage definitions.
- `utils/`
  Logging utilities and Numba helpers.

## First command

From the repository root:

```powershell
. D:\gemma4\gemma4\Scripts\Activate.ps1
python scripts/run_ssl_segmentation_lab.py
```

That command generates:

- `lettuce_ssl_segmentation_lab/logs/manifests/multirepresentation_manifest.csv`
- `lettuce_ssl_segmentation_lab/logs/manifests/manifest_summary.json`
- `lettuce_ssl_segmentation_lab/logs/research_pipeline_summary.json`

## Recommended execution order

1. Build and inspect the manifest.
2. Train or adapt a self-supervised backbone on healthy images first.
3. Fit a healthy feature model with PaDiM as the first localization baseline.
4. Add PatchCore once the healthy embedding quality is stable.
5. Fuse anomaly maps with CAMs and classical transforms to create pseudo masks.
6. Refine pseudo masks with CRF and morphology.
7. Train SegFormer first, then upgrade to Mask2Former if mask quality is good enough.
