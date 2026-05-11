from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from lettuce_ssl_segmentation_lab.config import LabConfig
from lettuce_ssl_segmentation_lab.data.manifest import (
    ManifestSummary,
    MultiRepresentationManifestBuilder,
)
from lettuce_ssl_segmentation_lab.pipeline.backbones import (
    DINOv2BackboneAdapter,
    MAEBackboneAdapter,
)
from lettuce_ssl_segmentation_lab.utils.logging_utils import ExperimentLogger


@dataclass
class StageRecommendation:
    stage_name: str
    goal: str
    recommended_method: str
    outputs: list[str]
    implementation_notes: list[str]


class SegmentationResearchOrchestrator:
    def __init__(self, config: LabConfig):
        self.config = config.resolve()
        self.logger = ExperimentLogger(self.config.logs_dir)
        self.manifest_builder = MultiRepresentationManifestBuilder(self.config)

    def build_manifest(self) -> tuple[pd.DataFrame, ManifestSummary]:
        manifest_df = self.manifest_builder.build()
        summary = self.manifest_builder.summarize(manifest_df)
        self.config.manifests_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.config.manifests_dir / "multirepresentation_manifest.csv"
        manifest_df.to_csv(manifest_path, index=False)
        coverage_df = (
            manifest_df.groupby(["split", "class_name", "label_kind"], dropna=False)
            .agg(
                samples=("image_path", "count"),
                felz_raw_coverage=("felz_raw_path", lambda s: int(s.notna().sum())),
                felz_boundary_coverage=("felz_boundary_path", lambda s: int(s.notna().sum())),
                felz_colored_coverage=("felz_colored_path", lambda s: int(s.notna().sum())),
                avg_variants=("num_felz_variants", "mean"),
            )
            .reset_index()
        )
        coverage_df.to_csv(self.config.manifests_dir / "class_split_coverage.csv", index=False)
        self.logger.write_json("manifests/manifest_summary.json", summary)
        return manifest_df, summary

    def research_plan(self) -> list[StageRecommendation]:
        return [
            StageRecommendation(
                stage_name="Stage 1: Multi-representation indexing",
                goal="Align every source image with all classical transform layers.",
                recommended_method="Use a manifest-driven dataset so every image can expose RGB, CLAHE, Felzenszwalb raw/boundary/colored, Watershed, and edge channels.",
                outputs=["CSV manifest", "coverage summary", "missing-channel report"],
                implementation_notes=[
                    "Keep all channels parallel, not sequentially merged later.",
                    "Choose one preferred Felzenszwalb signature now and keep all alternates in metadata.",
                ],
            ),
            StageRecommendation(
                stage_name="Stage 2: Healthy-only representation learning",
                goal="Learn the healthy leaf texture and structure distribution before disease localization.",
                recommended_method="Start with frozen DINOv2 features. Keep MAE as phase-2 domain adaptation if healthy-vs-disease separation is weak.",
                outputs=["healthy feature embeddings", "healthy centroid/statistics bank"],
                implementation_notes=[
                    "Use only HLTY images for healthy distribution fitting.",
                    "Avoid aggressive augmentations that destroy subtle lesion or vein structure.",
                ],
            ),
            StageRecommendation(
                stage_name="Stage 3: Anomaly localization",
                goal="Convert diseased samples into heatmaps relative to healthy structure.",
                recommended_method="PaDiM first, then PatchCore.",
                outputs=["per-image anomaly score", "pixel heatmap", "per-class anomaly summaries"],
                implementation_notes=[
                    "PaDiM is simpler and statistically grounded for a healthy-only baseline.",
                    "PatchCore is strong once feature quality is stable and memory-bank management is ready.",
                ],
            ),
            StageRecommendation(
                stage_name="Stage 4: CAM fusion and pseudo masks",
                goal="Fuse anomaly evidence with class-aware activations and classical priors.",
                recommended_method="Combine anomaly heatmaps, transformer CAMs, Felzenszwalb boundaries, Watershed regions, and edge maps.",
                outputs=["raw heatmap", "fused CAM", "pseudo mask"],
                implementation_notes=[
                    "Use healthy-vs-disease classification heads to produce CAM evidence.",
                    "Use superpixel and watershed regions to expand lesion evidence beyond only the most discriminative pixels.",
                ],
            ),
            StageRecommendation(
                stage_name="Stage 5: Refinement",
                goal="Sharpen pseudo masks into segmentation-quality supervision.",
                recommended_method="DenseCRF plus morphological cleanup.",
                outputs=["refined pseudo masks", "mask confidence score"],
                implementation_notes=[
                    "Keep connected lesion regions while removing isolated noise.",
                    "Log confidence and discard weak pseudo labels from early training rounds.",
                ],
            ),
            StageRecommendation(
                stage_name="Stage 6: Transformer segmentation training",
                goal="Train a final multi-class lettuce disease segmentation model.",
                recommended_method="SegFormer first, Mask2Former second.",
                outputs=["segmentation checkpoints", "per-class masks", "evaluation metrics"],
                implementation_notes=[
                    "SegFormer is the faster, lighter first baseline for your GPU.",
                    "Move to Mask2Former after pseudo-mask quality improves and compute budget allows.",
                ],
            ),
        ]

    def backbone_registry(self) -> list[dict]:
        adapters = [DINOv2BackboneAdapter(), MAEBackboneAdapter()]
        records = []
        for adapter in adapters:
            notes = adapter.notes()
            records.append({
                "name": notes.name,
                "objective": notes.objective,
                "strengths": notes.strengths,
                "caveats": notes.caveats,
            })
        return records

    def write_research_outputs(self, summary: ManifestSummary) -> None:
        self.logger.write_json(
            "research_pipeline_summary.json",
            {
                "recommendation": self.config.research_recommendation,
                "manifest_summary": asdict(summary),
                "stage_plan": [stage.__dict__ for stage in self.research_plan()],
                "backbones": self.backbone_registry(),
            },
        )
