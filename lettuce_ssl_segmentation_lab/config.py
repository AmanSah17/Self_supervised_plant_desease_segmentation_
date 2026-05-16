from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence


@dataclass
class LabConfig:
    repo_root: Path = Path(__file__).resolve().parents[1]
    dataset_base: Path = Path("Roboflow_Dataset")
    felz_base: Path = Path("felzenszwalb_masks_output")
    lab_root: Path = Path("lettuce_ssl_segmentation_lab")
    logs_dir: Path = Path("lettuce_ssl_segmentation_lab/logs")
    manifests_dir: Path = Path("lettuce_ssl_segmentation_lab/logs/manifests")

    splits: Sequence[str] = ("train", "validation", "test")
    class_names: Sequence[str] = (
        "BACT", "DML", "HLTY", "PML", "SBL", "SPW", "VIRL", "WLBL", "Leaf-Diseases"
    )
    healthy_class_name: str = "HLTY"

    img_size: tuple[int, int] = (256, 256)
    preferred_felz_signature: Optional[str] = "s80_0_sg0_5_ms20"

    use_cuda: bool = True
    use_numba: bool = True

    selected_channels: Sequence[str] = (
        "rgb",
        "clahe",
        "felz_raw",
        "felz_boundary",
        "felz_colored",
        "watershed",
        "edge",
        "exg",
    )

    research_recommendation: str = (
        "Start with DINOv2 features plus PaDiM for healthy-only anomaly localization, "
        "then add CAM fusion and SegFormer for pseudo-mask supervised segmentation."
    )

    def resolve(self) -> "LabConfig":
        self.dataset_base = self.repo_root / self.dataset_base
        self.felz_base = self.repo_root / self.felz_base
        self.lab_root = self.repo_root / self.lab_root
        self.logs_dir = self.repo_root / self.logs_dir
        self.manifests_dir = self.repo_root / self.manifests_dir
        return self

    @property
    def healthy_index(self) -> int:
        return list(self.class_names).index(self.healthy_class_name)
