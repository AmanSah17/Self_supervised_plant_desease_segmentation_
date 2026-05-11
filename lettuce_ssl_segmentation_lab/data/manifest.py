from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import re

import pandas as pd

from lettuce_ssl_segmentation_lab.config import LabConfig


def _strip_felz_suffix(name: str) -> str:
    for suffix in ("_raw.png", "_boundary.png", "_colored.png", "_info.txt"):
        if name.endswith(suffix):
            stem = name[: -len(suffix)]
            return re.sub(r"_(s\d+.*)$", "", stem)
    return Path(name).stem


def _variant_key(name: str) -> str:
    for suffix in ("_raw.png", "_boundary.png", "_colored.png", "_info.txt"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


@dataclass
class ManifestSummary:
    total_samples: int
    healthy_samples: int
    diseased_samples: int
    samples_with_felz_raw: int
    samples_with_felz_boundary: int
    samples_with_felz_colored: int
    classes_seen: list[str]
    splits_seen: list[str]


class MultiRepresentationManifestBuilder:
    def __init__(self, config: LabConfig):
        self.config = config.resolve()

    def build(self) -> pd.DataFrame:
        rows: list[dict] = []
        for split in self.config.splits:
            split_dir = self.config.dataset_base / split
            felz_split_dir = self.config.felz_base / split
            if not split_dir.exists():
                continue

            for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                class_name = class_dir.name
                label_kind = "healthy" if class_name == self.config.healthy_class_name else "disease"
                felz_index = self._index_felz_variants(felz_split_dir / class_name)

                for image_path in sorted(class_dir.iterdir()):
                    if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                        continue
                    sample_key = image_path.stem
                    variant_records = felz_index.get(sample_key, {})
                    chosen_variant = self._choose_variant(variant_records)
                    rows.append({
                        "split": split,
                        "class_name": class_name,
                        "label_kind": label_kind,
                        "image_path": str(image_path),
                        "image_stem": sample_key,
                        "num_felz_variants": len(variant_records),
                        "chosen_variant": chosen_variant,
                        "felz_raw_path": self._lookup_variant_asset(variant_records, chosen_variant, "raw"),
                        "felz_boundary_path": self._lookup_variant_asset(variant_records, chosen_variant, "boundary"),
                        "felz_colored_path": self._lookup_variant_asset(variant_records, chosen_variant, "colored"),
                        "felz_info_path": self._lookup_variant_asset(variant_records, chosen_variant, "info"),
                    })

        return pd.DataFrame(rows)

    def summarize(self, manifest_df: pd.DataFrame) -> ManifestSummary:
        return ManifestSummary(
            total_samples=int(len(manifest_df)),
            healthy_samples=int((manifest_df["label_kind"] == "healthy").sum()),
            diseased_samples=int((manifest_df["label_kind"] == "disease").sum()),
            samples_with_felz_raw=int(manifest_df["felz_raw_path"].notna().sum()),
            samples_with_felz_boundary=int(manifest_df["felz_boundary_path"].notna().sum()),
            samples_with_felz_colored=int(manifest_df["felz_colored_path"].notna().sum()),
            classes_seen=sorted(manifest_df["class_name"].dropna().unique().tolist()),
            splits_seen=sorted(manifest_df["split"].dropna().unique().tolist()),
        )

    def _index_felz_variants(self, class_dir: Path) -> dict[str, dict[str, dict[str, str]]]:
        index: dict[str, dict[str, dict[str, str]]] = {}
        if not class_dir.exists():
            return index

        for file_path in sorted(class_dir.iterdir()):
            if not file_path.is_file():
                continue
            file_name = file_path.name
            sample_key = _strip_felz_suffix(file_name)
            variant = _variant_key(file_name)
            asset_type = self._infer_asset_type(file_name)
            if asset_type is None:
                continue
            index.setdefault(sample_key, {}).setdefault(variant, {})[asset_type] = str(file_path)
        return index

    def _infer_asset_type(self, file_name: str) -> Optional[str]:
        if file_name.endswith("_raw.png"):
            return "raw"
        if file_name.endswith("_boundary.png"):
            return "boundary"
        if file_name.endswith("_colored.png"):
            return "colored"
        if file_name.endswith("_info.txt"):
            return "info"
        return None

    def _choose_variant(self, variant_records: dict[str, dict[str, str]]) -> Optional[str]:
        if not variant_records:
            return None
        keys = sorted(variant_records)
        preferred = self.config.preferred_felz_signature
        if preferred:
            for key in keys:
                if preferred in key:
                    return key
        for key in keys:
            record = variant_records[key]
            if "raw" in record and "boundary" in record:
                return key
        return keys[0]

    def _lookup_variant_asset(
        self,
        variant_records: dict[str, dict[str, str]],
        variant_name: Optional[str],
        asset_type: str,
    ) -> Optional[str]:
        if variant_name is None:
            return None
        record = variant_records.get(variant_name, {})
        return record.get(asset_type)
