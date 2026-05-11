from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from drsa_net.data.transforms import apply_clahe, apply_watershed
from lettuce_ssl_segmentation_lab.config import LabConfig
from lettuce_ssl_segmentation_lab.utils.numba_ops import fast_remap_segments


@dataclass
class SampleChannels:
    rgb: np.ndarray
    clahe: np.ndarray
    felz_raw: np.ndarray
    felz_boundary: np.ndarray
    felz_colored: np.ndarray
    watershed: np.ndarray
    edge: np.ndarray
    segments: np.ndarray


class MultiChannelLeafDataset(Dataset):
    """Loads all available representations as a parallel-channel tensor."""

    def __init__(self, manifest_df: pd.DataFrame, config: LabConfig, split: str = "train"):
        self.config = config.resolve()
        self.manifest_df = manifest_df[manifest_df["split"] == split].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.manifest_df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.manifest_df.iloc[index]
        rgb = self._load_rgb(Path(row["image_path"]))
        rgb_resized = cv2.resize(rgb, self.config.img_size[::-1], interpolation=cv2.INTER_LINEAR)
        clahe = apply_clahe(rgb_resized)
        watershed = apply_watershed(rgb_resized)
        felz_raw = self._load_gray(row.get("felz_raw_path"), rgb_resized.shape[:2])
        felz_boundary = self._load_gray(row.get("felz_boundary_path"), rgb_resized.shape[:2])
        felz_colored = self._load_rgb_optional(row.get("felz_colored_path"), rgb_resized.shape[:2])
        edge = self._compute_edge_map(rgb_resized)
        segments = self._segments_from_felz(felz_raw)

        channels = SampleChannels(
            rgb=rgb_resized,
            clahe=clahe,
            felz_raw=felz_raw,
            felz_boundary=felz_boundary,
            felz_colored=felz_colored,
            watershed=watershed,
            edge=edge,
            segments=segments,
        )
        tensor = self._stack_selected_channels(channels)

        return {
            "image": tensor,
            "segments": torch.from_numpy(segments.astype(np.int64)),
            "class_name": row["class_name"],
            "label_kind": row["label_kind"],
            "image_path": row["image_path"],
            "image_stem": row["image_stem"],
            "split": row["split"],
            "chosen_variant": row["chosen_variant"],
        }

    def _load_rgb(self, image_path: Path) -> np.ndarray:
        img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def _load_rgb_optional(self, path_value: Any, target_hw: tuple[int, int]) -> np.ndarray:
        height, width = target_hw
        if not isinstance(path_value, str) or not Path(path_value).exists():
            return np.zeros((height, width, 3), dtype=np.uint8)
        img_bgr = cv2.imread(path_value, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return np.zeros((height, width, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)

    def _load_gray(self, path_value: Any, target_hw: tuple[int, int]) -> np.ndarray:
        height, width = target_hw
        if not isinstance(path_value, str) or not Path(path_value).exists():
            return np.zeros((height, width), dtype=np.float32)
        arr = cv2.imread(path_value, cv2.IMREAD_GRAYSCALE)
        if arr is None:
            return np.zeros((height, width), dtype=np.float32)
        arr = cv2.resize(arr, (width, height), interpolation=cv2.INTER_NEAREST)
        return arr.astype(np.float32) / 255.0

    def _compute_edge_map(self, rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-6)
        return mag.astype(np.float32)

    def _segments_from_felz(self, felz_raw: np.ndarray) -> np.ndarray:
        quant = (felz_raw * 255.0).round().astype(np.uint8)
        unique_vals = np.unique(quant)
        return fast_remap_segments(quant, unique_vals)

    def _stack_selected_channels(self, sample: SampleChannels) -> torch.Tensor:
        channel_tensors: list[np.ndarray] = []
        for channel_name in self.config.selected_channels:
            value = getattr(sample, channel_name)
            if value.ndim == 2:
                channel_tensors.append(value[None, ...].astype(np.float32))
            else:
                channel_tensors.append(value.transpose(2, 0, 1).astype(np.float32) / 255.0)
        stacked = np.concatenate(channel_tensors, axis=0)
        return torch.from_numpy(stacked)
