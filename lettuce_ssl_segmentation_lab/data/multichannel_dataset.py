from __future__ import annotations
import random
import torchvision.transforms.functional as TF

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
from lettuce_ssl_segmentation_lab.utils.numba_ops import fast_remap_segments, fast_compute_exg


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
    exg: np.ndarray


class MultiChannelLeafDataset(Dataset):
    """Loads all available representations as a parallel-channel tensor."""

    def __init__(self, manifest_df: pd.DataFrame, config: LabConfig, split: str = "train", transform=None):
        self.config = config.resolve()
        self.manifest_df = manifest_df[manifest_df["split"] == split].reset_index(drop=True)
        self.transform = transform
        self.mask_dir = self.config.lab_root / "stage5_cam_attention_masks" / "masks"

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
        exg = fast_compute_exg(rgb_resized.astype(np.float32) / 255.0)
        
        # Load Pseudo Mask if available
        mask_path = self.mask_dir / f"{row['image_stem']}_mask.png"
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.config.img_size[1], self.config.img_size[0]), interpolation=cv2.INTER_NEAREST)
        else:
            mask = np.zeros(self.config.img_size, dtype=np.uint8)

        if self.transform:
            # We apply spatial transforms manually to ensure consistency between image and mask
            if random.random() > 0.5:
                rgb_resized = TF.hflip(torch.from_numpy(rgb_resized).permute(2, 0, 1)).permute(1, 2, 0).numpy()
                mask = TF.hflip(torch.from_numpy(mask[None, ...])).squeeze(0).numpy()
            
            if random.random() > 0.5:
                rgb_resized = TF.vflip(torch.from_numpy(rgb_resized).permute(2, 0, 1)).permute(1, 2, 0).numpy()
                mask = TF.vflip(torch.from_numpy(mask[None, ...])).squeeze(0).numpy()
                
            # Non-spatial transforms can be applied via self.transform (which should be color only now)
            # convert to PIL for standard transforms if needed, or just apply directly if they support tensors
            from PIL import Image
            img_pil = Image.fromarray(rgb_resized)
            img_pil = self.transform(img_pil)
            rgb_resized = np.array(img_pil)

        channels = SampleChannels(
            rgb=rgb_resized,
            clahe=clahe,
            felz_raw=felz_raw,
            felz_boundary=felz_boundary,
            felz_colored=felz_colored,
            watershed=watershed,
            edge=edge,
            segments=segments,
            exg=exg,
        )
        tensor = self._stack_selected_channels(channels)

        return {
            "image": tensor,
            "mask": torch.from_numpy(mask.astype(np.int64)),
            "segments": torch.from_numpy(segments.astype(np.int64)),
            "class_name": row["class_name"],
            "label_kind": str(row["label_kind"]) if pd.notna(row["label_kind"]) else "none",
            "image_path": str(row["image_path"]),
            "image_stem": str(row["image_stem"]),
            "split": str(row["split"]),
            "chosen_variant": str(row["chosen_variant"]) if pd.notna(row["chosen_variant"]) else "none",
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
