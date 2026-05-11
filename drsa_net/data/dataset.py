"""
DRSA-Net Dataset
================
Loads all 4 parallel representation streams for each image:

  1. Original RGB image
  2. Felzenszwalb mask 1 (raw label map — _raw.png)
  3. Felzenszwalb mask 2 (boundary mask — _boundary.png, encodes edge structure)
  4. CLAHE-enhanced image (clip_limit=10, computed on-the-fly)
  5. Watershed transform (computed on-the-fly)
  6. Superpixel segment integer map (derived from Felzenszwalb raw mask)

All 5 representations undergo IDENTICAL spatial transforms.
"""
from __future__ import annotations

import glob
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from numba import njit
import json
import time

from drsa_net.config import DRSAConfig
from drsa_net.data.transforms import (
    SynchronizedTransform,
    apply_clahe,
    apply_watershed,
    to_tensor_dict,
)


# --------------------------------------------------------------------------- #
#  Numba Accelerated Remapping                                                 #
# --------------------------------------------------------------------------- #

@njit
def fast_remap_segments(quant_img, unique_vals):
    """
    Accelerated segment remapping using Numba.
    Replaces slow np.vectorize mapping.
    """
    h, w = quant_img.shape
    out = np.zeros((h, w), dtype=np.int32)
    # Lookup table for uint8 (0-255)
    lut = np.zeros(256, dtype=np.int32)
    for i in range(len(unique_vals)):
        lut[int(unique_vals[i])] = i
    
    for y in range(h):
        for x in range(w):
            out[y, x] = lut[quant_img[y, x]]
    return out


class DRSADataset(Dataset):
    """
    PyTorch Dataset for DRSA-Net.

    Returns
    -------
    dict with keys:
        'rgb'       : (3, H, W) float32 tensor  — ImageNet-normalised
        'clahe'     : (3, H, W) float32 tensor  — CLAHE-enhanced, normalised
        'felz1'     : (1, H, W) float32 tensor  — raw Felz mask 1 (label map)
        'felz2'     : (1, H, W) float32 tensor  — boundary Felz mask 2
        'watershed' : (1, H, W) float32 tensor  — watershed map
        'segments'  : (H, W)   int64 tensor     — superpixel label map
        'label'     : int64 scalar              — disease class index
        'image_path': str
    """

    def __init__(
        self,
        config: DRSAConfig,
        split: str = "train",
        augment: bool = True,
    ):
        self.config = config
        self.split = split
        self.augment = augment
        self.class_to_idx = config.class_to_idx

        # Spatial + color augmentation pipeline
        self.transform = SynchronizedTransform(
            img_size=config.img_size,
            augment=augment,
        )

        # Discover image–mask pairs
        self.samples: List[Dict] = []
        self._discover_samples()

    # ------------------------------------------------------------------ #
    #  Discovery                                                           #
    # ------------------------------------------------------------------ #

    def _discover_samples(self) -> None:
        """
        Walk dataset_base/<split>/<class>/ and match original images to their Felzenszwalb masks.
        Uses a JSON cache if available in the output directory.
        """
        output_dir = Path(self.config.output_dir)
        cache_path = output_dir / f"sample_index_{self.split}.json"
        
        if cache_path.exists():
            print(f"  [CACHE] Loading {self.split} samples from {cache_path.name}")
            with open(cache_path) as f:
                self.samples = json.load(f)
                # Convert string paths back to Path objects if needed, 
                # but __getitem__ can handle strings or Paths.
            return

        print(f"  [INIT] Scanning {self.split} filesystem (one-time cost)...")
        dataset_dir = Path(self.config.dataset_base) / self.split
        felz_dir    = Path(self.config.felz_masks_base) / self.split

        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset split not found: {dataset_dir}")

        count = 0
        for cls_name in sorted(self.class_to_idx.keys()):
            img_dir  = dataset_dir / cls_name
            mask_dir = felz_dir    / cls_name

            if not img_dir.exists():
                continue

            img_paths = sorted(
                p for p in img_dir.iterdir()
                if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}
            )
            if self.config.samples_per_class is not None:
                img_paths = img_paths[:self.config.samples_per_class]

            for img_path in img_paths:
                raw_mask, boundary_mask = self._find_felz_masks(
                    img_path, mask_dir
                )
                self.samples.append({
                    'image_path': img_path,
                    'felz_raw':   raw_mask,       # may be None
                    'felz_bnd':   boundary_mask,  # may be None
                    'label':      self.class_to_idx[cls_name],
                    'class_name': cls_name,
                })
                count += 1

        if count == 0:
            raise RuntimeError(
                f"No samples discovered for split='{self.split}'. "
                f"Check paths: {dataset_dir}, {felz_dir}"
            )

        # Save to cache
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            # Convert Path objects to strings for JSON
            serializable = []
            for s in self.samples:
                serializable.append({
                    'image_path': str(s['image_path']),
                    'felz_raw':   str(s['felz_raw']) if s['felz_raw'] else None,
                    'felz_bnd':   str(s['felz_bnd']) if s['felz_bnd'] else None,
                    'label':      s['label'],
                    'class_name': s['class_name']
                })
            json.dump(serializable, f)
            print(f"  [CACHE] Saved {self.split} samples to {cache_path.name}")
        
        # Keep internal samples as strings or Paths consistently
        self.samples = serializable

    @staticmethod
    def _find_felz_masks(
        img_path: Path,
        mask_dir: Path,
    ) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Find _raw.png and _boundary.png masks for img_path inside mask_dir.
        Matching is done by original image stem (before any augmentation suffix).
        """
        if not mask_dir.exists():
            return None, None

        stem = img_path.stem
        raw_candidates = list(mask_dir.glob(f"{stem}*_raw.png"))
        bnd_candidates = list(mask_dir.glob(f"{stem}*_boundary.png"))

        raw_mask = raw_candidates[0] if raw_candidates else None
        bnd_mask = bnd_candidates[0] if bnd_candidates else None
        return raw_mask, bnd_mask

    # ------------------------------------------------------------------ #
    #  Felzenszwalb mask loading                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_felz_raw(path_val: Optional[str | Path], h: int, w: int) -> np.ndarray:
        """
        Load _raw.png (normalized 0–255 label map) → float32 [0,1].
        """
        if path_val is None:
            return np.zeros((h, w), dtype=np.float32)
        
        p = Path(path_val)
        if not p.exists():
            return np.zeros((h, w), dtype=np.float32)

        mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return np.zeros((h, w), dtype=np.float32)
        return mask.astype(np.float32) / 255.0

    @staticmethod
    def _load_felz_boundary(path_val: Optional[str | Path], h: int, w: int) -> np.ndarray:
        """
        Load _boundary.png (binary boundary map 0/255) → float32 {0,1}.
        """
        if path_val is None:
            return np.zeros((h, w), dtype=np.float32)
        
        p = Path(path_val)
        if not p.exists():
            return np.zeros((h, w), dtype=np.float32)

        mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return np.zeros((h, w), dtype=np.float32)
        return (mask > 127).astype(np.float32)

    @staticmethod
    def _felz_raw_to_segments(felz_raw: np.ndarray) -> np.ndarray:
        """
        Convert normalized float felz map → integer segment label map.
        Uses Numba-accelerated remapping.
        """
        quantised = (felz_raw * 255).round().astype(np.uint8)
        unique_vals = np.unique(quantised)
        return fast_remap_segments(quantised, unique_vals)

    # ------------------------------------------------------------------ #
    #  __getitem__                                                         #
    # ------------------------------------------------------------------ #

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # -- Load original RGB -----------------------------------------
        img_path = Path(sample['image_path'])
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read: {sample['image_path']}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        # -- Branch representations (computed before spatial transform) --
        clahe_img  = apply_clahe(
            img_rgb,
            clip_limit=self.config.clahe_clip_limit,
            tile_grid=self.config.clahe_tile_grid,
        )
        watershed_map = apply_watershed(
            img_rgb,
            compactness=self.config.watershed_compactness,
            min_size=self.config.watershed_min_size,
        )
        felz1 = self._load_felz_raw(sample['felz_raw'], h, w)
        felz2 = self._load_felz_boundary(sample['felz_bnd'], h, w)

        # Superpixel integer map from Felzenszwalb raw mask
        segments = self._felz_raw_to_segments(felz1)

        # -- Build representation dict ---------------------------------
        data = {
            'rgb':       img_rgb,      # H×W×3 uint8
            'clahe':     clahe_img,    # H×W×3 uint8
            'felz1':     felz1,        # H×W float32
            'felz2':     felz2,        # H×W float32
            'watershed': watershed_map, # H×W float32
            'segments':  segments,     # H×W int32
        }

        # -- Synchronized spatial transforms ---------------------------
        data = self.transform(data)

        # -- To tensors ------------------------------------------------
        tensors = to_tensor_dict(data)

        return {
            **tensors,
            'label':      torch.tensor(sample['label'], dtype=torch.int64),
            'image_path': str(sample['image_path']),
        }

    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------ #
    #  Collate helper                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """
        Custom collate that handles variable-length superpixel segment maps.
        Segment maps are int64, so we stack normally (all same spatial size).
        """
        keys = batch[0].keys()
        out = {}
        for key in keys:
            if key == 'image_path':
                out[key] = [b[key] for b in batch]
            else:
                out[key] = torch.stack([b[key] for b in batch])
        return out


# --------------------------------------------------------------------------- #
#  Factory functions                                                           #
# --------------------------------------------------------------------------- #

def build_dataloaders(config: DRSAConfig):
    """Build train / validation DataLoaders."""
    from torch.utils.data import DataLoader

    train_ds = DRSADataset(config, split='train',      augment=True)
    val_ds   = DRSADataset(config, split='validation', augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=DRSADataset.collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=DRSADataset.collate_fn,
    )
    return train_loader, val_loader
