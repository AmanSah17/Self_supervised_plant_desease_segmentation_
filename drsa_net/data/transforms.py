"""
DRSA-Net Image Transforms
=========================
All transforms operate on **all 4 representations simultaneously** using
identical spatial parameters (same crop, flip, resize) to maintain pixel
correspondence across branches.

CLAHE and Watershed are computed here at data-load time (not inside the
model) so they can be cached/batched efficiently.
"""
from __future__ import annotations

import random
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from scipy import ndimage as ndi
from skimage.segmentation import watershed


# --------------------------------------------------------------------------- #
#  Image-level representation transforms                                       #
# --------------------------------------------------------------------------- #

def apply_clahe(img_rgb: np.ndarray, clip_limit: float = 10.0,
                tile_grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE in LAB color space (L-channel only) for lesion enhancement.

    Parameters
    ----------
    img_rgb : H×W×3 uint8 RGB image
    clip_limit : CLAHE clip limit (10 as specified)
    tile_grid : CLAHE tile grid size

    Returns
    -------
    H×W×3 uint8 CLAHE-enhanced RGB image
    """
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
    return cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)


def apply_watershed(img_rgb: np.ndarray,
                    compactness: float = 0.002,
                    min_size: int = 10) -> np.ndarray:
    """
    Generate a Watershed-based segment map and return it as a normalized
    float32 single-channel image.

    Strategy:
      1. Compute gradient magnitude in LAB space (edge map)
      2. Place seeds via distance transform peaks
      3. Run compact watershed
      4. Normalize label map → [0, 1] float32

    Parameters
    ----------
    img_rgb : H×W×3 uint8 RGB image

    Returns
    -------
    H×W×1 float32 watershed map (normalized label IDs)
    """
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_ch = img_lab[:, :, 0] / 255.0

    # Gradient magnitude as "elevation" map
    gx = cv2.Sobel(l_ch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(l_ch, cv2.CV_32F, 0, 1, ksize=3)
    edge_map = np.sqrt(gx ** 2 + gy ** 2).astype(np.float32)
    edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min() + 1e-6)

    # Seed generation via distance transform
    smooth = cv2.GaussianBlur(l_ch, (5, 5), 1.0)
    # Otsu requires uint8
    smooth_u8 = (smooth * 255).clip(0, 255).astype(np.uint8)
    _, thresh_u8 = cv2.threshold(smooth_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dist = cv2.distanceTransform(thresh_u8, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, markers_bin = cv2.threshold(dist_norm, int(0.3 * dist_norm.max()), 255, 0)
    markers_u8 = markers_bin.astype(np.uint8)
    _, markers = cv2.connectedComponents(markers_u8)

    # Compact watershed
    edge_u8 = (edge_map * 255).astype(np.uint8)
    edge_3ch = cv2.cvtColor(edge_u8, cv2.COLOR_GRAY2BGR)
    ws_result = cv2.watershed(edge_3ch, markers.astype(np.int32))
    ws_result = np.clip(ws_result, 0, None).astype(np.float32)

    # Merge tiny segments (simple: flood fill isn't needed, just normalize)
    ws_max = ws_result.max()
    if ws_max > 0:
        ws_norm = ws_result / ws_max
    else:
        ws_norm = ws_result

    return ws_norm.astype(np.float32)   # H×W


# --------------------------------------------------------------------------- #
#  Synchronized spatial transforms                                             #
# --------------------------------------------------------------------------- #

class SynchronizedTransform:
    """
    Applies identical spatial augmentations to all representation streams
    simultaneously.  Pixel-level correspondence is preserved.

    Representations dict keys expected:
        'rgb'    : H×W×3 uint8
        'clahe'  : H×W×3 uint8
        'watershed': H×W   float32
        'felz1'  : H×W   float32   (first felzenszwalb mask)
        'felz2'  : H×W   float32   (second felzenszwalb mask)
        'segments': H×W  int32     (superpixel integer label map)
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (256, 256),
        augment: bool = True,
        hflip_p: float = 0.5,
        vflip_p: float = 0.3,
        rot_degrees: float = 30.0,
        color_jitter_p: float = 0.8,
        brightness: float = 0.3,
        contrast: float = 0.3,
        saturation: float = 0.3,
        hue: float = 0.1,
    ):
        self.img_size = img_size
        self.augment = augment
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.rot_degrees = rot_degrees
        self.color_jitter_p = color_jitter_p
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # ---- Resize all representations to target size ----
        h, w = self.img_size
        data = self._resize_all(data, h, w)

        if not self.augment:
            return data

        # ---- Shared spatial decisions ----
        do_hflip = random.random() < self.hflip_p
        do_vflip = random.random() < self.vflip_p
        angle = random.uniform(-self.rot_degrees, self.rot_degrees)
        do_color = random.random() < self.color_jitter_p

        if do_hflip:
            data = {k: np.flip(v, axis=1).copy() for k, v in data.items()}
        if do_vflip:
            data = {k: np.flip(v, axis=0).copy() for k, v in data.items()}

        # Rotation (nearest-neighbour for integer maps, bilinear for float)
        if abs(angle) > 0.5:
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            for key in ('rgb', 'clahe'):
                data[key] = cv2.warpAffine(
                    data[key], M, (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
            for key in ('watershed', 'felz1', 'felz2'):
                data[key] = cv2.warpAffine(
                    data[key], M, (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
            data['segments'] = cv2.warpAffine(
                data['segments'].astype(np.float32), M, (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_REFLECT_101,
            ).astype(np.int32)

        # Color jitter: only on RGB and CLAHE
        if do_color:
            br = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            cr = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            for key in ('rgb', 'clahe'):
                t = torch.from_numpy(data[key]).permute(2, 0, 1).float() / 255.0
                t = TF.adjust_brightness(t, br)
                t = TF.adjust_contrast(t, cr)
                data[key] = (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

        return data

    @staticmethod
    def _resize_all(data: Dict[str, np.ndarray], h: int, w: int) -> Dict[str, np.ndarray]:
        out = {}
        for key, val in data.items():
            if key in ('rgb', 'clahe'):
                out[key] = cv2.resize(val, (w, h), interpolation=cv2.INTER_LINEAR)
            elif key == 'segments':
                out[key] = cv2.resize(
                    val.astype(np.float32), (w, h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.int32)
            else:  # watershed, felz1, felz2 — float maps
                out[key] = cv2.resize(val, (w, h), interpolation=cv2.INTER_LINEAR)
        return out


# --------------------------------------------------------------------------- #
#  Tensor normalization                                                        #
# --------------------------------------------------------------------------- #

# ImageNet mean/std for RGB and CLAHE branches
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_tensor_dict(data: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    """
    Convert numpy representation dict → torch tensors (C, H, W) float32.

    Normalisation:
      - 'rgb', 'clahe'  : ImageNet norm (mean/std)
      - 'felz1', 'felz2', 'watershed' : scale to [0,1], add channel dim → (1, H, W)
      - 'segments' : int32 → int64 tensor (H, W)  [for scatter ops]
    """
    out: Dict[str, torch.Tensor] = {}

    for key in ('rgb', 'clahe'):
        arr = data[key].astype(np.float32) / 255.0
        arr = (arr - _IMAGENET_MEAN) / (_IMAGENET_STD + 1e-6)
        out[key] = torch.from_numpy(arr.transpose(2, 0, 1))  # (3, H, W)

    for key in ('felz1', 'felz2', 'watershed'):
        arr = data[key].astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
        out[key] = torch.from_numpy(arr).unsqueeze(0)          # (1, H, W)

    out['segments'] = torch.from_numpy(
        data['segments'].astype(np.int64)
    )  # (H, W)

    return out
