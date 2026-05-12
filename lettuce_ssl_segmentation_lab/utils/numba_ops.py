from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):  # type: ignore
        def wrapper(func):
            return func
        return wrapper


@njit
def fast_remap_segments(quant_img: np.ndarray, unique_vals: np.ndarray) -> np.ndarray:
    """Map grayscale superpixel IDs into dense consecutive segment IDs."""
    height, width = quant_img.shape
    out = np.zeros((height, width), dtype=np.int32)
    lut = np.zeros(256, dtype=np.int32)
    for idx in range(len(unique_vals)):
        lut[int(unique_vals[idx])] = idx
    for row in range(height):
        for col in range(width):
            out[row, col] = lut[quant_img[row, col]]
    return out

@njit
def fast_compute_exg(rgb_img: np.ndarray) -> np.ndarray:
    """Compute Excess Green Index (2*G - R - B)."""
    h, w, _ = rgb_img.shape
    exg = np.zeros((h, w), dtype=np.float32)
    # rgb_img is expected to be float32 [0, 1] or uint8 [0, 255]
    # We'll handle it generically
    for r in range(h):
        for c in range(w):
            red = rgb_img[r, c, 0]
            green = rgb_img[r, c, 1]
            blue = rgb_img[r, c, 2]
            exg[r, c] = 2.0 * green - red - blue
    return exg
