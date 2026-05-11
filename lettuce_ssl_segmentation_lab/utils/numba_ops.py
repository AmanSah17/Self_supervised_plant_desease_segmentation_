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
