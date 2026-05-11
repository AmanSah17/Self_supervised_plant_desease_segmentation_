from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy
import torch

print("Python   :", sys.version)
print("PyTorch  :", torch.__version__)
print("CUDA     :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU      :", torch.cuda.get_device_name(0))
    print("VRAM     :", round(torch.cuda.get_device_properties(0).total_memory/1e9, 2), "GB")
print("OpenCV   :", cv2.__version__)
print("NumPy    :", numpy.__version__)
