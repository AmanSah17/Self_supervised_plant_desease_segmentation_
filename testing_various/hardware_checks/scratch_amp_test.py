from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from drsa_net.model.multistream_encoder import ResidualBlock

torch.backends.cudnn.enabled = False

m = ResidualBlock(64).cuda()
x = torch.randn(2, 64, 128, 128).cuda()

with torch.amp.autocast('cuda'):
    out1 = m.conv1(x)
    print("conv1 NaN with cudnn disabled:", torch.isnan(out1).any().item())
