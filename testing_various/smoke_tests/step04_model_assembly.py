"""
STEP 4 (v2): Complex Model Assembly & Gradient Verification
- Initializes LARGE DRSA-Net (256 embed, 8 blocks, 64 branches)
- Verifies gradient flow and VRAM usage on GTX 1650
- Saves 'complex_model_verified' checkpoint
"""
from __future__ import annotations

import time
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding='utf-8')

from drsa_net.config import DRSAConfig
from drsa_net.model.drsa_net import DRSANet
from drsa_net.training.losses import DRSALoss
from drsa_net.data.dataset import build_dataloaders
from drsa_net.model.superpixel_tokenizer import build_adjacency_matrix

# ── Fresh Config (picking up new complex defaults) ────────────────────────
cfg = DRSAConfig()
cfg.validate()

OUT_DIR    = Path("drsa_net_output")
OUT_DIR.mkdir(exist_ok=True)
MODEL_CKPT = OUT_DIR / "step04_complex_model_verified.pth"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*65)
print("STEP 4: Complex Model Assembly & Gradient Check")
print("="*65)
print(f"  Device           : {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU              : {torch.cuda.get_device_name(0)}")
    print(f"  Total VRAM       : {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
print()

# ── Initialize Model ──────────────────────────────────────────────────────
print("  [1/4] Initializing Complex DRSA-Net...")
model = DRSANet(cfg).to(DEVICE)
criterion = DRSALoss(cfg).to(DEVICE)
print(model.summary())

# ── Load Batch ────────────────────────────────────────────────────────────
print("\n  [2/4] Loading sample batch...")
train_loader, _ = build_dataloaders(cfg)
batch = next(iter(train_loader))

# Move batch to device
batch_cuda = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
labels = batch_cuda['label']

# DEBUG: Check inputs
for k in ('rgb', 'clahe', 'felz1', 'felz2', 'watershed'):
    print(f"    - Input {k} NaN: {torch.isnan(batch_cuda[k]).any().item()}")

# ── Forward Pass ──────────────────────────────────────────────────────────
print("\n  [3/4] Running Forward Pass (Mixed Precision)...")
t0 = time.time()
with autocast(enabled=cfg.use_amp):
    # 1. Encoder
    fm = model.encoder(
        batch_cuda['rgb'], batch_cuda['clahe'], 
        batch_cuda['felz1'], batch_cuda['felz2'], batch_cuda['watershed']
    )
    print(f"    - Encoder feature map NaN: {torch.isnan(fm).any().item()}")

    # 2. Tokenizer
    tokens, local_mask, sg_mask, info = model.tokenizer(fm, batch_cuda['segments'])
    print(f"    - Tokens NaN: {torch.isnan(tokens).any().item()}")
    print(f"    - Local Mask -inf count: {torch.isinf(local_mask).sum().item()}")

    # 3. Transformer
    trans_out = model.transformer(tokens, local_mask, sg_mask)
    print(f"    - Transformer output NaN: {torch.isnan(trans_out['tokens_out']).any().item()}")

    # 4. CAM Gen
    rollout = model.transformer.get_rollout_attention(trans_out['attn_maps'])
    adj_bool = (local_mask == 0.0).float()
    cam_out = model.cam_gen(
        rollout, batch_cuda['watershed'], batch_cuda['felz2'], 
        batch_cuda['clahe'], info['seg_ds'], adj_bool, cfg.img_size
    )
    print(f"    - CAM NaN: {torch.isnan(cam_out['propagated_cam']).any().item()}")

print(f"    ✓ Forward Pass: {time.time()-t0:.3f}s")
if torch.cuda.is_available():
    print(f"    ✓ Peak VRAM Allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

# ── Backward Pass ─────────────────────────────────────────────────────────
print("\n  [4/4] Running Backward Pass...")
model.zero_grad()
out1 = model(
    batch_cuda['rgb'],
    batch_cuda['clahe'],
    batch_cuda['felz1'],
    batch_cuda['felz2'],
    batch_cuda['watershed'],
    batch_cuda['segments'],
    return_cam=True,
)

adj_list = []
num_tokens = out1['tokens_out'].shape[1]
for b_idx in range(labels.shape[0]):
    _, remap = torch.unique(out1['seg_ds'][b_idx], return_inverse=True)
    remap = remap.clamp(0, num_tokens - 1)
    adj_list.append(build_adjacency_matrix(remap, num_tokens, k_hop=1).float())
adj = torch.stack(adj_list).to(DEVICE)

losses = criterion(out1, out1, labels, adj, return_components=True)
losses['total'].backward()

# Verify gradients
grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
print(f"    ✓ Total Gradient Norm: {grad_norm:.4f}")

# ── Save Checkpoint ───────────────────────────────────────────────────────
torch.save({
    'model_state_dict': model.state_dict(),
    'config': cfg.__dict__,
}, MODEL_CKPT)

print(f"\n  ✓ Complex model verified and saved to: {MODEL_CKPT}")
print("[STEP 4 COMPLETE]")
