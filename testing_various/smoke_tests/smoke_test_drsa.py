"""
DRSA-Net Shape Smoke Test
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

# Force UTF-8 stdout on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import numpy as np


def banner(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print('='*60)


def check(cond: bool, msg: str) -> None:
    status = "  [PASS]" if cond else "  [FAIL]"
    print(f"{status}  {msg}")
    if not cond:
        sys.exit(1)


# -- Config ------------------------------------------------------------------
banner("Stage 0: Config")
from drsa_net.config import DRSAConfig
cfg = DRSAConfig()
cfg.img_size               = (128, 128)
cfg.embed_dim              = 64
cfg.branch_channels        = 16
cfg.num_transformer_layers = 2
cfg.num_attention_heads    = 4
cfg.max_superpixels        = 32
cfg.batch_size             = 2
cfg.num_workers            = 0
cfg.use_amp                = False
cfg.samples_per_class      = 1
cfg.validate()
check(True, "DRSAConfig created and validated")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
B, H, W = 2, 128, 128
D = cfg.embed_dim
print(f"  Device: {DEVICE}")

# -- Transforms --------------------------------------------------------------
banner("Stage 1: Transforms")
from drsa_net.data.transforms import (
    apply_clahe, apply_watershed, SynchronizedTransform, to_tensor_dict
)

dummy_rgb = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
clahe_out = apply_clahe(dummy_rgb, clip_limit=10.0)
check(clahe_out.shape == (H, W, 3), f"CLAHE output shape: {clahe_out.shape}")
check(clahe_out.dtype == np.uint8,  "CLAHE dtype uint8")

ws_out = apply_watershed(dummy_rgb)
check(ws_out.shape == (H, W),      f"Watershed output shape: {ws_out.shape}")
check(ws_out.dtype == np.float32,  "Watershed dtype float32")

transform = SynchronizedTransform(img_size=(H, W), augment=True)
dummy_data = {
    'rgb':       dummy_rgb,
    'clahe':     clahe_out,
    'felz1':     np.random.rand(H, W).astype(np.float32),
    'felz2':     np.random.rand(H, W).astype(np.float32),
    'watershed': ws_out,
    'segments':  np.random.randint(0, 10, (H, W), dtype=np.int32),
}
transformed = transform(dummy_data)
tensors = to_tensor_dict(transformed)
check(tensors['rgb'].shape       == (3, H, W), f"rgb tensor: {tensors['rgb'].shape}")
check(tensors['clahe'].shape     == (3, H, W), f"clahe tensor: {tensors['clahe'].shape}")
check(tensors['felz1'].shape     == (1, H, W), f"felz1 tensor: {tensors['felz1'].shape}")
check(tensors['watershed'].shape == (1, H, W), f"watershed tensor: {tensors['watershed'].shape}")
check(tensors['segments'].dtype  == torch.int64, "segments dtype int64")

# -- Multi-Stream Encoder ----------------------------------------------------
banner("Stage 2: MultiStreamEncoder")
from drsa_net.model.multistream_encoder import MultiStreamEncoder
encoder = MultiStreamEncoder(
    branch_channels=cfg.branch_channels,
    embed_dim=cfg.embed_dim,
).to(DEVICE)

rgb_t   = torch.randn(B, 3, H, W).to(DEVICE)
clahe_t = torch.randn(B, 3, H, W).to(DEVICE)
felz1_t = torch.randn(B, 1, H, W).to(DEVICE)
felz2_t = torch.randn(B, 1, H, W).to(DEVICE)
ws_t    = torch.randn(B, 1, H, W).to(DEVICE)

feat_map   = encoder(rgb_t, clahe_t, felz1_t, felz2_t, ws_t)
expected_h = H // 4
check(feat_map.shape == (B, D, expected_h, expected_h),
      f"Encoder output: {feat_map.shape} (expected {(B, D, expected_h, expected_h)})")

# -- Superpixel Tokenizer ----------------------------------------------------
banner("Stage 3: SuperpixelTokenizer")
from drsa_net.model.superpixel_tokenizer import SuperpixelTokenizer

seg_map   = torch.randint(0, 8, (B, H, W), dtype=torch.int64).to(DEVICE)
tokenizer = SuperpixelTokenizer(
    embed_dim=cfg.embed_dim,
    max_segments=cfg.max_superpixels,
    k_hop=cfg.attention_k_hop,
).to(DEVICE)

tokens, local_mask, sg_mask, info = tokenizer(feat_map, seg_map)
N_sp = tokens.shape[1]
check(tokens.shape[0]  == B,                 f"Tokens batch dim: {tokens.shape}")
check(tokens.shape[2]  == D,                 f"Tokens embed dim: {tokens.shape[2]}")
check(local_mask.shape == (B, N_sp, N_sp),   f"Local mask shape: {local_mask.shape}")
check(sg_mask.shape    == (B, N_sp, N_sp),   f"Semi-global mask shape: {sg_mask.shape}")
check(N_sp <= cfg.max_superpixels,           f"N_sp within max: {N_sp}")
print(f"  N_sp = {N_sp}")

# -- Region-Aware Transformer ------------------------------------------------
banner("Stage 4: RegionAwareTransformer")
from drsa_net.model.region_aware_transformer import RegionAwareTransformer

transformer = RegionAwareTransformer(
    num_layers=cfg.num_transformer_layers,
    embed_dim=cfg.embed_dim,
    num_heads=cfg.num_attention_heads,
    num_classes=cfg.num_classes,
).to(DEVICE)

trans_out = transformer(tokens, local_mask, sg_mask)
check(trans_out['cls_logits'].shape == (B, cfg.num_classes),
      f"CLS logits: {trans_out['cls_logits'].shape}")
check(trans_out['tokens_out'].shape == (B, N_sp, D),
      f"Token output: {trans_out['tokens_out'].shape}")
check(len(trans_out['attn_maps']) == cfg.num_transformer_layers,
      f"Num attention maps: {len(trans_out['attn_maps'])}")

rollout = transformer.get_rollout_attention(trans_out['attn_maps'])
check(rollout.shape == (B, N_sp), f"Rollout: {rollout.shape}")
check(not rollout.isnan().any(),  "Rollout has no NaN")

# -- CAM Generator -----------------------------------------------------------
banner("Stage 5: DiseaseAwareCAMGenerator")
from drsa_net.model.cam_generator import DiseaseAwareCAMGenerator

cam_gen = DiseaseAwareCAMGenerator(
    cam_threshold=cfg.cam_threshold,
    propagation_alpha=cfg.cam_propagation_alpha,
    propagation_steps=cfg.cam_propagation_steps,
).to(DEVICE)

adj_bool = (local_mask == 0.0).float()
seg_ds   = info['seg_ds']

cam_out = cam_gen(
    rollout_scores=rollout,
    ws_map=ws_t,
    felz2=felz2_t,
    clahe=clahe_t,
    segments_ds=seg_ds,
    adj_matrix=adj_bool,
    target_size=(H, W),
)
check(cam_out['raw_cam'].shape        == (B, 1, H, W), f"Raw CAM: {cam_out['raw_cam'].shape}")
check(cam_out['grown_cam'].shape      == (B, 1, H, W), f"Grown CAM: {cam_out['grown_cam'].shape}")
check(cam_out['propagated_cam'].shape == (B, 1, H, W), f"Prop CAM: {cam_out['propagated_cam'].shape}")
check(cam_out['raw_cam'].min() >= 0.0,             f"CAM min >= 0: {cam_out['raw_cam'].min().item():.4f}")
check(cam_out['raw_cam'].max() <= 1.0 + 1e-5,     f"CAM max <= 1: {cam_out['raw_cam'].max().item():.4f}")

# -- Full DRSANet forward ----------------------------------------------------
banner("Stage 6: Full DRSANet forward pass")
from drsa_net.model.drsa_net import DRSANet

model = DRSANet(cfg).to(DEVICE)
print(model.summary())

out = model(rgb_t, clahe_t, felz1_t, felz2_t, ws_t, seg_map, return_cam=True)
check('cls_logits'     in out, "cls_logits present")
check('propagated_cam' in out, "propagated_cam present")
check('rollout'        in out, "rollout present")
check(out['cls_logits'].shape == (B, cfg.num_classes),
      f"Full model logits: {out['cls_logits'].shape}")

# -- Gradient flow -----------------------------------------------------------
banner("Stage 7: Gradient flow")
labels = torch.randint(0, cfg.num_classes, (B,), device=DEVICE)
loss   = nn.CrossEntropyLoss()(out['cls_logits'], labels)
loss  += out['propagated_cam'].mean()
loss.backward()

no_grad = [n for n, p in model.named_parameters()
           if p.requires_grad and p.grad is None]
has_nan = [n for n, p in model.named_parameters()
           if p.grad is not None and p.grad.isnan().any()]
check(len(no_grad) == 0, f"All params have gradients (missing: {no_grad[:3]})")
check(len(has_nan) == 0, f"No NaN gradients (NaN in: {has_nan[:3]})")

# -- Losses ------------------------------------------------------------------
banner("Stage 8: DRSALoss")
from drsa_net.training.losses import DRSALoss
from drsa_net.model.superpixel_tokenizer import build_adjacency_matrix

criterion = DRSALoss(cfg).to(DEVICE)
model.zero_grad()
out1 = model(rgb_t, clahe_t, felz1_t, felz2_t, ws_t, seg_map, return_cam=True)
out2 = model(rgb_t, clahe_t, felz1_t, felz2_t, ws_t, seg_map, return_cam=True)

N_sp2    = out1['tokens_out'].shape[1]
adj_list = []
for b_idx in range(B):
    _, remap = torch.unique(out1['seg_ds'][b_idx], return_inverse=True)
    remap    = remap.clamp(0, N_sp2 - 1)
    adj_list.append(build_adjacency_matrix(remap, N_sp2, k_hop=1).float())
adj = torch.stack(adj_list).to(DEVICE)

all_losses = criterion(out1, out2, labels, adj, return_components=True)
check('total' in all_losses, "total loss present")
check(not all_losses['total'].isnan(),
      f"No NaN in total loss: {all_losses['total'].item():.4f}")
print("  Loss breakdown:")
for k, v in all_losses.items():
    print(f"    {k:<22} {v.item():.6f}")

banner("ALL CHECKS PASSED")
print("  DRSA-Net architecture is fully functional.\n")
