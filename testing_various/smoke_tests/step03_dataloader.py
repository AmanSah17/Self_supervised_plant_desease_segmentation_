"""
STEP 3: DataLoader creation + checkpoint for resumability
- Caches sample index to disk (JSON) so reinit is instant on resume
- Uses tqdm for progress bars
- Uses Numba JIT to accelerate segment remapping
- Fetches batches to CUDA to verify throughput
- Tuned for GTX 1650 (4 GB VRAM)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from numba import njit

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drsa_net.config import DRSAConfig
from drsa_net.data.transforms import (
    apply_clahe, apply_watershed, SynchronizedTransform, to_tensor_dict
)

# ── Numba Accelerated Remapping ───────────────────────────────────────────
@njit
def fast_remap_segments(quant_img, unique_vals):
    """
    Accelerated segment remapping using Numba.
    Replaces slow np.vectorize mapping.
    """
    h, w = quant_img.shape
    out = np.zeros((h, w), dtype=np.int32)
    # Create a lookup table for faster access
    # Since quant_img is uint8, max value is 255
    lut = np.zeros(256, dtype=np.int32)
    for i in range(len(unique_vals)):
        lut[int(unique_vals[i])] = i
    
    for y in range(h):
        for x in range(w):
            out[y, x] = lut[quant_img[y, x]]
    return out

# ── Config (GTX 1650 safe) ────────────────────────────────────────────────
cfg = DRSAConfig()
cfg.img_size               = (256, 256)
cfg.batch_size             = 4
cfg.num_workers            = 0     
cfg.max_superpixels        = 256
cfg.embed_dim              = 128
cfg.branch_channels        = 32
cfg.num_transformer_layers = 4
cfg.num_attention_heads    = 4
cfg.use_amp                = True
cfg.clahe_clip_limit       = 10.0
cfg.validate()

OUT_DIR    = Path("drsa_net_output")
OUT_DIR.mkdir(exist_ok=True)
INDEX_PATH = OUT_DIR / "sample_index_cache.json"   
CKPT_PATH  = OUT_DIR / "step03_dataloader_checkpoint.json"

print("="*55)
print("STEP 3: DataLoader Creation (CUDA + Numba + TQDM)")
print("="*55)
print(f"  PyTorch     : {torch.__version__}")
print(f"  CUDA        : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  Device      : {torch.cuda.get_device_name(0)}")
print(f"  Batch Size  : {cfg.batch_size}")
print()

# ── Fast sample-index builder ─────────────────────────────────────────────
DATASET_BASE = Path(cfg.dataset_base)
FELZ_BASE    = Path(cfg.felz_masks_base)
SPLITS       = ('train', 'validation', 'test')

def build_sample_index(force_rebuild: bool = False) -> dict:
    if INDEX_PATH.exists() and not force_rebuild:
        print(f"  [CACHE] Loading index from {INDEX_PATH.name}...")
        with open(INDEX_PATH) as f:
            return json.load(f)

    print("  [INIT] Building sample index (Scanning filesystem)...")
    index = {}
    for split in SPLITS:
        split_dir = DATASET_BASE / split
        felz_dir  = FELZ_BASE    / split
        if not split_dir.exists(): continue

        samples = []
        # Get all class directories
        cls_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        
        # tqdm progress bar for classes within a split
        for cls_dir in tqdm(cls_dirs, desc=f"  Scanning {split:<10}", leave=False):
            cls = cls_dir.name
            if cls not in cfg.class_to_idx: continue
            label = cfg.class_to_idx[cls]
            mask_dir = felz_dir / cls

            img_paths = sorted([str(p) for p in cls_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])

            for img_str in img_paths:
                img_path = Path(img_str)
                stem = img_path.stem
                raw_mask = mask_dir / f"{stem}_raw.png"
                bnd_mask = mask_dir / f"{stem}_boundary.png"

                samples.append({
                    'image_path': img_str,
                    'felz_raw':   str(raw_mask) if raw_mask.exists() else None,
                    'felz_bnd':   str(bnd_mask) if bnd_mask.exists() else None,
                    'label':      label,
                    'class_name': cls,
                })
        index[split] = samples
        print(f"    ✓ {split:<11}: {len(samples)} samples found")

    with open(INDEX_PATH, 'w') as f:
        json.dump(index, f)
    return index

# ── Dataset with Numba Acceleration ───────────────────────────────────────
class FastDRSADataset(Dataset):
    def __init__(self, samples: list, cfg: DRSAConfig, augment: bool = True):
        self.samples = samples
        self.cfg     = cfg
        self.transform = SynchronizedTransform(img_size=cfg.img_size, augment=augment)

    def __len__(self): return len(self.samples)

    def _load_gray_float(self, path_str, h, w):
        if not path_str: return np.zeros((h, w), dtype=np.float32)
        m = cv2.imread(path_str, cv2.IMREAD_GRAYSCALE)
        if m is None: return np.zeros((h, w), dtype=np.float32)
        return m.astype(np.float32) / 255.0

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_bgr = cv2.imread(s['image_path'])
        if img_bgr is None: raise FileNotFoundError(s['image_path'])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w    = img_rgb.shape[:2]

        # Stage 1: Representation transforms
        clahe_img = apply_clahe(img_rgb, clip_limit=self.cfg.clahe_clip_limit)
        ws_map    = apply_watershed(img_rgb)
        felz1 = self._load_gray_float(s['felz_raw'], h, w)
        felz2 = self._load_gray_float(s['felz_bnd'], h, w)

        # Stage 2: Numba-accelerated segment remapping
        quant    = (felz1 * 255).round().astype(np.uint8)
        uniq     = np.unique(quant)
        segments = fast_remap_segments(quant, uniq)

        data = {
            'rgb': img_rgb, 'clahe': clahe_img, 
            'felz1': felz1, 'felz2': felz2, 
            'watershed': ws_map, 'segments': segments
        }
        data = self.transform(data)
        t    = to_tensor_dict(data)

        return {
            **t,
            'label': torch.tensor(s['label'], dtype=torch.int64),
            'image_path': s['image_path'],
        }

    @staticmethod
    def collate_fn(batch):
        out = {}
        for key in batch[0]:
            if key == 'image_path':
                out[key] = [b[key] for b in batch]
            else:
                out[key] = torch.stack([b[key] for b in batch])
        return out

# ── Execution ─────────────────────────────────────────────────────────────
index = build_sample_index()
train_samples = index.get('train', [])
val_samples   = index.get('validation', [])

train_ds = FastDRSADataset(train_samples, cfg, augment=True)
val_ds   = FastDRSADataset(val_samples,   cfg, augment=False)

train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, 
                          num_workers=cfg.num_workers, pin_memory=True, 
                          collate_fn=FastDRSADataset.collate_fn)
val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, 
                          num_workers=cfg.num_workers, pin_memory=True, 
                          collate_fn=FastDRSADataset.collate_fn)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Verification with progress bar
print(f"\n  [VERIFY] Testing throughput on {DEVICE}...")
t0 = time.time()
num_test_batches = 5
for i, batch in enumerate(tqdm(train_loader, total=num_test_batches, desc="  Warmup batches")):
    if i >= num_test_batches: break
    # Move core tensors to CUDA
    batch_cuda = {k: v.to(DEVICE) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    if i == 0:
        print(f"    Sample Batch Shape: {batch_cuda['rgb'].shape}")

elapsed = time.time() - t0
print(f"  ✓ {num_test_batches} batches processed in {elapsed:.2f}s ({num_test_batches/elapsed:.2f} batch/s)")

# Save checkpoint
ckpt = {
    'config': cfg.__dict__,
    'dataset': {
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'class_to_idx': cfg.class_to_idx
    },
    'step': 3,
    'status': 'ready'
}
# Convert non-serializable to serializable
ckpt['config']['img_size'] = list(cfg.img_size)
if 'clahe_tile_grid' in ckpt['config']:
    ckpt['config']['clahe_tile_grid'] = list(cfg.clahe_tile_grid)

with open(CKPT_PATH, 'w') as f:
    json.dump(ckpt, f, indent=2)

print(f"\n  ✓ All checks passed. Loader ready for training.")
print(f"  ✓ Checkpoint saved to: {CKPT_PATH}")
print("[STEP 3 COMPLETE]")
