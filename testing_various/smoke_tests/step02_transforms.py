"""
STEP 2: Transforms & Augmentation verification
Loads one real image per split, applies CLAHE, Watershed, and 
SynchronizedTransform. Saves augmented outputs as PNGs for visual inspection.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding='utf-8')

from drsa_net.data.transforms import (
    apply_clahe, apply_watershed, SynchronizedTransform, to_tensor_dict
)
from drsa_net.config import DRSAConfig

cfg = DRSAConfig()
cfg.img_size = (256, 256)

OUT_DIR = Path("drsa_net_output/step02_transform_samples")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_DIR = Path("Lettuce_disease_datasets_split/train")
FELZ_DIR  = Path("felzenszwalb_masks_output/train")

# Pick first available image from each class
classes = ['BACT', 'DML', 'HLTY', 'PML', 'SBL', 'SPW', 'VIRL', 'WLBL']
transform_aug  = SynchronizedTransform(img_size=cfg.img_size, augment=True)
transform_val  = SynchronizedTransform(img_size=cfg.img_size, augment=False)

print("="*55)
print("STEP 2: Transforms & Augmentation")
print("="*55)

for cls in classes:
    img_dir  = TRAIN_DIR / cls
    mask_dir = FELZ_DIR  / cls
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'})
    if not imgs:
        print(f"  [SKIP] {cls} - no images")
        continue

    img_path = imgs[0]
    img_bgr  = cv2.imread(str(img_path))
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W     = img_rgb.shape[:2]

    # --- CLAHE ---
    clahe_img = apply_clahe(img_rgb, clip_limit=cfg.clahe_clip_limit,
                             tile_grid=cfg.clahe_tile_grid)

    # --- Watershed ---
    ws_map = apply_watershed(img_rgb,
                              compactness=cfg.watershed_compactness,
                              min_size=cfg.watershed_min_size)

    # --- Load Felz masks ---
    raw_masks = list(mask_dir.glob(f"{img_path.stem}*_raw.png"))
    bnd_masks = list(mask_dir.glob(f"{img_path.stem}*_boundary.png"))

    def load_gray_float(p, h, w):
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if p else None
        if m is None:
            return np.zeros((h, w), dtype=np.float32)
        return m.astype(np.float32) / 255.0

    felz1 = load_gray_float(raw_masks[0] if raw_masks else None, H, W)
    felz2 = load_gray_float(bnd_masks[0] if bnd_masks else None, H, W)

    # Reconstruct integer segments from felz1
    quant = (felz1 * 255).round().astype(np.uint8)
    uniq  = np.unique(quant)
    mapping = {v: i for i, v in enumerate(uniq)}
    segments = np.vectorize(mapping.get)(quant).astype(np.int32)

    data = {
        'rgb':       img_rgb,
        'clahe':     clahe_img,
        'felz1':     felz1,
        'felz2':     felz2,
        'watershed': ws_map,
        'segments':  segments,
    }

    # Apply with augmentation
    data_aug = transform_aug(data.copy())
    # Apply without augmentation (val mode)
    data_val = transform_val(data.copy())

    # Convert to tensors
    t_aug = to_tensor_dict(data_aug)
    t_val = to_tensor_dict(data_val)

    # Save visual samples
    def save_rgb_tensor(t, path):
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        arr  = t.permute(1,2,0).numpy()
        arr  = (arr * std + mean).clip(0,1)
        cv2.imwrite(str(path), cv2.cvtColor((arr*255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    def save_float_map(t, path):
        arr = t.squeeze().numpy()
        arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-6) * 255).astype(np.uint8)
        cv2.imwrite(str(path), arr)

    cls_dir_out = OUT_DIR / cls
    cls_dir_out.mkdir(exist_ok=True)

    save_rgb_tensor(t_aug['rgb'],       cls_dir_out / "rgb_aug.png")
    save_rgb_tensor(t_aug['clahe'],     cls_dir_out / "clahe_aug.png")
    save_float_map(t_aug['felz1'],      cls_dir_out / "felz1_aug.png")
    save_float_map(t_aug['felz2'],      cls_dir_out / "felz2_aug.png")
    save_float_map(t_aug['watershed'],  cls_dir_out / "watershed_aug.png")

    print(f"  {cls:<8}  img={img_rgb.shape}  segs={segments.max()+1}"
          f"  ws_unique={len(np.unique((ws_map*255).astype(np.uint8)))}"
          f"  tensors: rgb={tuple(t_aug['rgb'].shape)}"
          f"  felz1={tuple(t_aug['felz1'].shape)}")

print()
print(f"  Visual outputs saved to: {OUT_DIR}")
print("[STEP 2 COMPLETE]")
