"""
STEP 1: Dataset discovery
Scans all splits and classes, counts images and Felzenszwalb mask pairs.
Saves a CSV report: drsa_net_output/step01_dataset_report.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding='utf-8')

DATASET_BASE = Path("Lettuce_disease_datasets_split")
FELZ_BASE    = Path("felzenszwalb_masks_output")
SPLITS       = ["train", "validation", "test"]
OUT_DIR      = Path("drsa_net_output")
OUT_DIR.mkdir(exist_ok=True)

print("="*55)
print("STEP 1: Dataset Discovery")
print("="*55)

rows = []
total_imgs   = 0
total_paired = 0

for split in SPLITS:
    split_dir = DATASET_BASE / split
    felz_dir  = FELZ_BASE    / split
    if not split_dir.exists():
        print(f"  [SKIP] {split} — not found")
        continue

    for cls_dir in sorted(split_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls = cls_dir.name
        imgs = sorted(p for p in cls_dir.iterdir()
                      if p.suffix.lower() in {'.jpg','.jpeg','.png'})
        mask_dir = felz_dir / cls

        paired = 0
        for img in imgs:
            raw_masks = list(mask_dir.glob(f"{img.stem}*_raw.png")) if mask_dir.exists() else []
            if raw_masks:
                paired += 1

        rows.append({
            'split': split, 'class': cls,
            'images': len(imgs), 'paired_masks': paired,
            'mask_coverage': f"{100*paired/max(len(imgs),1):.1f}%"
        })
        total_imgs   += len(imgs)
        total_paired += paired
        print(f"  {split:<12} {cls:<8}  imgs={len(imgs):>4}  paired={paired:>4}  coverage={100*paired/max(len(imgs),1):.1f}%")

csv_path = OUT_DIR / "step01_dataset_report.csv"
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['split','class','images','paired_masks','mask_coverage'])
    writer.writeheader()
    writer.writerows(rows)

print()
print(f"  Total images   : {total_imgs}")
print(f"  Total paired   : {total_paired}  ({100*total_paired/max(total_imgs,1):.1f}% coverage)")
print(f"  Report saved   : {csv_path}")
print("[STEP 1 COMPLETE]")
