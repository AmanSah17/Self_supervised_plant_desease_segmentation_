import os
import cv2
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(r'd:\gemma4\segmentation_lattuce-desease')
sys.path.append(str(project_root))

from drsa_net.data.transforms import apply_clahe, apply_watershed
from lettuce_ssl_segmentation_lab.utils.numba_ops import fast_compute_exg

def main():
    image_path = project_root / 'Lettuce_disease_datasets_split' / 'train' / 'BACT' / 'BACT_10.jpg'
    output_dir = project_root / 'Report' / 'preprocessing_steps'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Original RGB
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"Error: Could not load image at {image_path}")
        return
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb_256 = cv2.resize(img_rgb, (256, 256))
    
    cv2.imwrite(str(output_dir / '01_original_rgb.png'), cv2.cvtColor(img_rgb_256, cv2.COLOR_RGB2BGR))
    print("Saved 01_original_rgb.png")

    # 2. CLAHE
    img_clahe = apply_clahe(img_rgb_256)
    cv2.imwrite(str(output_dir / '02_clahe.png'), cv2.cvtColor(img_clahe, cv2.COLOR_RGB2BGR))
    print("Saved 02_clahe.png")

    # 3. ExG (Excess Green Index)
    # ExG formula: 2G - R - B
    # Normalize for visualization
    exg = fast_compute_exg(img_rgb_256.astype(np.float32) / 255.0)
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(str(output_dir / '03_exg.png'), exg_norm)
    print("Saved 03_exg.png")

    # 4. Edge Map (Sobel)
    gray = cv2.cvtColor(img_rgb_256, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = np.sqrt(gx**2 + gy**2)
    edge_norm = cv2.normalize(edge_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(str(output_dir / '04_edge_map.png'), edge_norm)
    print("Saved 04_edge_map.png")

    # 5. Felzenszwalb (Load from pre-computed)
    felz_base = project_root / 'felzenszwalb_masks_output' / 'train' / 'BACT'
    felz_raw_path = felz_base / 'BACT_10_s80_0_sg0_5_ms20_raw.png'
    felz_colored_path = felz_base / 'BACT_10_s80_0_sg0_5_ms20_colored.png'
    felz_boundary_path = felz_base / 'BACT_10_s80_0_sg0_5_ms20_boundary.png'

    if felz_raw_path.exists():
        raw = cv2.imread(str(felz_raw_path), cv2.IMREAD_GRAYSCALE)
        raw_resized = cv2.resize(raw, (256, 256), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(output_dir / '05_felz_raw.png'), raw_resized)
        print("Saved 05_felz_raw.png")
    
    if felz_colored_path.exists():
        colored = cv2.imread(str(felz_colored_path))
        colored_resized = cv2.resize(colored, (256, 256), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(output_dir / '06_felz_colored.png'), colored_resized)
        print("Saved 06_felz_colored.png")

    if felz_boundary_path.exists():
        boundary = cv2.imread(str(felz_boundary_path), cv2.IMREAD_GRAYSCALE)
        boundary_resized = cv2.resize(boundary, (256, 256), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(output_dir / '07_felz_boundary.png'), boundary_resized)
        print("Saved 07_felz_boundary.png")

    # 6. Watershed
    ws_norm = apply_watershed(img_rgb_256)
    ws_u8 = (ws_norm * 255).astype(np.uint8)
    cv2.imwrite(str(output_dir / '08_watershed.png'), ws_u8)
    print("Saved 08_watershed.png")

if __name__ == '__main__':
    main()
