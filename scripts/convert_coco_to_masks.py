import os
import sys
from pathlib import Path

# Add project root to PYTHONPATH
root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import json
import numpy as np
import cv2
from PIL import Image, ImageDraw
from pathlib import Path
from tqdm import tqdm

def convert_to_masks(json_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Mapping from String Label (VIA) to Stage 6 Model Class ID
    # Stage 6 uses: ["BG"] + ["BACT", "DML", "HLTY", "PML", "SBL", "SPW", "VIRL", "WLBL"]
    class_map = {
        "BG": 0,
        "BACT": 1,
        "DML": 2,
        "HLTY": 3,
        "PML": 4,
        "SBL": 5,
        "SPW": 6,
        "VIRL": 7,
        "WLBL": 8
    }
    
    # Mapping from COCO Category ID to Stage 6 Model Class ID
    # COCO JSON IDs: 1:HLTY, 2:BACT, 3:DML, 4:PML, 5:SBL, 6:SPW, 7:VIRL, 8:WLBL
    coco_id_to_model_id = {
        1: 3, # HLTY -> 3
        2: 1, # BACT -> 1
        3: 2, # DML -> 2
        4: 4, # PML -> 4
        5: 5, # SBL -> 5
        6: 6, # SPW -> 6
        7: 7, # VIRL -> 7
        8: 8  # WLBL -> 8
    }

    processed_count = 0
    
    for jp in json_paths:
        with open(jp, 'r') as f:
            data = json.load(f)
            
        if isinstance(data, dict) and "images" in data and "annotations" in data:
            print(f"[INFO] Processing COCO file: {jp.name}")
            images = {img['id']: img for img in data['images']}
            annos_by_img = {}
            for anno in data['annotations']:
                img_id = anno['image_id']
                if img_id not in annos_by_img: annos_by_img[img_id] = []
                annos_by_img[img_id].append(anno)
                
            for img_id, img_info in images.items():
                file_name = img_info['file_name']
                width, height = img_info['width'], img_info['height']
                mask = np.zeros((height, width), dtype=np.uint8)
                
                img_annos = annos_by_img.get(img_id, [])
                img_annos.sort(key=lambda x: coco_id_to_model_id.get(x.get('category_id', 0), 0))
                
                for anno in img_annos:
                    coco_cat_id = anno.get('category_id')
                    if coco_cat_id is None: continue
                    model_cat_id = coco_id_to_model_id.get(coco_cat_id, 0)
                    
                    segmentation = anno.get('segmentation', [])
                    for poly in segmentation:
                        if len(poly) < 6: continue
                        poly_np = np.array(poly).reshape(-1, 2).astype(np.int32)
                        cv2.fillPoly(mask, [poly_np], int(model_cat_id))
                
                stem = Path(file_name).stem
                Image.fromarray(mask).save(os.path.join(output_dir, f"{stem}_mask.png"))
                processed_count += 1
                
        elif isinstance(data, dict):
            print(f"[INFO] Processing VIA file: {jp.name}")
            for key, val in data.items():
                if not isinstance(val, dict) or "filename" not in val: continue
                file_name = val['filename']
                regions = val.get('regions', [])
                if not regions: continue
                
                img_path = find_image(file_name)
                if not img_path: continue
                with Image.open(img_path) as img_pil:
                    width, height = img_pil.size
                
                mask = np.zeros((height, width), dtype=np.uint8)
                
                if isinstance(regions, dict): region_list = regions.values()
                else: region_list = regions
                
                for region in region_list:
                    if not isinstance(region, dict): continue
                    shape = region.get('shape_attributes', {})
                    attrs = region.get('region_attributes', {})
                    label = attrs.get('label', 'BG')
                    cat_id = class_map.get(label, 0)
                    
                    if shape.get('name') == 'polygon':
                        xs = shape.get('all_points_x', [])
                        ys = shape.get('all_points_y', [])
                        poly_np = np.stack([xs, ys], axis=1).astype(np.int32)
                        cv2.fillPoly(mask, [poly_np], int(cat_id))
                
                stem = Path(file_name).stem
                Image.fromarray(mask).save(os.path.join(output_dir, f"{stem}_mask.png"))
                processed_count += 1

    print(f"[OK] Generated {processed_count} masks in {output_dir}")

def find_image(file_name):
    # Search in common locations
    search_dirs = [
        Path(r"d:\gemma4\segmentation_lattuce-desease\Lettuce_disease_datasets_split\validation"),
        Path(r"d:\gemma4\segmentation_lattuce-desease\Lettuce_disease_datasets_split\train"),
        Path(r"d:\gemma4\segmentation_lattuce-desease\Lettuce_disease_datasets_split\test")
    ]
    for d in search_dirs:
        # Search recursively
        found = list(d.rglob(file_name))
        if found:
            return found[0]
    return None

if __name__ == "__main__":
    base_path = Path(r"d:\gemma4\segmentation_lattuce-desease")
    label_dir = base_path / "Lettuce_disease_datasets_split" / "validation" / "Manual_labels"
    output_mask_dir = base_path / "lettuce_ssl_segmentation_lab" / "manual_validation_masks"
    
    json_files = list(label_dir.glob("*.json"))
    convert_to_masks(json_files, output_mask_dir)
