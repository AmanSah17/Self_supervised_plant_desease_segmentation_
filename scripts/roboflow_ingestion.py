import os
import json
import shutil
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from PIL import Image

def ingest_roboflow(roboflow_root, output_root):
    roboflow_root = Path(roboflow_root)
    output_root = Path(output_root)
    
    splits = ["train", "valid", "test"]
    mapping = {"valid": "validation"} # Match pipeline split naming
    
    os.makedirs(output_root, exist_ok=True)
    
    for split in splits:
        src_split_dir = roboflow_root / split
        if not src_split_dir.exists():
            print(f"[WARN] Split {split} not found in {roboflow_root}")
            continue
            
        target_split = mapping.get(split, split)
        image_target_dir = output_root / target_split / "Leaf-Diseases"
        mask_target_dir = output_root / "Manual_labels" / target_split
        
        os.makedirs(image_target_dir, exist_ok=True)
        os.makedirs(mask_target_dir, exist_ok=True)
        
        # Load annotations
        anno_path = src_split_dir / "_annotations.coco.json"
        if not anno_path.exists():
            print(f"[ERROR] No annotations found for {split}")
            continue
            
        with open(anno_path, 'r') as f:
            data = json.load(f)
            
        images = {img['id']: img for img in data['images']}
        annos_by_img = {}
        for anno in data['annotations']:
            img_id = anno['image_id']
            if img_id not in annos_by_img: annos_by_img[img_id] = []
            annos_by_img[img_id].append(anno)
            
        print(f"[INFO] Ingesting {target_split}...")
        for img_id, img_info in tqdm(images.items()):
            file_name = img_info['file_name']
            width, height = img_info['width'], img_info['height']
            src_img_path = src_split_dir / file_name
            
            if not src_img_path.exists():
                continue
                
            # Copy image
            target_img_path = image_target_dir / file_name
            shutil.copy2(src_img_path, target_img_path)
            
            # Generate mask (Binary: 0 for BG, 1 for Disease)
            # Note: We use 1 for 'Leaf-Diseases' to match a simplified mapping.
            # In Stage 8 SFT, the model usually expects 0-N classes.
            mask = np.zeros((height, width), dtype=np.uint8)
            img_annos = annos_by_img.get(img_id, [])
            
            for anno in img_annos:
                segmentation = anno.get('segmentation', [])
                for poly in segmentation:
                    if len(poly) < 6: continue
                    poly_np = np.array(poly).reshape(-1, 2).astype(np.int32)
                    # We fill with 1 (Disease). HLTY will be 0 (BG) for this binary task.
                    # Or we could use the original pipeline IDs if we wanted to mimic 8 classes.
                    # But since they are all one class, 1 is best.
                    cv2.fillPoly(mask, [poly_np], 1)
            
            mask_stem = Path(file_name).stem
            mask_target_path = mask_target_dir / f"{mask_stem}_mask.png"
            Image.fromarray(mask).save(mask_target_path)

    print(f"[OK] Ingestion complete. Data at: {output_root}")

if __name__ == "__main__":
    roboflow_path = r"d:\gemma4\segmentation_lattuce-desease\Leaf Disease Segmentation.v1i.coco-segmentation"
    output_path = r"d:\gemma4\segmentation_lattuce-desease\Roboflow_Dataset"
    ingest_roboflow(roboflow_path, output_path)
