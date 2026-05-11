"""
End-to-End Felzenszwalb Pipeline
Automates hyperparameter tuning, GPU-accelerated parallel processing, 
checkpointing, and visualization.
"""

import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from felzenszwalb_hyperparameter_tuning import (
    FelzenszwalbTuner,
    HyperparameterConfig,
    analyze_results,
)
from felzenszwalb_segmentation_gpu import (
    FelzenszwalbConfig,
    FelzenszwalbSegmenter,
    MaskProcessor,
    collect_image_paths,
    load_rgb_image,
)


def run_phase_1_tuning() -> Dict:
    """Run hyperparameter tuning on a subset to find optimal parameters."""
    print("\n" + "="*70)
    print("PHASE 1: AUTOMATED HYPERPARAMETER TUNING")
    print("="*70)
    
    # Configure tuning - test smaller range for speed or use default
    config = HyperparameterConfig(
        samples_per_class=2,  # Keep it small for quick tuning
        use_gpu=True
    )
    
    tuner = FelzenszwalbTuner(config)
    results_df = tuner.grid_search()
    
    analysis = analyze_results(results_df)
    
    # We will pick best_uniformity as the optimal parameters
    best_params = analysis['best_uniformity']
    print("\nOptimal Parameters Found (Based on Best Uniformity):")
    print(f"Scale: {best_params['scale']}, Sigma: {best_params['sigma']}, Min Size: {best_params['min_size']}")
    print(f"Score: {best_params['score']:.4f}")
    
    # Save the full analysis just in case
    output_dir = Path(config.output_base)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "pipeline_recommendations_1.json", 'w') as f:
        json.dump(analysis, f, indent=2)
        
    return best_params


def process_image_worker(
    image_path: Path,
    class_name: str,
    split: str,
    config: FelzenszwalbConfig
) -> dict:
    """Worker function to process a single image, for use with multiprocessing."""
    start_time = time.time()
    try:
        # Initialize segmenter and mask processor inside worker to avoid pickle issues
        segmenter = FelzenszwalbSegmenter(config)
        mask_processor = MaskProcessor(config)
        
        img_rgb = load_rgb_image(image_path)
        height, width = img_rgb.shape[:2]
        
        segments = segmenter.segment_image(
            img_rgb,
            scale=config.felz_scale,
            sigma=config.felz_sigma,
            min_size=config.felz_min_size,
            enhance_edges=config.apply_edge_enhancement,
        )
        
        segments = segmenter.merge_tiny_segments(
            segments,
            img_rgb,
            min_size=max(config.felz_min_size - 2, 5),
        )
        
        saved_paths = mask_processor.save_masks(
            image_path=image_path,
            class_name=class_name,
            split=split,
            segments=segments,
            scale=config.felz_scale,
            sigma=config.felz_sigma,
            min_size=config.felz_min_size,
        )
        
        elapsed = time.time() - start_time
        
        return {
            'split': split,
            'class_name': class_name,
            'image_path': str(image_path),
            'status': 'success',
            'num_segments': int(segments.max() + 1),
            'colored_mask_path': str(saved_paths.get('colored', '')),
            'elapsed_seconds': round(elapsed, 3),
        }
    except Exception as exc:
        return {
            'split': split,
            'class_name': class_name,
            'image_path': str(image_path),
            'status': 'failed',
            'error': str(exc),
            'elapsed_seconds': round(time.time() - start_time, 3),
        }


def run_phase_2_segmentation(optimal_params: Dict) -> List[dict]:
    """Run mass generation with multiprocessing and checkpointing."""
    print("\n" + "="*70)
    print("PHASE 2: PARALLEL MASK GENERATION WITH CHECKPOINTING")
    print("="*70)
    
    config = FelzenszwalbConfig(
        splits=("train",),  # Only process train set
        felz_scale=optimal_params['scale'],
        felz_sigma=optimal_params['sigma'],
        felz_min_size=optimal_params['min_size'],
        use_gpu=True,
    )
    
    dataset_base = Path(config.dataset_base)
    output_base = Path(config.output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    
    checkpoint_file = output_base / "pipeline_checkpoint.json"
    processed_images = {}
    
    # Load checkpoint if exists
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                processed_images = json.load(f)
            print(f"Loaded checkpoint with {len(processed_images)} processed images.")
        except json.JSONDecodeError:
            print("Checkpoint file corrupted. Starting fresh.")
    
    all_results = list(processed_images.values())
    
    for split in config.splits:
        split_dir = dataset_base / split
        if not split_dir.exists():
            print(f"Split dir not found: {split_dir}")
            continue
            
        class_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
        
        for class_dir in class_dirs:
            class_name = class_dir.name
            
            # Collect images for this class only
            image_paths = []
            for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
                image_paths.extend(class_dir.glob(pattern))
            
            # Sort to ensure consistent order
            image_paths = sorted(list(set(image_paths)))
            
            # Filter pending items
            pending_items = []
            for img_path in image_paths:
                if str(img_path) not in processed_images or processed_images[str(img_path)]['status'] != 'success':
                    pending_items.append((class_name, img_path))
                    
            if not pending_items:
                print(f"[{split}/{class_name}] All images already processed.")
                continue
                
            print(f"\nProcessing {len(pending_items)} remaining images from {split}/{class_name} split...")
            
            # Use multithreading pool
            max_workers = min(os.cpu_count() or 4, 8) # Max 8 workers to not overload RAM/GPU
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_image_worker, path, cls_name, split, config): path
                    for cls_name, path in pending_items
                }
                
                progress = tqdm(total=len(pending_items), desc=f"{class_name}", unit="img", ncols=100)
                
                for future in as_completed(futures):
                    result = future.result()
                    img_path_str = result['image_path']
                    
                    # Update tracking
                    processed_images[img_path_str] = result
                    all_results.append(result)
                    
                    # Update checkpoint
                    with open(checkpoint_file, 'w') as f:
                        json.dump(processed_images, f, indent=2)
                        
                    if result['status'] == 'success':
                        progress.set_postfix({'status': 'ok', 'segments': result['num_segments']})
                    else:
                        progress.set_postfix({'status': 'failed'})
                    
                    progress.update(1)
                
                progress.close()
            
    # Save final summary
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(output_base / "pipeline_summary.csv", index=False)
    
    # Generate and log class-wise analytics
    if not summary_df.empty and 'status' in summary_df.columns:
        success_df = summary_df[summary_df['status'] == 'success']
        if not success_df.empty:
            analytics_df = success_df.groupby('class_name').agg(
                mean_segments=('num_segments', 'mean'),
                min_segments=('num_segments', 'min'),
                max_segments=('num_segments', 'max'),
                mean_time_sec=('elapsed_seconds', 'mean')
            ).reset_index()
            analytics_path = output_base / "class_wise_analytics.csv"
            analytics_df.to_csv(analytics_path, index=False)
            print("\n" + "-"*50)
            print("Class-wise Segmentation Analytics:")
            print("-"*50)
            print(analytics_df.to_string(index=False))
            print("-"*50 + "\n")
            
    print(f"\nPhase 2 Complete. Processed {len(all_results)} images total.")
    return all_results


def run_phase_3_visualization(results: List[dict]):
    """Visualize 10 random samples."""
    print("\n" + "="*70)
    print("PHASE 3: VISUALIZATION & COMPARISON")
    print("="*70)
    
    # Filter for successful runs that have a colored mask path
    success_results = [r for r in results if r['status'] == 'success' and r.get('colored_mask_path')]
    
    if not success_results:
        print("No successful images to visualize.")
        return
        
    num_samples = min(10, len(success_results))
    samples = random.sample(success_results, num_samples)
    
    print(f"Plotting {num_samples} random samples...")
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]
        
    for i, sample in enumerate(samples):
        orig_path = sample['image_path']
        mask_path = sample['colored_mask_path']
        
        # Load images
        orig_img = cv2.imread(orig_path)
        if orig_img is not None:
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            
        mask_img = cv2.imread(mask_path)
        if mask_img is not None:
            mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
            
        ax_orig = axes[i][0]
        ax_mask = axes[i][1]
        
        if orig_img is not None:
            ax_orig.imshow(orig_img)
        ax_orig.set_title(f"Original: {sample['class_name']}")
        ax_orig.axis('off')
        
        if mask_img is not None:
            ax_mask.imshow(mask_img)
        ax_mask.set_title(f"Generated Mask\nSegs: {sample.get('num_segments', 'N/A')}")
        ax_mask.axis('off')
        
    plt.tight_layout()
    viz_path = Path("pipeline_visualization.png").resolve()
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {viz_path}")


def main():
    start_total = time.time()
    
    # 1. Tune Hyperparameters
    optimal_params = run_phase_1_tuning()
    
    # 2. Parallel Generation
    results = run_phase_2_segmentation(optimal_params)
    
    # 3. Visualization
    run_phase_3_visualization(results)
    
    total_time = time.time() - start_total
    print("\n" + "="*70)
    print(f"PIPELINE COMPLETE! Total Time: {total_time/60:.2f} minutes.")
    print("="*70 + "\n")


if __name__ == '__main__':
    # Needed for cross-platform multiprocessing safety
    import multiprocessing
    multiprocessing.freeze_support()
    main()
