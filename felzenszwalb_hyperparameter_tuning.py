"""
Felzenszwalb Hyperparameter Tuning Script
Performs grid search over Felzenszwalb parameters with multiprocessing and checkpointing.
Evaluates and compares different parameter combinations.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb
from tqdm import tqdm


@dataclass
class HyperparameterConfig:
    """Configuration for hyperparameter tuning"""
    dataset_base: str = "Lettuce_disease_datasets_split"
    output_base: str = "felzenszwalb_hyperparameter_tuning"
    test_split: str = "validation"
    
    # Grid search ranges (Reduced complexity for detailed masks)
    scale_range: Tuple[float, ...] = (50.0, 80.0)
    sigma_range: Tuple[float, ...] = (0.8,)
    min_size_range: Tuple[int, ...] = (10, 15)
    
    # Sampling
    samples_per_class: int = 2  # Images to test per class
    max_classes: Optional[int] = None  # Limit classes for faster tuning
    
    # Device
    use_gpu: bool = True


class SegmentationMetrics:
    """Compute metrics for segmentation quality"""
    
    @staticmethod
    def compute_segment_statistics(segments: np.ndarray) -> Dict[str, float]:
        num_segments = int(segments.max()) + 1
        ids, counts = np.unique(segments, return_counts=True)
        return {
            'num_segments': float(num_segments),
            'mean_segment_size': float(np.mean(counts)),
            'std_segment_size': float(np.std(counts)),
            'min_segment_size': float(np.min(counts)),
            'max_segment_size': float(np.max(counts)),
            'size_uniformity': float(np.std(counts) / (np.mean(counts) + 1e-6)),
        }
    
    @staticmethod
    def compute_boundary_metrics(segments: np.ndarray, img_rgb: np.ndarray) -> Dict[str, float]:
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        l_channel = img_lab[..., 0] / 255.0
        a_channel = (img_lab[..., 1] - 128.0) / 127.0
        b_channel = (img_lab[..., 2] - 128.0) / 127.0
        
        edge_map = 0.6 * sobel(l_channel) + 0.2 * sobel(a_channel) + 0.2 * sobel(b_channel)
        edge_map = (edge_map - edge_map.min()) / (np.ptp(edge_map) + 1e-6)
        
        boundaries = ndi.binary_dilation(ndi.binary_erosion(segments)) ^ ndi.binary_erosion(segments)
        boundary_edges = edge_map[boundaries > 0]
        
        return {
            'boundary_edge_strength': float(np.mean(boundary_edges)) if len(boundary_edges) > 0 else 0.0,
            'boundary_edge_std': float(np.std(boundary_edges)) if len(boundary_edges) > 0 else 0.0,
        }
    
    @staticmethod
    def compute_compactness(segments: np.ndarray) -> float:
        compactness_values = []
        for seg_id in np.unique(segments):
            mask = segments == seg_id
            area = float(np.sum(mask))
            if area == 0:
                continue
            boundaries = ndi.binary_dilation(mask) ^ mask
            perimeter = float(np.sum(boundaries))
            compactness = 4 * np.pi * area / (perimeter ** 2 + 1e-6) if perimeter > 0 else 0.0
            compactness_values.append(compactness)
        return float(np.mean(compactness_values)) if compactness_values else 0.0


def evaluate_combination_worker(
    class_name: str,
    img_path: str,
    scale: float,
    sigma: float,
    min_size: int
) -> dict:
    """Module-level worker function for multiprocessing"""
    start_time = time.time()
    try:
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Could not load image at {img_path}")
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
        
        # Segment
        segments = felzenszwalb(
            img_lab,
            scale=scale,
            sigma=sigma,
            min_size=min_size,
            channel_axis=-1,
        )
        labels, relabeled = np.unique(segments, return_inverse=True)
        relabeled = relabeled.reshape(segments.shape).astype(np.int32)
        
        # Evaluate
        metrics = SegmentationMetrics()
        stats = metrics.compute_segment_statistics(relabeled)
        boundary_metrics = metrics.compute_boundary_metrics(relabeled, img_rgb)
        compactness = metrics.compute_compactness(relabeled)
        
        elapsed = time.time() - start_time
        
        return {
            'status': 'success',
            'class_name': class_name,
            'image': Path(img_path).stem,
            'image_path': img_path,
            'scale': float(scale),
            'sigma': float(sigma),
            'min_size': float(min_size),
            'num_segments': stats['num_segments'],
            'mean_segment_size': stats['mean_segment_size'],
            'size_uniformity': stats['size_uniformity'],
            'compactness': compactness,
            'boundary_edge_strength': boundary_metrics['boundary_edge_strength'],
            'elapsed_ms': elapsed * 1000,
            'worker_pid': os.getpid()
        }
    except Exception as e:
        return {
            'status': 'failed',
            'class_name': class_name,
            'image': Path(img_path).stem,
            'image_path': img_path,
            'scale': float(scale),
            'sigma': float(sigma),
            'min_size': float(min_size),
            'error': str(e),
            'elapsed_ms': (time.time() - start_time) * 1000,
            'worker_pid': os.getpid()
        }


class FelzenszwalbTuner:
    """Hyperparameter tuning for Felzenszwalb"""
    
    def __init__(self, config: HyperparameterConfig):
        self.config = config
        self.device = torch.device("cuda" if (config.use_gpu and torch.cuda.is_available()) else "cpu")
        print(f"FelzenszwalbTuner initialized. Device accessible: {self.device}")
    
    def grid_search(self) -> pd.DataFrame:
        dataset_base = Path(self.config.dataset_base)
        test_split_dir = dataset_base / self.config.test_split
        
        if not test_split_dir.exists():
            raise FileNotFoundError(f"Test split not found: {test_split_dir}")
        
        # Collect test images
        test_images: List[Tuple[str, str]] = []
        for class_dir in sorted(test_split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            if self.config.max_classes and len(test_images) >= self.config.max_classes:
                break
            image_paths = sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.png"))
            for img_path in image_paths[:self.config.samples_per_class]:
                test_images.append((class_dir.name, str(img_path)))
        
        if not test_images:
            raise ValueError("No test images found")
            
        param_combinations = list(product(
            self.config.scale_range,
            self.config.sigma_range,
            self.config.min_size_range,
        ))
        
        total_evals = len(test_images) * len(param_combinations)
        print(f"Using {len(test_images)} images for hyperparameter tuning")
        print(f"Testing {len(param_combinations)} parameter combinations...")
        print(f"Total segmentations to run: {total_evals}")
        
        # Setup Checkpointing
        output_dir = Path(self.config.output_base)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = output_dir / "tuning_checkpoint.csv"
        
        processed_keys = set()
        results: List[Dict] = []
        
        if checkpoint_file.exists():
            df_existing = pd.read_csv(checkpoint_file)
            for _, row in df_existing.iterrows():
                if row.get('status', 'success') == 'success':
                    key = f"{row['image_path']}_{row['scale']}_{row['sigma']}_{row['min_size']}"
                    processed_keys.add(key)
                    results.append(row.to_dict())
            print(f"Loaded {len(processed_keys)} completed evaluations from checkpoint.")
            
        pending_tasks = []
        for class_name, img_path in test_images:
            for scale, sigma, min_size in param_combinations:
                key = f"{img_path}_{float(scale)}_{float(sigma)}_{float(min_size)}"
                if key not in processed_keys:
                    pending_tasks.append((class_name, img_path, scale, sigma, min_size))
                    
        if not pending_tasks:
            print("All tuning evaluations completed from checkpoint!")
            return pd.DataFrame(results)
            
        print(f"Resuming {len(pending_tasks)} pending evaluations using Multiprocessing...")
        
        # Append to CSV function
        def append_to_csv(result_dict: dict, file_path: Path, is_first: bool):
            df = pd.DataFrame([result_dict])
            df.to_csv(file_path, mode='a', header=is_first, index=False)
            
        is_first_write = not checkpoint_file.exists() or len(processed_keys) == 0
        
        max_workers = min(os.cpu_count() or 4, 8)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(evaluate_combination_worker, cls, path, s, sig, ms): (cls, path, s, sig, ms)
                for cls, path, s, sig, ms in pending_tasks
            }
            
            pbar = tqdm(total=len(pending_tasks), desc="Tuning (Multithreading)", unit="eval", ncols=100)
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result['status'] == 'success':
                        results.append(result)
                        # Analytics logging to terminal
                        tqdm.write(f"[Worker {result['worker_pid']}] Evaluated {result['image']} (s={result['scale']}, sg={result['sigma']}, ms={result['min_size']}) -> Uniformity: {result['size_uniformity']:.4f} | {result['elapsed_ms']:.1f}ms")
                    else:
                        tqdm.write(f"[Worker {result['worker_pid']}] Failed {result['image']} (s={result['scale']}, sg={result['sigma']}, ms={result['min_size']}): {result.get('error')}")
                        
                    append_to_csv(result, checkpoint_file, is_first_write)
                    if is_first_write:
                        is_first_write = False
                except Exception as e:
                    tqdm.write(f"Task failed completely: {e}")
                
                pbar.update(1)
            pbar.close()
            
        return pd.DataFrame([r for r in results if r.get('status', 'success') == 'success'])


def analyze_results(results_df: pd.DataFrame) -> Dict:
    """Analyze tuning results and provide recommendations"""
    # Clean up results_df due to possible multi-threading race conditions in CSV logging
    numeric_cols = ['scale', 'sigma', 'min_size', 'num_segments', 'mean_segment_size', 'size_uniformity', 'compactness', 'boundary_edge_strength', 'elapsed_ms']
    for col in numeric_cols:
        if col in results_df.columns:
            results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
    results_df = results_df.dropna(subset=numeric_cols)

    grouped = results_df.groupby(['scale', 'sigma', 'min_size']).agg({
        'num_segments': ['mean', 'std'],
        'mean_segment_size': ['mean', 'std'],
        'size_uniformity': 'mean',
        'compactness': 'mean',
        'boundary_edge_strength': 'mean',
        'elapsed_ms': 'mean',
    }).reset_index()
    
    best_uniformity = grouped.loc[grouped[('size_uniformity', 'mean')].idxmin()]
    best_compactness = grouped.loc[grouped[('compactness', 'mean')].idxmax()]
    best_boundary = grouped.loc[grouped[('boundary_edge_strength', 'mean')].idxmax()]
    best_speed = grouped.loc[grouped[('elapsed_ms', 'mean')].idxmin()]
    
    return {
        'best_uniformity': {
            'scale': float(best_uniformity[('scale', '')]),
            'sigma': float(best_uniformity[('sigma', '')]),
            'min_size': int(best_uniformity[('min_size', '')]),
            'score': float(best_uniformity[('size_uniformity', 'mean')]),
        },
        'best_compactness': {
            'scale': float(best_compactness[('scale', '')]),
            'sigma': float(best_compactness[('sigma', '')]),
            'min_size': int(best_compactness[('min_size', '')]),
            'score': float(best_compactness[('compactness', 'mean')]),
        },
        'best_boundary': {
            'scale': float(best_boundary[('scale', '')]),
            'sigma': float(best_boundary[('sigma', '')]),
            'min_size': int(best_boundary[('min_size', '')]),
            'score': float(best_boundary[('boundary_edge_strength', 'mean')]),
        },
        'best_speed': {
            'scale': float(best_speed[('scale', '')]),
            'sigma': float(best_speed[('sigma', '')]),
            'min_size': int(best_speed[('min_size', '')]),
            'speed_ms': float(best_speed[('elapsed_ms', 'mean')]),
        },
    }

def main() -> None:
    config = HyperparameterConfig()
    tuner = FelzenszwalbTuner(config)
    results_df = tuner.grid_search()
    
    output_dir = Path(config.output_base)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_csv = output_dir / "tuning_results_final.csv"
    results_df.to_csv(results_csv, index=False)
    
    analysis = analyze_results(results_df)
    analysis_json = output_dir / "recommendations.json"
    with open(analysis_json, 'w') as f:
        json.dump(analysis, f, indent=2)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
