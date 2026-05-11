"""
Felzenszwalb Hyperparameter Tuning Script
Performs grid search over Felzenszwalb parameters with GPU acceleration.
Evaluates and compares different parameter combinations.
"""

from __future__ import annotations

import json
import time
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
    
    # Grid search ranges
    scale_range: Tuple[float, ...] = (80.0, 100.0, 120.0, 150.0, 180.0)
    sigma_range: Tuple[float, ...] = (0.3, 0.5, 0.7, 0.9, 1.1)
    min_size_range: Tuple[int, ...] = (8, 12, 16, 20, 25)
    
    # Sampling
    samples_per_class: int = 3  # Images to test per class
    max_classes: Optional[int] = None  # Limit classes for faster tuning
    
    # Device
    use_gpu: bool = True


class SegmentationMetrics:
    """Compute metrics for segmentation quality"""
    
    @staticmethod
    def compute_segment_statistics(segments: np.ndarray) -> Dict[str, float]:
        """Compute statistics about segmentation"""
        num_segments = int(segments.max()) + 1
        ids, counts = np.unique(segments, return_counts=True)
        
        return {
            'num_segments': float(num_segments),
            'mean_segment_size': float(np.mean(counts)),
            'std_segment_size': float(np.std(counts)),
            'min_segment_size': float(np.min(counts)),
            'max_segment_size': float(np.max(counts)),
            'size_uniformity': float(np.std(counts) / (np.mean(counts) + 1e-6)),  # Lower is better
        }
    
    @staticmethod
    def compute_boundary_metrics(segments: np.ndarray, img_rgb: np.ndarray) -> Dict[str, float]:
        """Compute metrics related to boundaries"""
        # Edge strength at boundaries
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        l_channel = img_lab[..., 0] / 255.0
        a_channel = (img_lab[..., 1] - 128.0) / 127.0
        b_channel = (img_lab[..., 2] - 128.0) / 127.0
        
        edge_map = 0.6 * sobel(l_channel) + 0.2 * sobel(a_channel) + 0.2 * sobel(b_channel)
        edge_map = (edge_map - edge_map.min()) / (np.ptp(edge_map) + 1e-6)
        
        # Find boundaries
        boundaries = ndi.binary_dilation(ndi.binary_erosion(segments)) ^ ndi.binary_erosion(segments)
        boundary_edges = edge_map[boundaries > 0]
        
        return {
            'boundary_edge_strength': float(np.mean(boundary_edges)) if len(boundary_edges) > 0 else 0.0,
            'boundary_edge_std': float(np.std(boundary_edges)) if len(boundary_edges) > 0 else 0.0,
        }
    
    @staticmethod
    def compute_compactness(segments: np.ndarray) -> float:
        """Compute average compactness of segments"""
        compactness_values = []
        
        for seg_id in np.unique(segments):
            mask = segments == seg_id
            area = float(np.sum(mask))
            
            if area == 0:
                continue
            
            # Perimeter approximation
            boundaries = ndi.binary_dilation(mask) ^ mask
            perimeter = float(np.sum(boundaries))
            
            if perimeter == 0:
                compactness = 0.0
            else:
                # Compactness = 4π * Area / Perimeter²
                compactness = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
            
            compactness_values.append(compactness)
        
        return float(np.mean(compactness_values)) if compactness_values else 0.0


class FelzenszwalbTuner:
    """Hyperparameter tuning for Felzenszwalb"""
    
    def __init__(self, config: HyperparameterConfig):
        self.config = config
        self.device = torch.device(
            "cuda" if (config.use_gpu and torch.cuda.is_available()) else "cpu"
        )
        self.metrics = SegmentationMetrics()
        print(f"FelzenszwalbTuner initialized on device: {self.device}")
    
    def segment_image(
        self,
        img_rgb: np.ndarray,
        scale: float,
        sigma: float,
        min_size: int,
    ) -> np.ndarray:
        """Segment image with given parameters"""
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
        segments = felzenszwalb(
            img_lab,
            scale=scale,
            sigma=sigma,
            min_size=min_size,
            channel_axis=-1,
        )
        
        # Relabel
        labels, relabeled = np.unique(segments, return_inverse=True)
        return relabeled.reshape(segments.shape).astype(np.int32)
    
    def evaluate_parameters(
        self,
        img_rgb: np.ndarray,
        scale: float,
        sigma: float,
        min_size: int,
    ) -> Dict[str, float]:
        """Evaluate segmentation quality for given parameters"""
        start_time = time.time()
        
        segments = self.segment_image(img_rgb, scale, sigma, min_size)
        
        stats = self.metrics.compute_segment_statistics(segments)
        boundary_metrics = self.metrics.compute_boundary_metrics(segments, img_rgb)
        compactness = self.metrics.compute_compactness(segments)
        
        elapsed = time.time() - start_time
        
        return {
            'scale': float(scale),
            'sigma': float(sigma),
            'min_size': float(min_size),
            'num_segments': stats['num_segments'],
            'mean_segment_size': stats['mean_segment_size'],
            'size_uniformity': stats['size_uniformity'],
            'compactness': compactness,
            'boundary_edge_strength': boundary_metrics['boundary_edge_strength'],
            'elapsed_ms': elapsed * 1000,
        }
    
    def grid_search(self) -> pd.DataFrame:
        """Perform grid search over hyperparameter space"""
        dataset_base = Path(self.config.dataset_base)
        test_split_dir = dataset_base / self.config.test_split
        
        if not test_split_dir.exists():
            raise FileNotFoundError(f"Test split not found: {test_split_dir}")
        
        # Collect test images
        test_images: List[Tuple[str, Path]] = []
        for class_dir in sorted(test_split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            
            if self.config.max_classes and len(test_images) >= self.config.max_classes:
                break
            
            image_paths = sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.png"))
            for img_path in image_paths[:self.config.samples_per_class]:
                test_images.append((class_dir.name, img_path))
        
        if not test_images:
            raise ValueError("No test images found")
        
        print(f"Using {len(test_images)} images for hyperparameter tuning")
        
        # Generate parameter combinations
        param_combinations = list(product(
            self.config.scale_range,
            self.config.sigma_range,
            self.config.min_size_range,
        ))
        
        print(f"Testing {len(param_combinations)} parameter combinations...")
        total_evals = len(test_images) * len(param_combinations)
        print(f"Total segmentations: {total_evals}")
        
        results: List[Dict] = []
        
        pbar = tqdm(total=total_evals, desc="Tuning", unit="eval", ncols=100)
        
        for class_name, img_path in test_images:
            try:
                img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img_bgr is None:
                    pbar.update(len(param_combinations))
                    continue
                
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                
                for scale, sigma, min_size in param_combinations:
                    try:
                        eval_result = self.evaluate_parameters(
                            img_rgb, scale, sigma, min_size
                        )
                        eval_result['class_name'] = class_name
                        eval_result['image'] = img_path.stem
                        results.append(eval_result)
                    except Exception as e:
                        print(f"Error evaluating {img_path} with params ({scale}, {sigma}, {min_size}): {e}")
                    
                    pbar.update(1)
            
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                pbar.update(len(param_combinations))
        
        pbar.close()
        
        return pd.DataFrame(results)


def analyze_results(results_df: pd.DataFrame) -> Dict:
    """Analyze tuning results and provide recommendations"""
    
    # Aggregate by parameters
    grouped = results_df.groupby(['scale', 'sigma', 'min_size']).agg({
        'num_segments': ['mean', 'std'],
        'mean_segment_size': ['mean', 'std'],
        'size_uniformity': 'mean',
        'compactness': 'mean',
        'boundary_edge_strength': 'mean',
        'elapsed_ms': 'mean',
    }).reset_index()
    
    # Find best parameters for different objectives
    best_uniformity = grouped.loc[grouped[('size_uniformity', 'mean')].idxmin()]
    best_compactness = grouped.loc[grouped[('compactness', 'mean')].idxmax()]
    best_boundary = grouped.loc[grouped[('boundary_edge_strength', 'mean')].idxmax()]
    best_speed = grouped.loc[grouped[('elapsed_ms', 'mean')].idxmin()]
    
    return {
        'best_uniformity': {
            'scale': float(best_uniformity['scale']),
            'sigma': float(best_uniformity['sigma']),
            'min_size': int(best_uniformity['min_size']),
            'score': float(best_uniformity[('size_uniformity', 'mean')]),
        },
        'best_compactness': {
            'scale': float(best_compactness['scale']),
            'sigma': float(best_compactness['sigma']),
            'min_size': int(best_compactness['min_size']),
            'score': float(best_compactness[('compactness', 'mean')]),
        },
        'best_boundary': {
            'scale': float(best_boundary['scale']),
            'sigma': float(best_boundary['sigma']),
            'min_size': int(best_boundary['min_size']),
            'score': float(best_boundary[('boundary_edge_strength', 'mean')]),
        },
        'best_speed': {
            'scale': float(best_speed['scale']),
            'sigma': float(best_speed['sigma']),
            'min_size': int(best_speed['min_size']),
            'speed_ms': float(best_speed[('elapsed_ms', 'mean')]),
        },
    }


def main() -> None:
    """Main execution"""
    config = HyperparameterConfig()
    
    print("\n" + "="*70)
    print("FELZENSZWALB HYPERPARAMETER TUNING")
    print("="*70)
    print(f"Scale range: {config.scale_range}")
    print(f"Sigma range: {config.sigma_range}")
    print(f"Min size range: {config.min_size_range}")
    print(f"Test split: {config.test_split}")
    print(f"Samples per class: {config.samples_per_class}")
    print("="*70 + "\n")
    
    tuner = FelzenszwalbTuner(config)
    results_df = tuner.grid_search()
    
    # Create output directory
    output_dir = Path(config.output_base)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    results_csv = output_dir / "tuning_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\n✓ Results saved to: {results_csv}")
    
    # Analyze results
    analysis = analyze_results(results_df)
    
    analysis_json = output_dir / "recommendations.json"
    with open(analysis_json, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"✓ Recommendations saved to: {analysis_json}")
    
    # Print recommendations
    print("\n" + "="*70)
    print("TUNING RECOMMENDATIONS")
    print("="*70)
    
    print("\nFor Best Size Uniformity (smaller std in segment sizes):")
    rec = analysis['best_uniformity']
    print(f"  Scale: {rec['scale']}, Sigma: {rec['sigma']}, Min Size: {rec['min_size']}")
    print(f"  Score: {rec['score']:.4f}")
    
    print("\nFor Best Compactness (more compact segments):")
    rec = analysis['best_compactness']
    print(f"  Scale: {rec['scale']}, Sigma: {rec['sigma']}, Min Size: {rec['min_size']}")
    print(f"  Score: {rec['score']:.4f}")
    
    print("\nFor Best Boundary Detection (highest edge strength at boundaries):")
    rec = analysis['best_boundary']
    print(f"  Scale: {rec['scale']}, Sigma: {rec['sigma']}, Min Size: {rec['min_size']}")
    print(f"  Score: {rec['score']:.4f}")
    
    print("\nFor Fastest Processing:")
    rec = analysis['best_speed']
    print(f"  Scale: {rec['scale']}, Sigma: {rec['sigma']}, Min Size: {rec['min_size']}")
    print(f"  Speed: {rec['speed_ms']:.2f} ms/image")
    
    print("\n" + "="*70)
    print("Top 10 Parameter Combinations (by size uniformity):")
    print("="*70)
    
    top_10 = results_df.groupby(['scale', 'sigma', 'min_size']).agg({
        'size_uniformity': 'mean',
        'compactness': 'mean',
        'boundary_edge_strength': 'mean',
        'elapsed_ms': 'mean',
    }).reset_index().sort_values('size_uniformity').head(10)
    
    print(top_10.to_string(index=False))
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
