"""
Felzenszwalb Segmentation Mask Generator with GPU Acceleration
Generates segmentation masks using Felzenszwalb algorithm with hyperparameter tuning.
Optimized for creating training masks for custom segmentation models.
"""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, find_boundaries
from tqdm import tqdm


@dataclass
class FelzenszwalbConfig:
    """Configuration for Felzenszwalb segmentation"""
    dataset_base: str = "Lettuce_disease_datasets_split"
    output_base: str = "felzenszwalb_masks_output"
    splits: Tuple[str, ...] = ("train", "validation", "test")
    
    # Felzenszwalb hyperparameters
    felz_scale: float = 120.0  # Controls size of regions
    felz_sigma: float = 0.6    # Gaussian blur sigma
    felz_min_size: int = 12    # Minimum segment size
    
    # Hyperparameter tuning ranges (if performing grid search)
    scale_range: Tuple[float, ...] = (80.0, 100.0, 120.0, 150.0)
    sigma_range: Tuple[float, ...] = (0.4, 0.6, 0.8, 1.0)
    min_size_range: Tuple[int, ...] = (8, 12, 16, 20)
    
    # Processing options
    apply_edge_enhancement: bool = True
    save_colored_masks: bool = True
    save_raw_masks: bool = True
    save_boundary_masks: bool = True
    overwrite: bool = False
    samples_per_class: Optional[int] = None
    
    # Performance
    device: Optional[str] = None
    use_gpu: bool = True


class EdgeEnhancer:
    """GPU-accelerated edge enhancement for better Felzenszwalb results"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
    
    def compute_edge_map(self, img_rgb: np.ndarray) -> np.ndarray:
        """Compute edge map using LAB color space gradients"""
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        l_channel = img_lab[..., 0] / 255.0
        a_channel = (img_lab[..., 1] - 128.0) / 127.0
        b_channel = (img_lab[..., 2] - 128.0) / 127.0
        
        edge_map = 0.6 * sobel(l_channel) + 0.2 * sobel(a_channel) + 0.2 * sobel(b_channel)
        edge_map = edge_map.astype(np.float32)
        return (edge_map - edge_map.min()) / (np.ptp(edge_map) + 1e-6)
    
    def enhance_for_felzenszwalb(self, img_rgb: np.ndarray) -> np.ndarray:
        """Enhance image for Felzenszwalb by combining LAB and edge information"""
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
        edge_map = self.compute_edge_map(img_rgb)
        
        # Combine LAB and edge information
        enhanced = np.dstack([img_lab, edge_map * 0.5])  # Scale edge info
        return enhanced


class FelzenszwalbSegmenter:
    """Felzenszwalb segmentation with GPU support"""
    
    def __init__(self, config: FelzenszwalbConfig):
        self.config = config
        self.device = torch.device(
            config.device if config.device is not None
            else ("cuda" if (config.use_gpu and torch.cuda.is_available()) else "cpu")
        )
        self.edge_enhancer = EdgeEnhancer(device=str(self.device))
        
        print(f"FelzenszwalbSegmenter initialized")
        print(f"  Device: {self.device}")
        print(f"  GPU Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
    
    def segment_image(
        self,
        img_rgb: np.ndarray,
        scale: float,
        sigma: float,
        min_size: int,
        enhance_edges: bool = True,
    ) -> np.ndarray:
        """
        Segment image using Felzenszwalb algorithm
        
        Args:
            img_rgb: RGB image array
            scale: Scale parameter (larger = larger segments)
            sigma: Gaussian blur sigma
            min_size: Minimum segment size
            enhance_edges: Whether to enhance edges before segmentation
        
        Returns:
            Segmentation mask with labeled regions
        """
        if enhance_edges:
            img_input = self.edge_enhancer.enhance_for_felzenszwalb(img_rgb)
            img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
            # Use LAB for actual segmentation, edges for guidance
            segments = felzenszwalb(
                img_lab,
                scale=scale,
                sigma=sigma,
                min_size=min_size,
                channel_axis=-1,
            )
        else:
            img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
            segments = felzenszwalb(
                img_lab,
                scale=scale,
                sigma=sigma,
                min_size=min_size,
                channel_axis=-1,
            )
        
        # Relabel to ensure consecutive IDs
        labels, relabeled = np.unique(segments, return_inverse=True)
        return relabeled.reshape(segments.shape).astype(np.int32)
    
    def merge_tiny_segments(
        self,
        segments: np.ndarray,
        img_rgb: np.ndarray,
        min_size: int = 10,
    ) -> np.ndarray:
        """Merge very small segments with neighbors"""
        if min_size <= 1:
            return segments
        
        segments = segments.copy()
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        changed = True
        iterations = 0
        max_iterations = 5
        
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            ids, counts = np.unique(segments, return_counts=True)
            small_ids = ids[counts < min_size]
            
            if len(small_ids) == 0:
                break
            
            for seg_id in small_ids:
                mask = segments == seg_id
                if not np.any(mask):
                    continue
                
                # Find neighboring segments
                neighbors = self._get_neighbors(segments, seg_id)
                if not neighbors:
                    continue
                
                # Merge with most similar neighbor
                best_neighbor = self._find_best_merge_neighbor(
                    segments, img_lab, mask, neighbors
                )
                segments[mask] = best_neighbor
                changed = True
            
            if changed:
                segments = self._relabel_segments(segments)
        
        return segments
    
    @staticmethod
    def _get_neighbors(segments: np.ndarray, seg_id: int) -> set:
        """Get neighboring segment IDs"""
        mask = segments == seg_id
        dilated = ndi.binary_dilation(mask, iterations=1)
        neighbors = set(segments[dilated].ravel()) - {seg_id}
        return neighbors
    
    @staticmethod
    def _find_best_merge_neighbor(
        segments: np.ndarray,
        img_lab: np.ndarray,
        mask: np.ndarray,
        neighbors: set,
    ) -> int:
        """Find best neighbor to merge with based on color similarity"""
        if not neighbors:
            return int(segments[mask][0]) if np.any(mask) else 0
        
        seg_color = img_lab[mask].mean(axis=0)
        best_neighbor = list(neighbors)[0]
        best_distance = float('inf')
        
        for neighbor_id in neighbors:
            neighbor_mask = segments == neighbor_id
            if not np.any(neighbor_mask):
                continue
            neighbor_color = img_lab[neighbor_mask].mean(axis=0)
            distance = np.linalg.norm(seg_color - neighbor_color)
            if distance < best_distance:
                best_distance = distance
                best_neighbor = neighbor_id
        
        return best_neighbor
    
    @staticmethod
    def _relabel_segments(segments: np.ndarray) -> np.ndarray:
        """Relabel segments to have consecutive IDs"""
        labels, relabeled = np.unique(segments, return_inverse=True)
        return relabeled.reshape(segments.shape).astype(np.int32)


class MaskProcessor:
    """Process and save segmentation masks"""
    
    def __init__(self, config: FelzenszwalbConfig):
        self.config = config
    
    def create_colored_mask(self, segments: np.ndarray) -> np.ndarray:
        """Create a colored visualization of segments"""
        num_segments = int(segments.max()) + 1
        colors = np.random.RandomState(42).randint(0, 255, (num_segments, 3), dtype=np.uint8)
        colored_mask = colors[segments]
        return colored_mask
    
    def create_boundary_mask(self, segments: np.ndarray) -> np.ndarray:
        """Create a mask highlighting segment boundaries"""
        boundaries = find_boundaries(segments, mode='inner')
        boundary_mask = np.where(boundaries, 255, 0).astype(np.uint8)
        return boundary_mask
    
    def save_masks(
        self,
        image_path: Path,
        class_name: str,
        split: str,
        segments: np.ndarray,
        scale: float,
        sigma: float,
        min_size: int,
    ) -> Dict[str, Path]:
        """Save segmentation masks to output directory"""
        stem = image_path.stem
        scale_str = f"{scale:.1f}".replace('.', '_')
        sigma_str = f"{sigma:.1f}".replace('.', '_')
        param_suffix = f"s{scale_str}_sg{sigma_str}_ms{min_size}"
        
        base_dir = Path(self.config.output_base)
        output_dir = base_dir / split / class_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = {}
        
        # Save raw segmentation mask
        if self.config.save_raw_masks:
            raw_mask_path = output_dir / f"{stem}_{param_suffix}_raw.png"
            # Normalize segments to 0-255 range for visualization
            segments_normalized = ((segments / (segments.max() + 1)) * 255).astype(np.uint8)
            cv2.imwrite(str(raw_mask_path), segments_normalized)
            saved_paths['raw'] = raw_mask_path
        
        # Save colored mask for visualization
        if self.config.save_colored_masks:
            colored_mask = self.create_colored_mask(segments)
            colored_mask_path = output_dir / f"{stem}_{param_suffix}_colored.png"
            colored_bgr = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(colored_mask_path), colored_bgr)
            saved_paths['colored'] = colored_mask_path
        
        # Save boundary mask
        if self.config.save_boundary_masks:
            boundary_mask = self.create_boundary_mask(segments)
            boundary_mask_path = output_dir / f"{stem}_{param_suffix}_boundary.png"
            cv2.imwrite(str(boundary_mask_path), boundary_mask)
            saved_paths['boundary'] = boundary_mask_path
        
        # Save segment information
        info_path = output_dir / f"{stem}_{param_suffix}_info.txt"
        with open(info_path, 'w') as f:
            f.write(f"Image: {image_path.name}\n")
            f.write(f"Scale: {scale}\n")
            f.write(f"Sigma: {sigma}\n")
            f.write(f"Min Size: {min_size}\n")
            f.write(f"Num Segments: {int(segments.max() + 1)}\n")
            f.write(f"Image Shape: {segments.shape}\n")
        
        return saved_paths


def load_rgb_image(image_path: Path) -> np.ndarray:
    """Load image in RGB format"""
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def collect_image_paths(
    split_dir: Path,
    samples_per_class: Optional[int] = None,
) -> List[Tuple[str, Path]]:
    """Collect image paths from split directory"""
    items: List[Tuple[str, Path]] = []
    for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        image_path_map: Dict[str, Path] = {}
        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            for path in class_dir.glob(pattern):
                image_path_map[str(path.resolve()).lower()] = path
        
        image_paths = sorted(image_path_map.values())
        if samples_per_class is not None:
            image_paths = image_paths[:samples_per_class]
        
        items.extend((class_dir.name, path) for path in image_paths)
    
    return items


def process_image(
    image_path: Path,
    class_name: str,
    split: str,
    config: FelzenszwalbConfig,
    segmenter: FelzenszwalbSegmenter,
    mask_processor: MaskProcessor,
) -> dict:
    """Process single image and generate segmentation masks"""
    start_time = time.time()
    
    try:
        img_rgb = load_rgb_image(image_path)
        height, width = img_rgb.shape[:2]
        
        # Perform segmentation
        segments = segmenter.segment_image(
            img_rgb,
            scale=config.felz_scale,
            sigma=config.felz_sigma,
            min_size=config.felz_min_size,
            enhance_edges=config.apply_edge_enhancement,
        )
        
        # Optional: Merge tiny segments
        segments = segmenter.merge_tiny_segments(
            segments,
            img_rgb,
            min_size=max(config.felz_min_size - 2, 5),
        )
        
        # Save masks
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
            'height': height,
            'width': width,
            'num_segments': int(segments.max() + 1),
            'scale': config.felz_scale,
            'sigma': config.felz_sigma,
            'min_size': config.felz_min_size,
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


def process_dataset(config: FelzenszwalbConfig) -> pd.DataFrame:
    """Process entire dataset"""
    dataset_base = Path(config.dataset_base)
    
    if not dataset_base.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_base}")
    
    # Initialize components
    segmenter = FelzenszwalbSegmenter(config)
    mask_processor = MaskProcessor(config)
    
    all_rows: List[dict] = []
    
    for split in config.splits:
        split_dir = dataset_base / split
        
        if not split_dir.exists():
            print(f"Skipping split '{split}': directory not found")
            continue
        
        items = collect_image_paths(split_dir, config.samples_per_class)
        
        if not items:
            print(f"No images found in {split_dir}")
            continue
        
        print(f"\nProcessing {split} split ({len(items)} images)...")
        progress = tqdm(items, desc=f"{split:>12}", unit="image", ncols=100)
        
        for class_name, image_path in progress:
            result = process_image(
                image_path=image_path,
                class_name=class_name,
                split=split,
                config=config,
                segmenter=segmenter,
                mask_processor=mask_processor,
            )
            
            all_rows.append(result)
            
            if result['status'] == 'success':
                progress.set_postfix({
                    'segments': result['num_segments'],
                    'time': f"{result['elapsed_seconds']}s"
                })
            else:
                progress.set_postfix({'status': 'failed'})
    
    # Save summary
    summary = pd.DataFrame(all_rows)
    summary_path = Path(config.output_base) / "processing_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    
    print(f"\n✓ Summary saved to: {summary_path}")
    
    return summary


def print_configuration(config: FelzenszwalbConfig) -> None:
    """Print configuration details"""
    print("\n" + "="*70)
    print("FELZENSZWALB SEGMENTATION CONFIGURATION")
    print("="*70)
    print(f"Dataset Base: {config.dataset_base}")
    print(f"Output Base: {config.output_base}")
    print(f"Splits: {', '.join(config.splits)}")
    print(f"\nFelzenszwalb Hyperparameters:")
    print(f"  Scale: {config.felz_scale}")
    print(f"  Sigma: {config.felz_sigma}")
    print(f"  Min Size: {config.felz_min_size}")
    print(f"\nProcessing Options:")
    print(f"  Edge Enhancement: {config.apply_edge_enhancement}")
    print(f"  Save Colored Masks: {config.save_colored_masks}")
    print(f"  Save Raw Masks: {config.save_raw_masks}")
    print(f"  Save Boundary Masks: {config.save_boundary_masks}")
    print(f"  Overwrite: {config.overwrite}")
    print(f"\nDevice:")
    print(f"  Use GPU: {config.use_gpu}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU Device: {torch.cuda.get_device_name(0)}")
    print("="*70 + "\n")


def main() -> None:
    """Main execution"""
    config = FelzenszwalbConfig()
    print_configuration(config)
    
    summary = process_dataset(config)
    
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print(f"Total images processed: {len(summary)}")
    successful = (summary['status'] == 'success').sum()
    failed = (summary['status'] == 'failed').sum()
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nOutput directory: {config.output_base}")
    print("="*70 + "\n")
    
    print("Sample Results:")
    print(summary[['split', 'class_name', 'status', 'num_segments', 'elapsed_seconds']].head(10))


if __name__ == "__main__":
    main()
