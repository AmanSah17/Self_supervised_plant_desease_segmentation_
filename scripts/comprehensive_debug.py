"""
Comprehensive Debug & Analysis Script for Lettuce SSL Segmentation Lab
Tests all components with detailed logging and tqdm progress tracking.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import traceback
import json
from tqdm import tqdm
from tqdm.auto import trange
import pandas as pd
import numpy as np
import torch
import cv2

from lettuce_ssl_segmentation_lab.config import LabConfig
from lettuce_ssl_segmentation_lab.pipeline.orchestrator import SegmentationResearchOrchestrator
from lettuce_ssl_segmentation_lab.data.multichannel_dataset import MultiChannelLeafDataset


class DebugLogger:
    """Helper class for formatted console output"""
    
    @staticmethod
    def section(title: str, width: int = 72):
        print(f"\n{'='*width}")
        print(f"{title.center(width)}")
        print(f"{'='*width}")
    
    @staticmethod
    def subsection(title: str, width: int = 72):
        print(f"\n{'-'*width}")
        print(f"{title}")
        print(f"{'-'*width}")
    
    @staticmethod
    def success(msg: str):
        print(f"✓ {msg}")
    
    @staticmethod
    def warning(msg: str):
        print(f"⚠️  {msg}")
    
    @staticmethod
    def error(msg: str):
        print(f"✗ {msg}")
    
    @staticmethod
    def info(msg: str, indent: int = 0):
        prefix = "  " * indent
        print(f"{prefix}{msg}")


def test_1_config_validation():
    """Test 1: Configuration validation"""
    DebugLogger.section("TEST 1: Configuration Validation")
    
    try:
        config = LabConfig().resolve()
        
        print("\nConfiguration Settings:")
        DebugLogger.info(f"Dataset base: {config.dataset_base}", 1)
        DebugLogger.info(f"Felzenszwalb base: {config.felz_base}", 1)
        DebugLogger.info(f"Lab root: {config.lab_root}", 1)
        DebugLogger.info(f"Logs dir: {config.logs_dir}", 1)
        DebugLogger.info(f"Image size: {config.img_size}", 1)
        DebugLogger.info(f"Use CUDA: {config.use_cuda}", 1)
        DebugLogger.info(f"Use Numba: {config.use_numba}", 1)
        DebugLogger.info(f"Splits: {config.splits}", 1)
        DebugLogger.info(f"Classes ({len(config.class_names)}): {', '.join(config.class_names)}", 1)
        DebugLogger.info(f"Selected channels ({len(config.selected_channels)}): {', '.join(config.selected_channels)}", 1)
        
        # Validate directories exist
        print("\nDirectory Validation:")
        dirs_to_check = [
            ("Dataset base", config.dataset_base),
            ("Logs dir", config.logs_dir),
        ]
        
        for dir_name, dir_path in dirs_to_check:
            if dir_path.exists():
                DebugLogger.success(f"{dir_name}: {dir_path}")
            else:
                DebugLogger.warning(f"{dir_name}: {dir_path} (does not exist)")
        
        DebugLogger.success("Configuration valid")
        return config
    
    except Exception as e:
        DebugLogger.error("Configuration validation failed")
        traceback.print_exc()
        return None


def test_2_manifest_building(config):
    """Test 2: Manifest building with progress tracking"""
    DebugLogger.section("TEST 2: Manifest Building")
    
    if config is None:
        DebugLogger.warning("Skipped (config failed)")
        return None, None
    
    try:
        orchestrator = SegmentationResearchOrchestrator(config)
        
        print("\nBuilding manifest...")
        manifest_df, summary = orchestrator.build_manifest()
        
        print("\nManifest Statistics:")
        DebugLogger.info(f"Total samples: {len(manifest_df)}", 1)
        DebugLogger.info(f"Healthy samples: {summary.healthy_samples}", 1)
        DebugLogger.info(f"Diseased samples: {summary.diseased_samples}", 1)
        
        print("\nFelzenszwalb Coverage:")
        raw_pct = 100 * summary.samples_with_felz_raw / len(manifest_df)
        boundary_pct = 100 * summary.samples_with_felz_boundary / len(manifest_df)
        colored_pct = 100 * summary.samples_with_felz_colored / len(manifest_df)
        
        DebugLogger.info(f"Raw: {summary.samples_with_felz_raw}/{len(manifest_df)} ({raw_pct:.1f}%)", 1)
        DebugLogger.info(f"Boundary: {summary.samples_with_felz_boundary}/{len(manifest_df)} ({boundary_pct:.1f}%)", 1)
        DebugLogger.info(f"Colored: {summary.samples_with_felz_colored}/{len(manifest_df)} ({colored_pct:.1f}%)", 1)
        
        print("\nSplits Coverage:")
        for split in summary.splits_seen:
            split_count = len(manifest_df[manifest_df["split"] == split])
            DebugLogger.info(f"{split.capitalize()}: {split_count} samples", 1)
        
        print("\nClasses Coverage:")
        for class_name in summary.classes_seen:
            class_count = len(manifest_df[manifest_df["class_name"] == class_name])
            healthy_count = len(manifest_df[(manifest_df["class_name"] == class_name) & (manifest_df["label_kind"] == "healthy")])
            disease_count = class_count - healthy_count
            DebugLogger.info(f"{class_name}: {class_count} samples (healthy: {healthy_count}, disease: {disease_count})", 1)
        
        orchestrator.write_research_outputs(summary)
        DebugLogger.success("Manifest built and research outputs written")
        
        return manifest_df, config
    
    except Exception as e:
        DebugLogger.error("Manifest building failed")
        traceback.print_exc()
        return None, None


def test_3_dataset_instantiation(manifest_df, config):
    """Test 3: Dataset instantiation for all splits"""
    DebugLogger.section("TEST 3: Dataset Instantiation")
    
    if manifest_df is None or config is None:
        DebugLogger.warning("Skipped (manifest failed)")
        return {}
    
    datasets = {}
    
    for split in config.splits:
        try:
            split_df = manifest_df[manifest_df["split"] == split]
            if len(split_df) == 0:
                DebugLogger.warning(f"{split.capitalize()}: No samples found")
                continue
            
            dataset = MultiChannelLeafDataset(manifest_df, config, split=split)
            datasets[split] = dataset
            DebugLogger.success(f"{split.capitalize()}: {len(dataset)} samples")
        
        except Exception as e:
            DebugLogger.error(f"{split.capitalize()}: {str(e)[:100]}")
            traceback.print_exc()
    
    return datasets


def test_4_sample_loading(datasets, config, split="train", num_samples=5):
    """Test 4: Sample loading with tqdm"""
    DebugLogger.section("TEST 4: Sample Loading and Inspection")
    
    if split not in datasets:
        DebugLogger.warning(f"{split.capitalize()} dataset not available")
        return
    
    dataset = datasets[split]
    num_to_test = min(num_samples, len(dataset))
    
    print(f"\nLoading {num_to_test} samples from {split} split...\n")
    
    sample_stats = {
        "image_shapes": [],
        "segment_shapes": [],
        "num_segments": [],
        "classes": [],
        "label_kinds": [],
        "felz_coverage": 0,
    }
    
    for idx in tqdm(range(num_to_test), desc=f"Loading {split} samples", unit="sample", ncols=80):
        try:
            sample = dataset[idx]
            
            # Validate structure
            assert "image" in sample, "Missing 'image' key"
            assert "segments" in sample, "Missing 'segments' key"
            assert "class_name" in sample, "Missing 'class_name' key"
            
            image_tensor = sample["image"]
            segments = sample["segments"]
            
            # Collect statistics
            sample_stats["image_shapes"].append(tuple(image_tensor.shape))
            sample_stats["segment_shapes"].append(tuple(segments.shape))
            sample_stats["num_segments"].append(int(segments.unique().numel()))
            sample_stats["classes"].append(sample["class_name"])
            sample_stats["label_kinds"].append(sample["label_kind"])
            
            if sample.get("chosen_variant") is not None:
                sample_stats["felz_coverage"] += 1
        
        except Exception as e:
            DebugLogger.error(f"Sample {idx}: {str(e)[:100]}")
            if idx == 0:  # Always show first sample error
                traceback.print_exc()
            break
    
    # Print statistics
    print(f"\n{'-'*80}")
    print("Sample Statistics:")
    print(f"{'-'*80}")
    
    DebugLogger.info(f"Image shapes: {set(sample_stats['image_shapes'])}", 1)
    DebugLogger.info(f"Segment shapes: {set(sample_stats['segment_shapes'])}", 1)
    DebugLogger.info(f"Num segments - Min: {min(sample_stats['num_segments'])}, Max: {max(sample_stats['num_segments'])}, Avg: {np.mean(sample_stats['num_segments']):.1f}", 1)
    DebugLogger.info(f"Classes found: {set(sample_stats['classes'])}", 1)
    DebugLogger.info(f"Label kinds: {set(sample_stats['label_kinds'])}", 1)
    DebugLogger.info(f"Felz coverage: {sample_stats['felz_coverage']}/{num_to_test}", 1)


def test_5_dataloader(datasets, config, split="train", batch_size=4, num_batches=3):
    """Test 5: DataLoader with batching"""
    DebugLogger.section("TEST 5: DataLoader Batching")
    
    if split not in datasets:
        DebugLogger.warning(f"{split.capitalize()} dataset not available")
        return
    
    dataset = datasets[split]
    
    try:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        
        print(f"\nDataLoader created: {len(dataloader)} batches")
        print(f"Testing first {min(num_batches, len(dataloader))} batches...\n")
        
        batch_stats = {
            "batch_shapes": [],
            "image_ranges": [],
            "segment_ranges": [],
        }
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Loading {split} batches", 
                                               total=min(num_batches, len(dataloader)), 
                                               unit="batch", ncols=80)):
            try:
                images = batch["image"]
                segments = batch["segments"]
                classes = batch["class_name"]
                
                batch_stats["batch_shapes"].append(tuple(images.shape))
                batch_stats["image_ranges"].append((float(images.min()), float(images.max())))
                batch_stats["segment_ranges"].append((int(segments.min()), int(segments.max())))
                
                if batch_idx >= num_batches - 1:
                    break
            
            except Exception as e:
                DebugLogger.error(f"Batch {batch_idx}: {str(e)[:100]}")
                traceback.print_exc()
                break
        
        # Print batch statistics
        print(f"\n{'-'*80}")
        print("Batch Statistics:")
        print(f"{'-'*80}")
        
        DebugLogger.info(f"Batch shapes: {set(batch_stats['batch_shapes'])}", 1)
        DebugLogger.info(f"Image ranges: {batch_stats['image_ranges'][0]}", 1)
        DebugLogger.info(f"Segment ranges: {batch_stats['segment_ranges'][0]}", 1)
        
        DebugLogger.success(f"DataLoader test passed")
    
    except Exception as e:
        DebugLogger.error(f"DataLoader error: {str(e)}")
        traceback.print_exc()


def test_6_channel_verification(datasets, config, split="train", num_samples=2):
    """Test 6: Verify all selected channels are available"""
    DebugLogger.section("TEST 6: Channel Verification")
    
    if split not in datasets:
        DebugLogger.warning(f"{split.capitalize()} dataset not available")
        return
    
    dataset = datasets[split]
    
    print(f"\nVerifying channels: {', '.join(config.selected_channels)}\n")
    
    expected_channels = len(config.selected_channels)
    
    for idx in tqdm(range(min(num_samples, len(dataset))), desc="Verifying channels", unit="sample", ncols=80):
        try:
            sample = dataset[idx]
            image_tensor = sample["image"]
            
            if image_tensor.shape[0] != expected_channels:
                DebugLogger.warning(f"Sample {idx}: Expected {expected_channels} channels, got {image_tensor.shape[0]}")
            else:
                if idx == 0:
                    print(f"\nSample 0 channel breakdown:")
                    for ch_idx, ch_name in enumerate(config.selected_channels):
                        channel_data = image_tensor[ch_idx]
                        DebugLogger.info(f"{ch_name}: shape={tuple(channel_data.shape)}, dtype={channel_data.dtype}, range=[{channel_data.min():.4f}, {channel_data.max():.4f}]", 2)
        
        except Exception as e:
            DebugLogger.error(f"Sample {idx} channel verification failed: {str(e)[:100]}")
    
    DebugLogger.success("Channel verification complete")


def main():
    """Run all tests"""
    DebugLogger.section("LETTUCE SSL SEGMENTATION LAB - COMPREHENSIVE DEBUG", 80)
    print("\nRunning smoke tests and diagnostics...\n")
    
    # Test 1: Config
    config = test_1_config_validation()
    
    # Test 2: Manifest
    manifest_df, config = test_2_manifest_building(config)
    
    # Test 3: Dataset instantiation
    datasets = test_3_dataset_instantiation(manifest_df, config)
    
    # Test 4: Sample loading
    if datasets:
        test_4_sample_loading(datasets, config, split="train", num_samples=5)
    
    # Test 5: DataLoader
    if datasets:
        test_5_dataloader(datasets, config, split="train", batch_size=4, num_batches=3)
    
    # Test 6: Channel verification
    if datasets:
        test_6_channel_verification(datasets, config, split="train", num_samples=3)
    
    DebugLogger.section("DEBUG TESTS COMPLETED", 80)
    print()


if __name__ == "__main__":
    main()
