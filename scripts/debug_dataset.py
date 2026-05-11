"""
Debug script for testing dataset loading with tqdm progress bars.
Tests each component step-by-step.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import traceback
from tqdm import tqdm
import pandas as pd
import torch

from lettuce_ssl_segmentation_lab.config import LabConfig
from lettuce_ssl_segmentation_lab.pipeline.orchestrator import SegmentationResearchOrchestrator
from lettuce_ssl_segmentation_lab.data.multichannel_dataset import MultiChannelLeafDataset


def test_manifest_building():
    """Test 1: Manifest building"""
    print("\n" + "="*72)
    print("TEST 1: Building Manifest")
    print("="*72)
    
    try:
        config = LabConfig().resolve()
        orchestrator = SegmentationResearchOrchestrator(config)
        
        print(f"Dataset base: {config.dataset_base}")
        print(f"Felzenszwalb base: {config.felz_base}")
        print(f"Splits: {config.splits}")
        print(f"Classes: {config.class_names}")
        
        manifest_df, summary = orchestrator.build_manifest()
        print(f"✓ Manifest built successfully")
        print(f"  - Total samples: {len(manifest_df)}")
        print(f"  - Healthy: {summary.healthy_samples}")
        print(f"  - Diseased: {summary.diseased_samples}")
        print(f"  - Felz raw coverage: {summary.samples_with_felz_raw}/{len(manifest_df)} ({100*summary.samples_with_felz_raw/len(manifest_df):.1f}%)")
        
        orchestrator.write_research_outputs(summary)
        print(f"✓ Research outputs written")
        
        return manifest_df, config
    except Exception as e:
        print(f"✗ Error in manifest building:")
        traceback.print_exc()
        return None, None


def test_dataset_instantiation(manifest_df, config):
    """Test 2: Dataset instantiation"""
    print("\n" + "="*72)
    print("TEST 2: Dataset Instantiation")
    print("="*72)
    
    if manifest_df is None:
        print("✗ Skipped (manifest build failed)")
        return None
    
    try:
        for split in ["train", "validation", "test"]:
            split_df = manifest_df[manifest_df["split"] == split]
            if len(split_df) == 0:
                print(f"⚠️  {split}: No samples found")
                continue
            
            print(f"\nCreating {split} dataset...")
            dataset = MultiChannelLeafDataset(manifest_df, config, split=split)
            print(f"✓ {split} dataset created: {len(dataset)} samples")
        
        return dataset
    except Exception as e:
        print(f"✗ Error in dataset instantiation:")
        traceback.print_exc()
        return None


def test_dataset_sampling(manifest_df, config, split="train", num_samples=5):
    """Test 3: Dataset sampling with tqdm"""
    print("\n" + "="*72)
    print(f"TEST 3: Dataset Sampling ({split} split)")
    print("="*72)
    
    if manifest_df is None:
        print("✗ Skipped (manifest build failed)")
        return
    
    try:
        dataset = MultiChannelLeafDataset(manifest_df, config, split=split)
        
        if len(dataset) == 0:
            print(f"⚠️  {split} dataset is empty")
            return
        
        num_to_test = min(num_samples, len(dataset))
        print(f"\nLoading {num_to_test} samples from {split} set...\n")
        
        for idx in tqdm(range(num_to_test), desc=f"Loading {split} samples", unit="sample"):
            try:
                sample = dataset[idx]
                
                # Validate output structure
                assert "image" in sample, "Missing 'image' key"
                assert "segments" in sample, "Missing 'segments' key"
                assert "class_name" in sample, "Missing 'class_name' key"
                
                image_tensor = sample["image"]
                segments = sample["segments"]
                
                # Validate tensor shapes
                assert isinstance(image_tensor, torch.Tensor), f"image is {type(image_tensor)}, expected Tensor"
                assert isinstance(segments, torch.Tensor), f"segments is {type(segments)}, expected Tensor"
                
                # Validate values
                assert image_tensor.ndim == 3, f"image should be 3D (C, H, W), got {image_tensor.shape}"
                assert segments.ndim == 2, f"segments should be 2D (H, W), got {segments.shape}"
                
                if idx == 0:
                    print(f"\nSample 0 inspection:")
                    print(f"  - Image shape: {image_tensor.shape} (C, H, W)")
                    print(f"  - Image dtype: {image_tensor.dtype}")
                    print(f"  - Image range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
                    print(f"  - Segments shape: {segments.shape}")
                    print(f"  - Segments dtype: {segments.dtype}")
                    print(f"  - Segments unique values: {segments.unique().numel()}")
                    print(f"  - Class: {sample['class_name']}")
                    print(f"  - Label kind: {sample['label_kind']}")
                    print()
                
            except Exception as e:
                print(f"✗ Error loading sample {idx}:")
                print(f"  {str(e)}")
                traceback.print_exc()
                break
        else:
            print(f"✓ All {num_to_test} samples loaded successfully")
    
    except Exception as e:
        print(f"✗ Error in dataset sampling:")
        traceback.print_exc()


def test_dataloader(manifest_df, config, split="train", batch_size=2, num_batches=2):
    """Test 4: DataLoader with tqdm"""
    print("\n" + "="*72)
    print(f"TEST 4: DataLoader ({split} split, batch_size={batch_size})")
    print("="*72)
    
    if manifest_df is None:
        print("✗ Skipped (manifest build failed)")
        return
    
    try:
        dataset = MultiChannelLeafDataset(manifest_df, config, split=split)
        
        if len(dataset) == 0:
            print(f"⚠️  {split} dataset is empty")
            return
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        
        print(f"DataLoader created: {len(dataloader)} batches\n")
        
        num_batches_to_test = min(num_batches, len(dataloader))
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Loading {split} batches", 
                                               total=num_batches_to_test, unit="batch")):
            try:
                images = batch["image"]
                segments = batch["segments"]
                classes = batch["class_name"]
                
                assert images.shape[0] == batch_size, f"Batch size mismatch: {images.shape[0]} vs {batch_size}"
                
                if batch_idx == 0:
                    print(f"\nBatch 0 inspection:")
                    print(f"  - Images shape: {images.shape}")
                    print(f"  - Segments shape: {segments.shape}")
                    print(f"  - Classes: {classes}")
                    print()
                
                if batch_idx >= num_batches_to_test - 1:
                    break
            except Exception as e:
                print(f"✗ Error in batch {batch_idx}:")
                print(f"  {str(e)}")
                traceback.print_exc()
                break
        else:
            print(f"✓ All {num_batches_to_test} batches loaded successfully")
    
    except Exception as e:
        print(f"✗ Error in dataloader:")
        traceback.print_exc()


def main():
    """Run all tests"""
    print("\n" + "="*72)
    print("LETTUCE SSL SEGMENTATION LAB - DEBUG & SMOKE TEST")
    print("="*72)
    
    # Test 1: Manifest
    manifest_df, config = test_manifest_building()
    
    # Test 2: Dataset instantiation
    test_dataset_instantiation(manifest_df, config)
    
    # Test 3: Dataset sampling
    test_dataset_sampling(manifest_df, config, split="train", num_samples=3)
    
    # Test 4: DataLoader
    test_dataloader(manifest_df, config, split="train", batch_size=2, num_batches=2)
    
    print("\n" + "="*72)
    print("DEBUG TESTS COMPLETED")
    print("="*72 + "\n")


if __name__ == "__main__":
    main()
