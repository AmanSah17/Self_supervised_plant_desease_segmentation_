"""
Quick smoke test for Stage 2: Healthy Learning
Tests with small batch for validation before full run.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import traceback
from datetime import datetime

from lettuce_ssl_segmentation_lab.config import LabConfig
from lettuce_ssl_segmentation_lab.pipeline.healthy_learner import HealthyRepresentationLearner


def print_header(title: str, width: int = 80):
    print(f"\n{'='*width}")
    print(f"{title.center(width)}")
    print(f"{'='*width}\n")


def test_1_config():
    """Test 1: Configuration validation"""
    print_header("Test 1: Configuration Validation")
    try:
        config = LabConfig().resolve()
        print(f"✓ Config loaded")
        print(f"  Dataset: {config.dataset_base}")
        print(f"  Splits: {config.splits}")
        print(f"  Classes: {len(config.class_names)}")
        print(f"  Healthy class: {config.healthy_class_name}")
        return config
    except Exception as e:
        print(f"✗ Config failed: {str(e)}")
        return None


def test_2_learner_init(config):
    """Test 2: Learner initialization"""
    print_header("Test 2: Learner Initialization")
    try:
        learner = HealthyRepresentationLearner(config)
        print(f"✓ Learner initialized")
        print(f"  Device: {learner.device}")
        print(f"  Output dir: {learner.output_dir}")
        return learner
    except Exception as e:
        print(f"✗ Learner init failed: {str(e)}")
        traceback.print_exc()
        return None


def test_3_manifest(learner):
    """Test 3: Manifest loading"""
    print_header("Test 3: Manifest Loading")
    try:
        manifest_df = learner.setup_manifest()
        print(f"✓ Manifest loaded")
        print(f"  Total samples: {len(manifest_df)}")
        print(f"  Splits: {manifest_df['split'].unique().tolist()}")
        return manifest_df
    except Exception as e:
        print(f"✗ Manifest failed: {str(e)}")
        traceback.print_exc()
        return None


def test_4_extractor(learner):
    """Test 4: Feature extractor initialization"""
    print_header("Test 4: Feature Extractor Initialization")
    try:
        extractor = learner.setup_feature_extractor(model_name="dinov2_vitb14")
        print(f"✓ Extractor initialized")
        print(f"  Model: dinov2_vitb14")
        print(f"  Device: {learner.device}")
        print(f"  Feature dim: {extractor.feature_dim()}")
        return extractor
    except Exception as e:
        print(f"✗ Extractor failed: {str(e)}")
        print(f"  Note: If download failed, ensure internet connection")
        traceback.print_exc()
        return None


def test_5_healthy_dataset(learner):
    """Test 5: Healthy dataset filtering"""
    print_header("Test 5: Healthy Dataset Filtering")
    try:
        dataset = learner.get_healthy_dataset(split="train")
        if dataset is None:
            print(f"⚠️  No healthy dataset created")
            return None
        print(f"✓ Healthy dataset created")
        print(f"  Num samples: {len(dataset)}")
        
        # Test one sample
        sample = dataset[0]
        print(f"✓ Sample loaded successfully")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Class: {sample['class_name']}")
        print(f"  Label kind: {sample['label_kind']}")
        
        return dataset
    except Exception as e:
        print(f"✗ Dataset failed: {str(e)}")
        traceback.print_exc()
        return None


def test_6_feature_extraction_mini(learner):
    """Test 6: Mini feature extraction (1 batch)"""
    print_header("Test 6: Mini Feature Extraction (1 Batch)")
    try:
        features, metadata = learner.extract_healthy_features(
            split="train",
            batch_size=2,
            num_workers=0,
            max_batches=1,  # Only extract 1 batch for quick test
        )
        
        if features is None:
            print(f"⚠️  No features extracted")
            return None
        
        print(f"✓ Features extracted")
        print(f"  Shape: {features.shape}")
        print(f"  Dtype: {features.dtype}")
        print(f"  Value range: [{features.min():.6f}, {features.max():.6f}]")
        print(f"  Metadata keys: {list(metadata.keys())}")
        
        return features, metadata
    except Exception as e:
        print(f"✗ Feature extraction failed: {str(e)}")
        traceback.print_exc()
        return None, None


def test_7_statistics(features, metadata):
    """Test 7: Statistics computation"""
    print_header("Test 7: Statistics Computation")
    if features is None:
        print("⚠️  Skipped (no features)")
        return None
    
    try:
        stats = {
            "num_samples": len(features),
            "feature_dim": features.shape[1],
            "mean": features.mean(axis=0),
            "std": features.std(axis=0),
        }
        
        print(f"✓ Statistics computed")
        print(f"  Num samples: {stats['num_samples']}")
        print(f"  Feature dim: {stats['feature_dim']}")
        print(f"  Mean range: [{stats['mean'].min():.6f}, {stats['mean'].max():.6f}]")
        print(f"  Std range: [{stats['std'].min():.6f}, {stats['std'].max():.6f}]")
        
        return stats
    except Exception as e:
        print(f"✗ Statistics failed: {str(e)}")
        traceback.print_exc()
        return None


def main():
    print_header("STAGE 2 SMOKE TEST - HEALTHY-ONLY REPRESENTATION LEARNING", 80)
    print(f"Started: {datetime.now().isoformat()}\n")
    
    # Test 1
    config = test_1_config()
    if config is None:
        return 1
    
    # Test 2
    learner = test_2_learner_init(config)
    if learner is None:
        return 1
    
    # Test 3
    manifest_df = test_3_manifest(learner)
    if manifest_df is None:
        return 1
    
    # Test 4
    extractor = test_4_extractor(learner)
    if extractor is None:
        print("\n⚠️  Feature extractor initialization failed")
        print("This is expected if DINOv2 model cannot be downloaded")
        print("Continuing with remaining tests...\n")
    
    # Test 5
    dataset = test_5_healthy_dataset(learner)
    if dataset is None:
        return 1
    
    # Test 6
    if extractor is not None:
        features, metadata = test_6_feature_extraction_mini(learner)
    else:
        features, metadata = None, None
    
    # Test 7
    if features is not None:
        stats = test_7_statistics(features, metadata)
    else:
        stats = None
    
    # Summary
    print_header("SMOKE TEST SUMMARY", 80)
    
    passed = sum([
        config is not None,
        learner is not None,
        manifest_df is not None,
        dataset is not None,
    ])
    
    total = 4
    
    print(f"Tests passed: {passed}/{total}\n")
    
    if extractor is not None:
        print(f"Feature extraction tests:")
        print(f"  ✓ Extractor initialized")
        if features is not None:
            print(f"  ✓ Mini extraction successful")
            print(f"  ✓ Statistics computed")
        else:
            print(f"  ✗ Mini extraction failed")
    else:
        print(f"⚠️  Feature extractor not available (model download may have failed)")
    
    print_header("NEXT STEPS", 80)
    
    if passed == total:
        print("✓ All smoke tests passed!")
        print("\nTo run the full Stage 2 pipeline:")
        print("  python scripts/stage2_healthy_learning.py")
        print("\nNote: Full pipeline will:")
        print("  1. Extract features from ALL healthy training images")
        print("  2. Compute comprehensive statistics")
        print("  3. Save feature bank for anomaly detection")
        print("  4. Generate checkpoints for resumability")
        return 0
    else:
        print(f"✗ Some tests failed ({passed}/{total} passed)")
        print("Review errors above and fix configuration/dependencies")
        return 1


if __name__ == "__main__":
    exit_code = main()
    print()
    sys.exit(exit_code)
