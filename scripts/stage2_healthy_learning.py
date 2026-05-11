"""
Stage 2 Execution Script: Healthy-Only Representation Learning
Extracts DINOv2 features from healthy leaf images with full checkpointing.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
from datetime import datetime
import traceback

from lettuce_ssl_segmentation_lab.config import LabConfig
from lettuce_ssl_segmentation_lab.pipeline.healthy_learner import HealthyRepresentationLearner


def print_section(title: str, width: int = 80):
    """Print formatted section header."""
    print(f"\n{'='*width}")
    print(f"{title.center(width)}")
    print(f"{'='*width}\n")


def main():
    """Run Stage 2 pipeline."""
    
    print_section("STAGE 2: HEALTHY-ONLY REPRESENTATION LEARNING", 80)
    
    print(f"Execution started at: {datetime.now().isoformat()}")
    print(f"Python version: {sys.version}")
    print(f"PyTorch available: Available")
    
    # Configuration
    config = LabConfig().resolve()
    
    print(f"\nConfiguration:")
    print(f"  Dataset base: {config.dataset_base}")
    print(f"  Healthy class: {config.healthy_class_name}")
    print(f"  Image size: {config.img_size}")
    print(f"  Device: {'CUDA' if config.use_cuda else 'CPU'}")
    
    # Initialize learner
    try:
        print_section("Initializing Learner")
        learner = HealthyRepresentationLearner(config, output_dir=None)
        print(f"✓ Learner initialized")
        print(f"  Output dir: {learner.output_dir}")
    
    except Exception as e:
        print(f"✗ Failed to initialize learner: {str(e)}")
        traceback.print_exc()
        return 1
    
    # Run pipeline
    try:
        print_section("Running Pipeline")
        
        result = learner.run_pipeline(
            split="train",
            batch_size=4,
            num_workers=0,  # Set to 0 for safety, increase based on your system
            model_name="dinov2_vitb14",
        )
        
        if result["status"] == "completed":
            print_section("Pipeline Execution Summary")
            print(f"✓ Status: COMPLETED")
            print(f"  Samples processed: {result['num_samples']}")
            print(f"  Feature dimension: {result['feature_dim']}")
            print(f"  Feature shape: {result['feature_shape']}")
            
            print(f"\nOutput files:")
            output_files = [
                "healthy_features_train.npy",
                "healthy_stats_train.json",
                "healthy_metadata_train.pkl",
                "healthy_bank_summary_train.json",
                "checkpoint_metadata.json",
            ]
            for fname in output_files:
                fpath = learner.output_dir / fname
                if fpath.exists():
                    print(f"  ✓ {fname} ({fpath.stat().st_size / 1024 / 1024:.2f} MB)")
                else:
                    print(f"  ⚠️  {fname} (not found)")
            
            print_section("Next Steps")
            print("Stage 2 complete! Ready for Stage 3:")
            print("1. Stage 3: Anomaly Localization")
            print("   - Use healthy feature bank with PaDiM")
            print("   - Generate anomaly scores for diseased images")
            print("")
            print("2. Then proceed with:")
            print("   - Stage 4: CAM fusion and pseudo masks")
            print("   - Stage 5: Mask refinement")
            print("   - Stage 6: Segmentation training with SegFormer")
        
        else:
            print_section("Pipeline Execution Summary")
            print(f"✗ Status: FAILED")
            print(f"  Error: {result.get('error', 'Unknown error')}")
            return 1
    
    except KeyboardInterrupt:
        print(f"\n\n✗ Pipeline interrupted by user")
        learner._log("Pipeline interrupted by user", level="error")
        return 130
    
    except Exception as e:
        print(f"\n✗ Pipeline execution failed: {str(e)}")
        traceback.print_exc()
        return 1
    
    print_section("Execution Complete")
    print(f"Completed at: {datetime.now().isoformat()}\n")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
