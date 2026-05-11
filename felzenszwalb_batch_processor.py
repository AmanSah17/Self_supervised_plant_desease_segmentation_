"""
Batch Processing Utility for Felzenszwalb Segmentation
Allows running multiple configurations and comparing results easily.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import pandas as pd


@dataclass
class SegmentationConfig:
    """A single segmentation configuration"""
    name: str
    felz_scale: float
    felz_sigma: float
    felz_min_size: int
    description: str = ""


class BatchProcessor:
    """Process multiple Felzenszwalb configurations"""
    
    def __init__(self, base_output_dir: str = "felzenszwalb_batch_results"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.results_log = self.base_output_dir / "batch_log.csv"
    
    def run_configuration(
        self,
        config: SegmentationConfig,
        script_path: str = "felzenszwalb_segmentation_gpu.py",
        config_file: Optional[str] = None,
    ) -> dict:
        """Run a single configuration"""
        print(f"\n{'='*70}")
        print(f"Running: {config.name}")
        print(f"Description: {config.description}")
        print(f"Parameters: scale={config.felz_scale}, sigma={config.felz_sigma}, min_size={config.felz_min_size}")
        print(f"{'='*70}")
        
        # Create temporary config file
        config_dict = {
            'felz_scale': config.felz_scale,
            'felz_sigma': config.felz_sigma,
            'felz_min_size': config.felz_min_size,
            'output_base': str(self.base_output_dir / config.name),
        }
        
        temp_config = self.base_output_dir / f"{config.name}_config.json"
        with open(temp_config, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Run segmentation script
        result = {
            'config_name': config.name,
            'scale': config.felz_scale,
            'sigma': config.felz_sigma,
            'min_size': config.felz_min_size,
            'description': config.description,
            'config_file': str(temp_config),
        }
        
        try:
            # This is a placeholder - actual integration would require modifying the main script
            # to accept config file as argument
            result['status'] = 'ready'
            result['output_dir'] = str(self.base_output_dir / config.name)
            print(f"✓ Configuration {config.name} prepared")
            print(f"  Output directory: {result['output_dir']}")
            print(f"  Config file: {temp_config}")
        
        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
            print(f"✗ Configuration {config.name} failed: {e}")
        
        return result
    
    def run_multiple_configs(self, configs: List[SegmentationConfig]) -> pd.DataFrame:
        """Run multiple configurations"""
        results = []
        
        for config in configs:
            result = self.run_configuration(config)
            results.append(result)
        
        # Save results log
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.results_log, index=False)
        print(f"\n✓ Batch log saved to: {self.results_log}")
        
        return results_df
    
    def compare_results(self) -> None:
        """Compare results from different configurations"""
        if not self.results_log.exists():
            print("No results log found. Run configurations first.")
            return
        
        results_df = pd.read_csv(self.results_log)
        
        print("\n" + "="*70)
        print("BATCH PROCESSING RESULTS")
        print("="*70)
        print(results_df[['config_name', 'scale', 'sigma', 'min_size', 'status']].to_string(index=False))
        print("="*70 + "\n")


# Predefined configuration sets

LETTUCE_DISEASE_CONFIGS = [
    SegmentationConfig(
        name="fine_detail",
        felz_scale=80.0,
        felz_sigma=0.5,
        felz_min_size=8,
        description="Fine detail segmentation - many small segments"
    ),
    SegmentationConfig(
        name="balanced",
        felz_scale=120.0,
        felz_sigma=0.6,
        felz_min_size=12,
        description="Balanced segmentation - moderate segments"
    ),
    SegmentationConfig(
        name="coarse_regions",
        felz_scale=150.0,
        felz_sigma=0.7,
        felz_min_size=16,
        description="Coarse region segmentation - fewer large segments"
    ),
    SegmentationConfig(
        name="very_coarse",
        felz_scale=200.0,
        felz_sigma=0.9,
        felz_min_size=25,
        description="Very coarse segmentation - large regional segments"
    ),
]

EDGE_FOCUSED_CONFIGS = [
    SegmentationConfig(
        name="high_precision_edges",
        felz_scale=100.0,
        felz_sigma=0.4,
        felz_min_size=10,
        description="High precision edge detection"
    ),
    SegmentationConfig(
        name="disease_boundary",
        felz_scale=110.0,
        felz_sigma=0.5,
        felz_min_size=8,
        description="Optimized for disease boundary detection"
    ),
    SegmentationConfig(
        name="leaf_lesion_detail",
        felz_scale=85.0,
        felz_sigma=0.45,
        felz_min_size=6,
        description="Detailed lesion boundaries"
    ),
]

MULTI_SCALE_CONFIGS = [
    SegmentationConfig(
        name="multi_scale_small",
        felz_scale=70.0,
        felz_sigma=0.5,
        felz_min_size=5,
        description="Multi-scale analysis - small scale"
    ),
    SegmentationConfig(
        name="multi_scale_medium",
        felz_scale=120.0,
        felz_sigma=0.6,
        felz_min_size=12,
        description="Multi-scale analysis - medium scale"
    ),
    SegmentationConfig(
        name="multi_scale_large",
        felz_scale=180.0,
        felz_sigma=0.8,
        felz_min_size=20,
        description="Multi-scale analysis - large scale"
    ),
]


def print_menu() -> None:
    """Print batch processing menu"""
    print("\n" + "="*70)
    print("FELZENSZWALB BATCH PROCESSING UTILITY")
    print("="*70)
    print("\nConfiguration Sets Available:")
    print("  1. Lettuce Disease Defaults (4 configs)")
    print("  2. Edge-Focused Configs (3 configs)")
    print("  3. Multi-Scale Analysis (3 configs)")
    print("  4. Custom Configuration")
    print("  5. Compare Results")
    print("  6. Exit")
    print("="*70)


def get_user_choice() -> int:
    """Get user menu choice"""
    while True:
        try:
            choice = int(input("\nEnter your choice (1-6): "))
            if 1 <= choice <= 6:
                return choice
            print("Invalid choice. Please enter 1-6.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def get_custom_config() -> SegmentationConfig:
    """Get custom configuration from user"""
    print("\nEnter custom configuration parameters:")
    name = input("Configuration name (e.g., 'my_config'): ").strip()
    
    while True:
        try:
            scale = float(input("Felzenszwalb scale (50-250): "))
            if 50 <= scale <= 250:
                break
            print("Scale should be between 50 and 250")
        except ValueError:
            print("Invalid input. Enter a number.")
    
    while True:
        try:
            sigma = float(input("Felzenszwalb sigma (0.1-2.0): "))
            if 0.1 <= sigma <= 2.0:
                break
            print("Sigma should be between 0.1 and 2.0")
        except ValueError:
            print("Invalid input. Enter a number.")
    
    while True:
        try:
            min_size = int(input("Minimum segment size (1-100): "))
            if 1 <= min_size <= 100:
                break
            print("Min size should be between 1 and 100")
        except ValueError:
            print("Invalid input. Enter a number.")
    
    description = input("Description (optional): ").strip()
    
    return SegmentationConfig(
        name=name,
        felz_scale=scale,
        felz_sigma=sigma,
        felz_min_size=min_size,
        description=description
    )


def main() -> None:
    """Main interactive menu"""
    processor = BatchProcessor()
    
    while True:
        print_menu()
        choice = get_user_choice()
        
        if choice == 1:
            print("\nPreparing Lettuce Disease Default Configurations...")
            results = processor.run_multiple_configs(LETTUCE_DISEASE_CONFIGS)
            print("\nConfigurations prepared:")
            print(results[['config_name', 'output_dir', 'status']].to_string(index=False))
        
        elif choice == 2:
            print("\nPreparing Edge-Focused Configurations...")
            results = processor.run_multiple_configs(EDGE_FOCUSED_CONFIGS)
            print("\nConfigurations prepared:")
            print(results[['config_name', 'output_dir', 'status']].to_string(index=False))
        
        elif choice == 3:
            print("\nPreparing Multi-Scale Analysis Configurations...")
            results = processor.run_multiple_configs(MULTI_SCALE_CONFIGS)
            print("\nConfigurations prepared:")
            print(results[['config_name', 'output_dir', 'status']].to_string(index=False))
        
        elif choice == 4:
            print("\nCustom Configuration Mode")
            custom_config = get_custom_config()
            result = processor.run_configuration(custom_config)
            print(f"\nConfiguration '{custom_config.name}' prepared:")
            print(f"  Output directory: {result['output_dir']}")
            print(f"  Parameters: scale={custom_config.felz_scale}, "
                  f"sigma={custom_config.felz_sigma}, min_size={custom_config.felz_min_size}")
        
        elif choice == 5:
            processor.compare_results()
        
        elif choice == 6:
            print("\nExiting batch processor...")
            break


def setup_configuration(
    name: str,
    scale: float,
    sigma: float,
    min_size: int,
    output_dir: str = "felzenszwalb_batch_results"
) -> None:
    """
    Setup a specific configuration programmatically.
    
    Usage in scripts:
        setup_configuration("my_config", scale=120, sigma=0.6, min_size=12)
    """
    config = SegmentationConfig(
        name=name,
        felz_scale=scale,
        felz_sigma=sigma,
        felz_min_size=min_size,
    )
    processor = BatchProcessor(base_output_dir=output_dir)
    result = processor.run_configuration(config)
    print(f"Configuration '{name}' ready at: {result['output_dir']}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command line usage
        if sys.argv[1] == "setup":
            if len(sys.argv) >= 6:
                setup_configuration(
                    name=sys.argv[2],
                    scale=float(sys.argv[3]),
                    sigma=float(sys.argv[4]),
                    min_size=int(sys.argv[5]),
                )
            else:
                print("Usage: python script.py setup <name> <scale> <sigma> <min_size>")
        else:
            print("Unknown command. Use: python script.py setup <name> <scale> <sigma> <min_size>")
    else:
        # Interactive menu
        main()
