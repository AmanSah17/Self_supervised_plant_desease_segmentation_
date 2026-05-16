"""
Stage 9: Test Inference Pipeline
Generates segmentation masks, disease classification, and analytics for test set images.
Produces comprehensive dashboard reports with disease spread analysis.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import json

import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image

# Add project root to PYTHONPATH
root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from lettuce_ssl_segmentation_lab.config import LabConfig
from lettuce_ssl_segmentation_lab.data.multichannel_dataset import MultiChannelLeafDataset
from lettuce_ssl_segmentation_lab.pipeline.models import SegmentationModelFactory
from lettuce_ssl_segmentation_lab.pipeline.metrics import SegmentationMetrics


def print_section(title: str, width: int = 80):
    """Print formatted section header."""
    print(f"\n{'='*width}")
    print(f"{title:^{width}}")
    print(f"{'='*width}\n")


class DiseaseAnalytics:
    """Analyzes disease spread and classification metrics."""
    
    def __init__(self, num_classes: int, class_names: list):
        self.num_classes = num_classes
        self.class_names = class_names
        
    def compute_disease_spread(self, mask: np.ndarray, prediction: np.ndarray) -> dict:
        """
        Compute disease spread metrics for a single image.
        
        Args:
            mask: Ground truth mask (H, W)
            prediction: Predicted mask (H, W)
            
        Returns:
            Dictionary with disease spread metrics
        """
        total_pixels = mask.size
        healthy_pixels = np.sum(mask == self.class_names.index('HLTY')) if 'HLTY' in self.class_names else 0
        diseased_pixels = total_pixels - healthy_pixels - np.sum(mask == 0)  # Exclude background
        
        # Calculate spread score: diseased_pixels / total_pixels
        spread_score = (diseased_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        
        # Count per-class pixels
        per_class_counts = {}
        for i, class_name in enumerate(self.class_names):
            count = np.sum(prediction == i)
            per_class_counts[class_name] = {
                'pixel_count': int(count),
                'percentage': (count / total_pixels * 100) if total_pixels > 0 else 0
            }
        
        return {
            'total_pixels': int(total_pixels),
            'healthy_pixels': int(healthy_pixels),
            'diseased_pixels': int(diseased_pixels),
            'spread_score': float(spread_score),
            'per_class_distribution': per_class_counts
        }
    
    def compute_confidence_score(self, logits: np.ndarray, prediction: np.ndarray) -> dict:
        """Compute confidence scores from softmax probabilities."""
        if len(logits.shape) == 3:  # (C, H, W)
            probs = torch.softmax(torch.from_numpy(logits), dim=0).numpy()
            confidence_map = np.max(probs, axis=0)  # (H, W) max probability per pixel
            avg_confidence = float(np.mean(confidence_map))
            per_class_confidence = {}
            
            for i, class_name in enumerate(self.class_names):
                class_mask = (prediction == i)
                if np.sum(class_mask) > 0:
                    per_class_confidence[class_name] = float(np.mean(confidence_map[class_mask]))
            
            return {
                'average_confidence': avg_confidence,
                'per_class_confidence': per_class_confidence,
                'confidence_map': confidence_map
            }
        else:
            return {'average_confidence': 0.0, 'per_class_confidence': {}}


class TestInferencePipeline:
    """Complete test inference pipeline with analytics and reporting."""
    
    def __init__(self, config: LabConfig, model_path: str):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.metrics = SegmentationMetrics(num_classes=len(self.config.class_names) + 1)
        self.analytics = DiseaseAnalytics(len(self.config.class_names) + 1, self.config.class_names)
        
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load fine-tuned model."""
        print(f"[INFO] Loading model from {model_path}")
        model = SegmentationModelFactory.get_model(
            name="segformer",
            num_classes=len(self.config.class_names) + 1,
            input_channels=14
        )
        
        state_dict = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model
    
    def run_inference(self, test_loader: DataLoader, output_dir: str) -> dict:
        """Run inference on test set."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        masks_dir = output_dir / "segmentation_masks"
        masks_dir.mkdir(exist_ok=True)
        
        all_results = []
        
        print_section("RUNNING INFERENCE ON TEST SET")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Test Inference")):
                images = batch["image"].to(self.device)
                image_stems = batch["image_stem"]
                gt_masks = batch["mask"].cpu().numpy() if "mask" in batch else None
                
                # Forward pass
                outputs = self.model(images)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                probs = torch.softmax(logits, dim=1)
                predictions = torch.argmax(probs, dim=1).cpu().numpy()
                
                # Interpolate to original size if needed
                if predictions.shape != images.shape[-2:]:
                    probs = torch.nn.functional.interpolate(
                        probs, size=images.shape[-2:], mode='bilinear', align_corners=False
                    )
                    predictions = torch.argmax(probs, dim=1).cpu().numpy()
                
                probs_np = probs.cpu().numpy()
                
                # Process each image in batch
                for i, stem in enumerate(image_stems):
                    pred_mask = predictions[i]
                    confidence = probs_np[i]
                    
                    # Save segmentation mask as PNG
                    mask_path = masks_dir / f"{stem}_predicted_mask.png"
                    cv2.imwrite(str(mask_path), pred_mask.astype(np.uint8))
                    
                    # Compute metrics
                    if gt_masks is not None:
                        self.metrics.update(
                            torch.from_numpy(pred_mask[np.newaxis, :, :]).long(),
                            torch.from_numpy(gt_masks[i][np.newaxis, :, :]).long()
                        )
                        disease_spread = self.analytics.compute_disease_spread(
                            gt_masks[i], pred_mask
                        )
                    else:
                        disease_spread = self.analytics.compute_disease_spread(
                            np.zeros_like(pred_mask), pred_mask
                        )
                    
                    confidence_scores = self.analytics.compute_confidence_score(
                        confidence, pred_mask
                    )
                    
                    # Determine dominant disease class (non-background, non-healthy)
                    unique_classes, counts = np.unique(pred_mask, return_counts=True)
                    class_counts = {self.config.class_names[c-1] if c > 0 else 'background': int(cnt) 
                                   for c, cnt in zip(unique_classes, counts) if c < len(self.config.class_names)}
                    dominant_class = max(class_counts.items(), key=lambda x: x[1])[0] if class_counts else 'unknown'
                    
                    result = {
                        'image_stem': stem,
                        'predicted_mask_path': str(mask_path),
                        'dominant_disease_class': dominant_class,
                        'confidence_score': confidence_scores['average_confidence'],
                        'disease_spread': disease_spread,
                        'per_class_confidence': confidence_scores['per_class_confidence']
                    }
                    all_results.append(result)
        
        print(f"[INFO] Inference complete. Processed {len(all_results)} images.")
        
        return {
            'results': all_results,
            'metrics': self.metrics.compute(),
            'total_samples': len(all_results),
            'device': str(self.device)
        }
    
    def save_results(self, inference_results: dict, output_dir: str):
        """Save detailed inference results to JSON."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / "inference_results.json"
        
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj) if isinstance(obj, np.floating) else int(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_types(item) for item in obj]
            return obj
        
        converted_results = convert_types(inference_results)
        
        with open(results_file, 'w') as f:
            json.dump(converted_results, f, indent=2)
        
        print(f"[INFO] Results saved to {results_file}")
        
        return results_file


def main():
    config = LabConfig()
    config.resolve()
    
    print_section("STAGE 9: TEST INFERENCE PIPELINE")
    
    # Load test dataset
    manifest_path = config.manifests_dir / "multirepresentation_manifest.csv"
    manifest_df = pd.read_csv(manifest_path)
    test_df = manifest_df[manifest_df['split'] == 'test'].reset_index(drop=True)
    
    print(f"[INFO] Found {len(test_df)} test samples")
    
    # Create test dataset and dataloader
    test_dataset = MultiChannelLeafDataset(test_df, config, split='test')
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    # Initialize inference pipeline
    model_path = config.lab_root / "stage6_segmentation_training" / "supervised_finetune" / "best_model.pth"
    
    if not model_path.exists():
        print(f"[ERROR] Model not found at {model_path}")
        print("[INFO] Using epoch_15.pth as fallback")
        model_path = config.lab_root / "stage6_segmentation_training" / "epoch_15.pth"
    
    pipeline = TestInferencePipeline(config, str(model_path))
    
    # Run inference
    output_dir = config.lab_root / "stage9_test_inference"
    inference_results = pipeline.run_inference(test_loader, str(output_dir))
    
    # Save results
    pipeline.save_results(inference_results, str(output_dir))
    
    # Print summary
    print_section("INFERENCE SUMMARY")
    print(f"Total samples processed: {inference_results['total_samples']}")
    print(f"Device used: {inference_results['device']}")
    
    metrics = inference_results['metrics']
    print(f"\nSegmentation Metrics:")
    print(f"  mIoU: {metrics['miou']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    
    print(f"\nDisease Spread Analysis:")
    all_spreads = [r['disease_spread']['spread_score'] for r in inference_results['results']]
    print(f"  Average disease spread: {np.mean(all_spreads):.2f}%")
    print(f"  Min disease spread: {np.min(all_spreads):.2f}%")
    print(f"  Max disease spread: {np.max(all_spreads):.2f}%")
    
    print("\n[SUCCESS] Stage 9 inference complete!")


if __name__ == "__main__":
    main()
