import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import random
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(r'd:\gemma4\segmentation_lattuce-desease')
sys.path.append(str(project_root))

from lettuce_ssl_segmentation_lab.data.multichannel_dataset import MultiChannelLeafDataset
from lettuce_ssl_segmentation_lab.config import LabConfig
from lettuce_ssl_segmentation_lab.pipeline.models import SegmentationModelFactory
from lettuce_ssl_segmentation_lab.utils.feature_extractor import DINOv2FeatureExtractor
from lettuce_ssl_segmentation_lab.pipeline.anomaly_detector import PaDiMDetector
from lettuce_ssl_segmentation_lab.pipeline.classifier_head import DiseaseClassifierHead
from lettuce_ssl_segmentation_lab.pipeline.cam_attention_fusion import CAMAttentionRefiner
from lettuce_ssl_segmentation_lab.utils.background_segmenter import DINOv2BackgroundSegmenter
from lettuce_ssl_segmentation_lab.pipeline.orchestrator import SegmentationResearchOrchestrator

def main():
    config = LabConfig().resolve()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = project_root / 'Report' / 'final_pipeline_viz'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load All Models
    print("[INFO] Loading all pipeline models...")
    extractor = DINOv2FeatureExtractor(model_name="dinov2_vitb14", device=device)
    padim = PaDiMDetector.load(config.lab_root / "stage3_anomaly_localization" / "padim_model.pkl", device=device)
    
    unique_classes = sorted(['BACT', 'DML', 'HLTY', 'PML', 'SBL', 'SPW', 'VIRL', 'WLBL'])
    class_to_id = {cls: i for i, cls in enumerate(unique_classes)}
    
    head = DiseaseClassifierHead(num_classes=8)
    head.load(config.lab_root / "stage4_pseudo_masks" / "classifier_head_multiclass.pth", device=device)
    
    bg_segmenter = DINOv2BackgroundSegmenter(device=device)
    refiner = CAMAttentionRefiner(device=device)
    
    # Load Stage 6 SegFormer (Target 9 classes)
    # Check input channels from dataset
    orchestrator = SegmentationResearchOrchestrator(config)
    manifest_df, _ = orchestrator.build_manifest()
    dataset = MultiChannelLeafDataset(manifest_df, config, split="train")
    input_channels = dataset[0]["image"].shape[0]
    
    segformer = SegmentationModelFactory.get_model("segformer-b3", num_classes=9, input_channels=input_channels)
    weights_path = config.lab_root / "stage6_segmentation_training" / "epoch_15.pth"
    if weights_path.exists():
        print(f"[INFO] Loading SegFormer weights from {weights_path}")
        checkpoint = torch.load(weights_path, map_location=device)
        if "model_state_dict" in checkpoint:
            segformer.load_state_dict(checkpoint["model_state_dict"])
        else:
            segformer.load_state_dict(checkpoint)
    segformer.to(device).eval()

    # 2. Select 4 samples
    selected_classes = ['HLTY', 'BACT', 'DML', 'SBL']
    samples = []
    for cls in selected_classes:
        indices = manifest_df[manifest_df['class_name'] == cls].index.tolist()
        idx = random.choice(indices)
        samples.append((cls, idx))

    print(f"Processing {len(samples)} samples for comprehensive visualization...")

    # Mapping for segmentation classes
    other_classes = [c for c in unique_classes if c != 'HLTY']
    def get_seg_label(cls):
        if cls == 'HLTY': return 1
        return other_classes.index(cls) + 2

    # Plot Grid: 4 Samples x 6 Columns
    # RGB | Anomaly (S3) | CAM (S4) | Attention (S5) | Pseudo Mask (S5) | SegFormer (S6)
    fig, axes = plt.subplots(4, 6, figsize=(24, 16))

    for i, (cls_name, idx) in enumerate(samples):
        batch = dataset[idx]
        image_tensor = batch["image"] # (14, 256, 256)
        segments = batch["segments"].numpy()
        rgb_np = image_tensor[0:3].permute(1, 2, 0).numpy()
        
        # A. Anomaly Map
        input_rgb = image_tensor[0:3].unsqueeze(0).to(device)
        anomaly_map = padim.score(input_rgb, extractor)
        am_np = anomaly_map.squeeze().cpu().numpy()
        # Normalize
        am_np = (am_np - am_np.min()) / (am_np.max() - am_np.min() + 1e-8)
        
        # B. CAM Map
        with torch.no_grad():
            patch_feats = extractor.extract_patch_features(input_rgb)
            cam = head.generate_cam(patch_feats, class_to_id[cls_name])
            cam_np = torch.nn.functional.interpolate(
                cam, size=(256, 256), mode='bilinear', align_corners=False
            ).squeeze().cpu().numpy()
            
        # C. Attention Map (Stage 5 logic)
        fg_map = bg_segmenter.segment(rgb_np, patch_feats)
        a_tensor = torch.from_numpy(am_np).float().unsqueeze(0).unsqueeze(0).to(device)
        c_tensor = torch.from_numpy(cam_np).float().unsqueeze(0).unsqueeze(0).to(device)
        f_tensor = (torch.from_numpy(fg_map).float() > 0.35).float().unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            fused = refiner.fusion_module(a_tensor, c_tensor, f_tensor)
            attn_np = fused.squeeze().cpu().numpy()
            
        # D. Final Pseudo Mask
        pseudo_mask = refiner.refine(
            rgb_np, am_np, cam_np, fg_map, segments, 
            cat_label=get_seg_label(cls_name)
        )
        
        # E. SegFormer Prediction
        with torch.no_grad():
            # SegFormer expects (B, C, H, W)
            # Normalization might be needed if not handled in model wrapper
            # But the dataset usually returns float32 [0, 1]
            seg_out = segformer(image_tensor.unsqueeze(0).to(device))
            logits = seg_out.logits # (1, 9, 64, 64)
            # Upsample logits to 256x256
            upsampled_logits = torch.nn.functional.interpolate(
                logits, size=(256, 256), mode='bilinear', align_corners=False
            )
            pred_mask = torch.argmax(upsampled_logits, dim=1).squeeze().cpu().numpy()

        # Visualizations
        axes[i, 0].imshow(rgb_np)
        axes[i, 0].set_title(f"Original RGB ({cls_name})")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(am_np, cmap='jet')
        axes[i, 1].set_title("S3: Anomaly Evidence")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(cam_np, cmap='magma')
        axes[i, 2].set_title("S4: CAM Attention")
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(attn_np, cmap='hot')
        axes[i, 3].set_title("S5: Spatial Attention")
        axes[i, 3].axis('off')
        
        # Use discrete colormap for masks
        axes[i, 4].imshow(pseudo_mask, cmap='tab10', vmin=0, vmax=9)
        axes[i, 4].set_title("S5: Pseudo Mask (Target)")
        axes[i, 4].axis('off')
        
        axes[i, 5].imshow(pred_mask, cmap='tab10', vmin=0, vmax=9)
        axes[i, 5].set_title("S6: SegFormer Prediction")
        axes[i, 5].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / "comprehensive_pipeline_comparison.png", bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved comprehensive_pipeline_comparison.png to {output_dir}")

if __name__ == '__main__':
    main()
