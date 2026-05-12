"""
Model Factory for Lettuce Disease Segmentation.
Supports SegFormer (Transformers) and DeepLabV3+ (Torchvision).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation, SegformerConfig
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

class SegmentationModelFactory:
    """Factory class to build segmentation models."""
    
    @staticmethod
    def build_segformer(
        num_classes: int = 9, 
        model_id: str = "nvidia/mit-b3",
        pretrained: bool = True,
        input_channels: int = 3
    ) -> nn.Module:
        """
        Build SegFormer model.
        nvidia/mit-b3 is recommended for a balance of speed and performance.
        """
        if pretrained:
            print(f"[INFO] Loading pretrained SegFormer: {model_id}")
            model = SegformerForSemanticSegmentation.from_pretrained(
                model_id,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            print(f"[INFO] Initializing SegFormer from config: {model_id}")
            config = SegformerConfig.from_pretrained(model_id, num_labels=num_classes)
            model = SegformerForSemanticSegmentation(config)
            
        if input_channels != 3:
            print(f"[INFO] Adapting SegFormer for {input_channels} input channels")
            old_conv = model.segformer.encoder.patch_embeddings[0].proj
            model.segformer.encoder.patch_embeddings[0].proj = nn.Conv2d(
                input_channels, 
                old_conv.out_channels, 
                kernel_size=old_conv.kernel_size, 
                stride=old_conv.stride, 
                padding=old_conv.padding
            )
            
        return model

    @staticmethod
    def build_deeplabv3(
        num_classes: int = 9,
        pretrained: bool = True,
        input_channels: int = 3
    ) -> nn.Module:
        """Build DeepLabV3+ model with ResNet-101 backbone."""
        weights = DeepLabV3_ResNet101_Weights.DEFAULT if pretrained else None
        print(f"[INFO] Building DeepLabV3+ (ResNet-101) pretrained={pretrained}")
        
        model = deeplabv3_resnet101(weights=weights)
        
        # Replace the classifier head
        # DeepLabV3 has two heads: 'classifier' and 'aux_classifier'
        in_channels = model.classifier[4].in_channels
        model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        
        if hasattr(model, 'aux_classifier') and model.aux_classifier is not None:
            in_channels_aux = model.aux_classifier[4].in_channels
            model.aux_classifier[4] = nn.Conv2d(in_channels_aux, num_classes, kernel_size=1)
            
        if input_channels != 3:
            print(f"[INFO] Adapting DeepLabV3+ for {input_channels} input channels")
            old_conv = model.backbone.conv1
            model.backbone.conv1 = nn.Conv2d(
                input_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
            
        return model

    @staticmethod
    def get_model(name: str, num_classes: int, input_channels: int = 3, **kwargs) -> nn.Module:
        """Unified entry point for model building."""
        name = name.lower()
        if "segformer" in name:
            model_id = "nvidia/mit-b3"
            if "b0" in name:
                model_id = "nvidia/mit-b0"
            elif "b1" in name:
                model_id = "nvidia/mit-b1"
            elif "b2" in name:
                model_id = "nvidia/mit-b2"
                
            return SegmentationModelFactory.build_segformer(
                num_classes, model_id=model_id, input_channels=input_channels, **kwargs
            )
        elif "deeplab" in name:
            return SegmentationModelFactory.build_deeplabv3(
                num_classes, input_channels=input_channels, **kwargs
            )
        else:
            raise ValueError(f"Unknown model name: {name}")
