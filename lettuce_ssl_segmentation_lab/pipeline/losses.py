"""
Loss functions for Lettuce Disease Segmentation.
Includes Dice Loss and combined Cross-Entropy + Dice Loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """Dice Loss for multi-class segmentation."""
    def __init__(self, smooth=1.0, ignore_index=None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        logits: (B, C, H, W)
        targets: (B, H, W)
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).float().unsqueeze(1)
            probs = probs * mask
            targets_one_hot = targets_one_hot * mask
            
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs + targets_one_hot, dims)
        
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1 - torch.mean(dice_score)

class SegmentationLoss(nn.Module):
    """Combined Cross-Entropy and Dice Loss."""
    def __init__(self, ce_weight=1.0, dice_weight=1.0, ignore_index=255, class_weights=None):
        super(SegmentationLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=class_weights)
        self.dice = DiceLoss(ignore_index=ignore_index if ignore_index != 255 else None)

    def forward(self, logits, targets):
        # Handle SegFormer output format if necessary (some versions return a dict or tuple)
        if hasattr(logits, 'logits'):
            logits = logits.logits
            
        # Upsample if targets are larger (e.g. SegFormer output is 1/4 size)
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(logits, size=targets.shape[-2:], mode='bilinear', align_corners=False)
            
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss
