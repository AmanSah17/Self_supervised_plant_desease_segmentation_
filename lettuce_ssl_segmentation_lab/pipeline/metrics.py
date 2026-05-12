"""
Metrics for Lettuce Disease Segmentation.
Computes mIoU, Precision, and Recall using confusion matrices.
"""
import torch
import numpy as np

class SegmentationMetrics:
    """Computes segmentation metrics for multi-class problems."""
    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """Update confusion matrix with new predictions and targets."""
        preds = preds.detach().cpu().numpy().flatten()
        targets = targets.detach().cpu().numpy().flatten()
        
        # Mask out ignore index
        mask = (targets != self.ignore_index)
        preds = preds[mask]
        targets = targets[mask]
        
        # Calculate confusion matrix
        # (targets * num_classes + preds) gives unique index for each (target, pred) pair
        idx = targets * self.num_classes + preds
        counts = np.bincount(idx, minlength=self.num_classes**2)
        self.confusion_matrix += counts.reshape((self.num_classes, self.num_classes))

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def compute(self):
        """Compute metrics from the confusion matrix."""
        cm = self.confusion_matrix
        
        # Intersection = Diagonal
        intersection = np.diag(cm)
        
        # Union = Sum of row + Sum of column - Intersection
        union = np.sum(cm, axis=0) + np.sum(cm, axis=1) - intersection
        
        # IoU
        iou = intersection / (union + 1e-8)
        miou = np.mean(iou)
        
        # Precision = TP / (TP + FP) -> Diagonal / Column Sum
        precision = intersection / (np.sum(cm, axis=0) + 1e-8)
        m_precision = np.mean(precision)
        
        # Recall = TP / (TP + FN) -> Diagonal / Row Sum
        recall = intersection / (np.sum(cm, axis=1) + 1e-8)
        m_recall = np.mean(recall)
        
        return {
            "miou": float(miou),
            "precision": float(m_precision),
            "recall": float(m_recall),
            "class_iou": iou.tolist(),
            "class_precision": precision.tolist(),
            "class_recall": recall.tolist()
        }
