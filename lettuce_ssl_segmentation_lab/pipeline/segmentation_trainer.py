"""
Modular Trainer for Lettuce Disease Segmentation.
Includes MLflow integration for experiment tracking.
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import mlflow
from pathlib import Path
from typing import Dict, Any, Optional

from lettuce_ssl_segmentation_lab.pipeline.metrics import SegmentationMetrics

class SegmentationTrainer:
    """Trainer class for segmentation models with MLflow logging."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str,
        num_classes: int,
        experiment_name: str = "Lettuce_Segmentation",
        output_dir: Optional[Path] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.num_classes = num_classes
        self.output_dir = output_dir
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
        # Set MLflow experiment
        mlflow.set_experiment(experiment_name)
        
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch in pbar:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device).long()
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Handle model specific output formats
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs if isinstance(outputs, torch.Tensor) else outputs['out']

            # Upsample for loss calculation if needed (e.g. SegFormer 1/4 size)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = torch.nn.functional.interpolate(
                    logits, size=masks.shape[-2:], mode='bilinear', align_corners=False
                )
                
            loss = self.criterion(logits, masks)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(self.train_loader)
        mlflow.log_metric("train_loss", avg_loss, step=epoch)
        return avg_loss

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        metrics = SegmentationMetrics(self.num_classes)
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
        for batch in pbar:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device).long()
            
            outputs = self.model(images)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs if isinstance(outputs, torch.Tensor) else outputs['out']
            
            # Upsample for metric calculation if needed
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = torch.nn.functional.interpolate(
                    logits, size=masks.shape[-2:], mode='bilinear', align_corners=False
                )
                
            loss = self.criterion(logits, masks)
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            metrics.update(preds, masks)
            
        if len(self.val_loader) == 0:
            print("[WARNING] Validation loader is empty. Skipping validation metrics.")
            return {"miou": 0, "precision": 0, "recall": 0, "accuracy": 0}
            
        avg_loss = total_loss / len(self.val_loader)
        results = metrics.compute()
        
        # Log to MLflow
        mlflow.log_metric("val_loss", avg_loss, step=epoch)
        mlflow.log_metric("val_miou", results["miou"], step=epoch)
        mlflow.log_metric("val_precision", results["precision"], step=epoch)
        mlflow.log_metric("val_recall", results["recall"], step=epoch)
        mlflow.log_metric("val_accuracy", results["accuracy"], step=epoch)
        
        print(f"\n[Val Results] Loss: {avg_loss:.4f} | mIoU: {results['miou']:.4f} | Acc: {results['accuracy']:.4f} | Prec: {results['precision']:.4f}")
        
        return results

    def fit(self, num_epochs: int, patience: int = 15, resume_path: Optional[str] = None):
        start_epoch = 1
        best_miou = 0
        epochs_no_improve = 0
        
        if resume_path and Path(resume_path).exists():
            print(f"[INFO] Resuming training from {resume_path}")
            checkpoint = torch.load(resume_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_miou = checkpoint.get('miou', 0)
            print(f"  [OK] Resumed from epoch {checkpoint['epoch']} with best mIoU {best_miou:.4f}")

        print(f"[INFO] Starting training from epoch {start_epoch} to {num_epochs} on {self.device} (Patience: {patience})")
        
        with mlflow.start_run(nested=True if resume_path else False):
            # Log hyperparameters
            mlflow.log_params({
                "model_type": self.model.__class__.__name__,
                "optimizer": self.optimizer.__class__.__name__,
                "lr": self.optimizer.param_groups[0]['lr'],
                "num_epochs": num_epochs,
                "num_classes": self.num_classes,
                "patience": patience,
                "resumed": resume_path is not None
            })
            
            for epoch in range(start_epoch, num_epochs + 1):
                train_loss = self.train_epoch(epoch)
                val_results = self.validate(epoch)
                
                if val_results["miou"] > best_miou:
                    best_miou = val_results["miou"]
                    epochs_no_improve = 0
                    self.save_checkpoint("best_model.pth", epoch, best_miou)
                else:
                    epochs_no_improve += 1
                    
                if epoch % 5 == 0:
                    self.save_checkpoint(f"epoch_{epoch}.pth", epoch, val_results["miou"])
                
                if epochs_no_improve >= patience:
                    print(f"\n[INFO] Early stopping triggered after {patience} epochs without improvement.")
                    break
                    
            print(f"[OK] Training complete. Best mIoU: {best_miou:.4f}")

    def save_checkpoint(self, filename: str, epoch: int, miou: float):
        if not self.output_dir:
            return
            
        save_path = self.output_dir / filename
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'miou': miou,
        }, save_path)
        
        # Log model as artifact in MLflow if best
        if "best" in filename:
            mlflow.log_artifact(str(save_path))
            print(f"  [INFO] Saved best model to {save_path}")
