"""
DRSA-Net Trainer
================
Full training loop with:
  - Two-view augmentation per batch (for CAM consistency + contrastive)
  - Mixed precision (AMP)
  - Cosine LR schedule with linear warmup
  - Gradient clipping
  - MLflow metric logging
  - Periodic checkpointing with resume support
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from drsa_net.config import DRSAConfig
from drsa_net.model.drsa_net import DRSANet
from drsa_net.training.losses import DRSALoss
from drsa_net.data.dataset import build_dataloaders, DRSADataset


# --------------------------------------------------------------------------- #
#  Warmup + Cosine LR                                                          #
# --------------------------------------------------------------------------- #

class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
    ):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.min_lr        = min_lr
        self.base_lrs      = [pg['lr'] for pg in optimizer.param_groups]

    def step(self, epoch: int) -> None:
        if epoch < self.warmup_epochs:
            scale = (epoch + 1) / max(self.warmup_epochs, 1)
        else:
            import math
            progress = (epoch - self.warmup_epochs) / max(
                self.total_epochs - self.warmup_epochs, 1
            )
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = max(base_lr * scale, self.min_lr)

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']


# --------------------------------------------------------------------------- #
#  Training utilities                                                          #
# --------------------------------------------------------------------------- #

def _batch_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _forward_one_view(
    model: DRSANet,
    batch: Dict,
    return_cam: bool = True,
) -> Dict:
    return model(
        rgb       = batch['rgb'],
        clahe     = batch['clahe'],
        felz1     = batch['felz1'],
        felz2     = batch['felz2'],
        watershed = batch['watershed'],
        segments  = batch['segments'],
        return_cam= return_cam,
    )


# --------------------------------------------------------------------------- #
#  Metrics                                                                     #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


# --------------------------------------------------------------------------- #
#  Trainer                                                                     #
# --------------------------------------------------------------------------- #

class DRSATrainer:
    """
    Two-view trainer for DRSA-Net.

    Each batch is augmented TWICE (view1, view2) using the same base dataset
    but with different random augmentation seeds.  Both views share the same
    Felzenszwalb masks (mask paths are deterministic).
    """

    def __init__(self, config: DRSAConfig):
        self.config = config
        self.device = torch.device('cpu') # FORCE CPU for debug
        print(f"[DRSATrainer] Device: {self.device}")

        # Directories
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

        # Model
        self.model = DRSANet(config).to(self.device)
        print(self.model.summary())

        # Two datasets for two augmented views (same split, different RNG)
        self.train_loader, self.val_loader = build_dataloaders(config)

        # For two-view: we also build a second training loader
        train_ds_v2 = DRSADataset(config, split='train', augment=True)
        self.train_loader_v2 = DataLoader(
            train_ds_v2,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=DRSADataset.collate_fn,
        )

        # Loss
        self.criterion = DRSALoss(config).to(self.device)

        # Optimizer (include both model and adaptive loss params)
        # We give the loss parameters a slightly higher learning rate for faster adaptation
        params = [
            {'params': self.model.parameters()},
            {'params': self.criterion.adaptive_mtl.parameters(), 'lr': config.learning_rate * 10, 'weight_decay': 0.0}
        ]
        
        self.optimizer = AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # LR scheduler
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=config.warmup_epochs,
            total_epochs=config.num_epochs,
        )

        # Mixed precision
        self.scaler = GradScaler(enabled=config.use_amp)

        # State
        self.start_epoch  = 0
        self.best_val_acc = 0.0

        # Resume
        if config.resume_checkpoint is not None:
            self._load_checkpoint(config.resume_checkpoint)

        # MLflow
        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(config.mlflow_tracking_uri)
            mlflow.set_experiment(config.mlflow_experiment)

    # ---------------------------------------------------------------------- #
    #  Checkpoint I/O                                                          #
    # ---------------------------------------------------------------------- #

    def _save_checkpoint(self, epoch: int, tag: str = "") -> None:
        name = f"drsa_epoch{epoch:04d}{tag}.pth"
        path = Path(self.config.checkpoint_dir) / name
        torch.save({
            'epoch':          epoch,
            'model':          self.model.state_dict(),
            'optimizer':      self.optimizer.state_dict(),
            'scaler':         self.scaler.state_dict(),
            'best_val_acc':   self.best_val_acc,
            'config':         self.config,
        }, path)
        print(f"  [OK] Checkpoint saved: {path}")

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scaler.load_state_dict(ckpt['scaler'])
        self.start_epoch  = ckpt['epoch'] + 1
        self.best_val_acc = ckpt.get('best_val_acc', 0.0)
        print(f"  [OK] Resumed from epoch {ckpt['epoch']}: {path}")

    # ---------------------------------------------------------------------- #
    #  Training epoch                                                          #
    # ---------------------------------------------------------------------- #

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        running = {k: 0.0 for k in
                   ['total', 'cls', 'cam_consist', 'contrastive',
                    'compactness', 'graph_smooth', 'acc']}
        n_batches = 0

        # Zip two augmented views
        loader_v2_iter = iter(self.train_loader_v2)
        bar = tqdm(self.train_loader, desc=f"Epoch {epoch:03d} [train]",
                   ncols=110, leave=False)

        for batch1 in bar:
            try:
                batch2 = next(loader_v2_iter)
            except StopIteration:
                loader_v2_iter = iter(self.train_loader_v2)
                batch2 = next(loader_v2_iter)

            batch1 = _batch_to_device(batch1, self.device)
            batch2 = _batch_to_device(batch2, self.device)
            labels = batch1['label']

            if n_batches == 0 and epoch == 0:
                print("\n  [DEBUG] First Batch Stats:")
                for k in ('rgb', 'clahe', 'felz1', 'felz2', 'watershed'):
                    v = batch1[k]
                    print(f"    - {k:<10}: min={v.min():.4f}, max={v.max():.4f}, nan={torch.isnan(v).any()}")

            # Build adjacency from view1 segments (same for both — mask is deterministic)
            # We extract the local_mask from tokenizer inside model forward;
            # for loss computation we need adj_matrix separately.
            # Quick approach: forward pass returns 'seg_ds', rebuild adjacency.

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.config.use_amp):
                # Forward view 1 (with CAM)
                out1 = _forward_one_view(self.model, batch1, return_cam=True)
                # Forward view 2 (with CAM, stop grad on cls for contrastive)
                out2 = _forward_one_view(self.model, batch2, return_cam=True)

                # Build adj from seg_ds (reuse tokenizer helper)
                from drsa_net.model.superpixel_tokenizer import build_adjacency_matrix
                B = labels.shape[0]
                N_sp = out1['tokens_out'].shape[1]
                seg_ds = out1['seg_ds']
                # Build per-batch adjacency
                adj_list = []
                for b_idx in range(B):
                    _, remapped = torch.unique(seg_ds[b_idx], return_inverse=True)
                    remapped = remapped.clamp(0, N_sp - 1)
                    adj_b = build_adjacency_matrix(remapped, N_sp, k_hop=1)
                    adj_list.append(adj_b.float())
                adj_matrix = torch.stack(adj_list)   # (B, N_sp, N_sp)

                losses = self.criterion(
                    out1, out2, labels, adj_matrix,
                    return_components=True,
                )

            self.scaler.scale(losses['total']).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Metrics
            acc = compute_accuracy(out1['cls_logits'], labels)
            for k in ['total', 'cls', 'cam_consist', 'contrastive',
                      'compactness', 'graph_smooth']:
                running[k] += losses[k].item()
            running['acc'] += acc
            n_batches += 1

            bar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'acc':  f"{acc:.3f}",
                'lr':   f"{self.scheduler.get_lr():.2e}",
            })

        return {k: v / max(n_batches, 1) for k, v in running.items()}

    # ---------------------------------------------------------------------- #
    #  Validation epoch                                                        #
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_acc  = 0.0
        n_batches  = 0

        bar = tqdm(self.val_loader, desc=f"Epoch {epoch:03d} [val]  ",
                   ncols=110, leave=False)
        for batch in bar:
            batch = _batch_to_device(batch, self.device)
            labels = batch['label']

            with autocast(enabled=self.config.use_amp):
                out = _forward_one_view(self.model, batch, return_cam=False)
                loss = nn.CrossEntropyLoss()(out['cls_logits'], labels)

            total_loss += loss.item()
            total_acc  += compute_accuracy(out['cls_logits'], labels)
            n_batches  += 1

        return {
            'loss': total_loss / max(n_batches, 1),
            'acc':  total_acc  / max(n_batches, 1),
        }

    # ---------------------------------------------------------------------- #
    #  Main training loop                                                      #
    # ---------------------------------------------------------------------- #

    def train(self) -> None:
        cfg = self.config

        run_ctx = (
            mlflow.start_run() if MLFLOW_AVAILABLE
            else _NullContext()
        )

        with run_ctx:
            if MLFLOW_AVAILABLE:
                mlflow.log_params({
                    'embed_dim':         cfg.embed_dim,
                    'num_layers':        cfg.num_transformer_layers,
                    'num_heads':         cfg.num_attention_heads,
                    'batch_size':        cfg.batch_size,
                    'learning_rate':     cfg.learning_rate,
                    'num_epochs':        cfg.num_epochs,
                    'training_mode':     cfg.training_mode,
                    'cam_prop_alpha':    cfg.cam_propagation_alpha,
                    'cam_prop_steps':    cfg.cam_propagation_steps,
                    'k_hop':             cfg.attention_k_hop,
                    'clahe_clip_limit':  cfg.clahe_clip_limit,
                })

            for epoch in range(self.start_epoch, cfg.num_epochs):
                self.scheduler.step(epoch)
                t0 = time.time()

                train_metrics = self._train_epoch(epoch)
                val_metrics   = self._val_epoch(epoch)

                elapsed = time.time() - t0

                print(
                    f"Epoch {epoch:03d} | "
                    f"train_loss={train_metrics['total']:.4f} "
                    f"train_acc={train_metrics['acc']:.3f} | "
                    f"val_loss={val_metrics['loss']:.4f} "
                    f"val_acc={val_metrics['acc']:.3f} | "
                    f"{elapsed:.1f}s"
                )

                if MLFLOW_AVAILABLE:
                    mtl_weights = self.criterion.get_mtl_weights()
                    mlflow.log_metrics({
                        'train/total_loss':     train_metrics['total'],
                        'train/cls_loss':       train_metrics['cls'],
                        'train/cam_loss':       train_metrics['cam_consist'],
                        'train/contrastive':    train_metrics['contrastive'],
                        'train/compactness':    train_metrics['compactness'],
                        'train/graph_smooth':   train_metrics['graph_smooth'],
                        'train/accuracy':       train_metrics['acc'],
                        'val/loss':             val_metrics['loss'],
                        'val/accuracy':         val_metrics['acc'],
                        'lr':                   self.scheduler.get_lr(),
                        'mtl_weight/cls':       mtl_weights[0],
                        'mtl_weight/cam':       mtl_weights[1],
                        'mtl_weight/cont':      mtl_weights[2],
                        'mtl_weight/comp':      mtl_weights[3],
                        'mtl_weight/gs':        mtl_weights[4],
                    }, step=epoch)

                # Save best
                if val_metrics['acc'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['acc']
                    self._save_checkpoint(epoch, tag='_best')

                # Periodic save
                if (epoch + 1) % cfg.save_every_n_epochs == 0:
                    self._save_checkpoint(epoch)

        print(f"\n[OK] Training complete. Best val accuracy: {self.best_val_acc:.4f}")


class _NullContext:
    """Fallback context manager when MLflow is unavailable."""
    def __enter__(self): return self
    def __exit__(self, *args): pass
