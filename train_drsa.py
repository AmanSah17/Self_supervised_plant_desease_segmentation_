"""
DRSA-Net Training Entry Point
==============================
Usage:
    python train_drsa.py
    python train_drsa.py --mode self_supervised
    python train_drsa.py --resume drsa_net_output/checkpoints/drsa_epoch0040_best.pth
    python train_drsa.py --smoke  (5-sample quick smoke test)
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

# Disable cuDNN to prevent NaN issues on GTX 1650 (known driver/cuDNN bug)
cudnn.enabled = False

from drsa_net.config import DRSAConfig
from drsa_net.training.trainer import DRSATrainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DRSA-Net training")
    p.add_argument('--mode', type=str, default='weakly_supervised',
                   choices=['weakly_supervised', 'self_supervised'],
                   help='Training mode')
    p.add_argument('--epochs', type=int, default=None,
                   help='Override num_epochs in config')
    p.add_argument('--batch', type=int, default=None,
                   help='Override batch_size in config')
    p.add_argument('--lr', type=float, default=None,
                   help='Override learning_rate in config')
    p.add_argument('--resume', type=str, default=None,
                   help='Path to checkpoint to resume from')
    p.add_argument('--smoke', action='store_true',
                   help='Smoke test: 2 samples per class, 2 epochs')
    p.add_argument('--no-amp', action='store_true',
                   help='Disable mixed-precision training')
    p.add_argument('--embed-dim', type=int, default=None)
    p.add_argument('--layers', type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Build config ───────────────────────────────────────────────────────
    config = DRSAConfig()
    config.training_mode = args.mode

    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch is not None:
        config.batch_size = args.batch
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.resume is not None:
        config.resume_checkpoint = args.resume
    if args.no_amp:
        config.use_amp = False
    if args.embed_dim is not None:
        config.embed_dim = args.embed_dim
    if args.layers is not None:
        config.num_transformer_layers = args.layers
    if args.smoke:
        config.samples_per_class = 2
        config.num_epochs        = 2
        config.batch_size        = 2
        config.num_workers       = 0
        config.use_amp           = False
        print("=" * 60)
        print("SMOKE TEST MODE — 2 samples/class, 2 epochs")
        print("=" * 60)

    config.validate()
    set_seed(config.seed)

    # ── Print config ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DRSA-Net Configuration")
    print("=" * 60)
    for field_name, value in config.__dict__.items():
        print(f"  {field_name:<30} {value}")
    print("=" * 60 + "\n")

    # ── Launch training ────────────────────────────────────────────────────
    trainer = DRSATrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
