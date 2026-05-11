"""
DRSA-Net Configuration
======================
Single source of truth for all hyperparameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class DRSAConfig:
    # ------------------------------------------------------------------ #
    #  Paths
    # ------------------------------------------------------------------ #
    dataset_base: str = "Lettuce_disease_datasets_split"
    felz_masks_base: str = "felzenszwalb_masks_output"
    output_dir: str = "drsa_net_output"
    splits: Tuple[str, ...] = ("train", "validation", "test")

    # ------------------------------------------------------------------ #
    #  Dataset
    # ------------------------------------------------------------------ #
    # 8 disease classes from folder names
    class_names: Tuple[str, ...] = (
        "BACT", "DML", "HLTY", "PML", "SBL", "SPW", "VIRL", "WLBL"
    )
    # Image resize target (H, W)
    img_size: Tuple[int, int] = (256, 256)
    # Max superpixels to keep per image (pad/truncate)
    max_superpixels: int = 512

    # ------------------------------------------------------------------ #
    #  CLAHE
    # ------------------------------------------------------------------ #
    clahe_clip_limit: float = 10.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)

    # ------------------------------------------------------------------ #
    #  Watershed
    # ------------------------------------------------------------------ #
    watershed_compactness: float = 0.002
    watershed_min_size: int = 10

    # ------------------------------------------------------------------ #
    #  Multi-stream encoder
    # ------------------------------------------------------------------ #
    # Number of channels output by each branch BEFORE projection
    branch_channels: int = 64
    # Shared embedding dim after 4-branch projection
    embed_dim: int = 256
    # Spatial downsampling factor inside each branch (stride product)
    encoder_stride: int = 4   # H→H/4, W→W/4

    # ------------------------------------------------------------------ #
    #  Region-Aware Transformer
    # ------------------------------------------------------------------ #
    num_transformer_layers: int = 8
    num_attention_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    # k-hop neighborhood for semi-global attention mode
    attention_k_hop: int = 2
    # Learned gating init (0.0 = local mode, 1.0 = semi-global mode)
    gate_init: float = 0.5

    # ------------------------------------------------------------------ #
    #  CAM generation
    # ------------------------------------------------------------------ #
    cam_propagation_alpha: float = 0.5   # α in Y^(t+1) = αAY^(t)+(1-α)Y^(0)
    cam_propagation_steps: int = 3       # T
    cam_threshold: float = 0.3          # for region growing

    # ------------------------------------------------------------------ #
    #  Training
    # ------------------------------------------------------------------ #
    training_mode: str = "weakly_supervised"   # "self_supervised" | "weakly_supervised"
    batch_size: int = 4
    num_epochs: int = 80
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    use_amp: bool = False         # Disabled to prevent GTX 1650 Conv2d FP16 NaNs
    grad_clip: float = 1.0
    num_workers: int = 0

    # Loss weights
    loss_cls_weight: float = 1.0       # classification (weakly-supervised)
    loss_cam_consist_weight: float = 0.5   # CAM consistency across augmentations
    loss_contrastive_weight: float = 0.3   # NT-Xent contrastive
    loss_compact_weight: float = 0.2       # superpixel compactness
    loss_graph_smooth_weight: float = 0.2  # graph smoothness

    # Contrastive
    contrastive_temperature: float = 0.07

    # ------------------------------------------------------------------ #
    #  Device / checkpointing
    # ------------------------------------------------------------------ #
    device: Optional[str] = None         # None = auto-detect cuda/cpu
    checkpoint_dir: str = "drsa_net_output/checkpoints"
    log_dir: str = "drsa_net_output/logs"
    save_every_n_epochs: int = 5
    resume_checkpoint: Optional[str] = None

    # ------------------------------------------------------------------ #
    #  MLflow
    # ------------------------------------------------------------------ #
    mlflow_experiment: str = "DRSA-Net_Lettuce_Disease"
    mlflow_tracking_uri: str = "mlruns"

    # ------------------------------------------------------------------ #
    #  Misc
    # ------------------------------------------------------------------ #
    seed: int = 42
    samples_per_class: Optional[int] = None  # None = use all

    def validate(self) -> None:
        """Basic sanity checks."""
        assert self.embed_dim % self.num_attention_heads == 0, (
            f"embed_dim ({self.embed_dim}) must be divisible by "
            f"num_attention_heads ({self.num_attention_heads})"
        )
        assert self.training_mode in {"self_supervised", "weakly_supervised"}, (
            f"Unknown training_mode: {self.training_mode}"
        )
        assert 0.0 < self.cam_propagation_alpha < 1.0, "α must be in (0, 1)"

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def class_to_idx(self) -> dict:
        return {name: idx for idx, name in enumerate(self.class_names)}

    @property
    def felz_branch_in_channels(self) -> int:
        """Felzenszwalb branch takes 2 masks (raw1 + raw2) → 2 input channels."""
        return 2

    @property
    def total_projection_in_channels(self) -> int:
        """4 branches × branch_channels concatenated."""
        return 4 * self.branch_channels
