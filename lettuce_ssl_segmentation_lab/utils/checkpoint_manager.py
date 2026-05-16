"""
Checkpoint management for resumable training and feature extraction.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional
from datetime import datetime


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""
    stage: str
    timestamp: str
    epoch: int
    batch_idx: int
    total_samples_processed: int
    status: str  # "in_progress", "completed", "failed"
    error_msg: Optional[str] = None
    metrics: dict = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}


class CheckpointManager:
    """Manage checkpoints for resumable training."""
    
    def __init__(self, checkpoint_dir: Path, stage_name: str):
        self.checkpoint_dir = checkpoint_dir / stage_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.stage_name = stage_name
        self.metadata_path = self.checkpoint_dir / "checkpoint_metadata.json"
        self.latest_checkpoint = None
        self.load_latest_checkpoint()
    
    def save_checkpoint(
        self,
        data: dict,
        epoch: int,
        batch_idx: int,
        total_samples: int,
        status: str = "in_progress",
        metrics: Optional[dict] = None,
        error_msg: Optional[str] = None,
    ) -> Path:
        """Save a checkpoint with metadata."""
        checkpoint_name = f"checkpoint_epoch{epoch:04d}_batch{batch_idx:06d}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Save data
        import torch
        torch.save(data, checkpoint_path)
        
        # Save metadata
        metadata = CheckpointMetadata(
            stage=self.stage_name,
            timestamp=datetime.now().isoformat(),
            epoch=epoch,
            batch_idx=batch_idx,
            total_samples_processed=total_samples,
            status=status,
            error_msg=error_msg,
            metrics=metrics or {},
        )
        
        # Update metadata file
        metadata_dict = asdict(metadata)
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        self.latest_checkpoint = (checkpoint_path, metadata)
        return checkpoint_path
    
    def load_latest_checkpoint(self) -> tuple[Optional[dict], Optional[CheckpointMetadata]]:
        """Load the latest checkpoint if available."""
        if not self.metadata_path.exists():
            return None, None
        
        try:
            with open(self.metadata_path, 'r') as f:
                metadata_dict = json.load(f)
            
            # Find latest checkpoint file
            checkpoint_files = sorted(self.checkpoint_dir.glob("checkpoint_*.pt"))
            if not checkpoint_files:
                return None, None
            
            latest_path = checkpoint_files[-1]
            
            import torch
            data = torch.load(latest_path, map_location='cpu')
            
            metadata = CheckpointMetadata(
                stage=metadata_dict.get("stage", self.stage_name),
                timestamp=metadata_dict.get("timestamp", ""),
                epoch=metadata_dict.get("epoch", 0),
                batch_idx=metadata_dict.get("batch_idx", 0),
                total_samples_processed=metadata_dict.get("total_samples_processed", 0),
                status=metadata_dict.get("status", "unknown"),
                error_msg=metadata_dict.get("error_msg"),
                metrics=metadata_dict.get("metrics", {}),
            )
            
            self.latest_checkpoint = (latest_path, metadata)
            return data, metadata
        
        except Exception as e:
            print(f"[ERROR] Error loading checkpoint: {str(e)}")
            return None, None
    
    def get_resume_info(self) -> tuple[int, int, int]:
        """Get (epoch, batch_idx, total_samples) to resume from."""
        if self.latest_checkpoint is None:
            return 0, 0, 0
        
        _, metadata = self.latest_checkpoint
        return metadata.epoch, metadata.batch_idx, metadata.total_samples_processed
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 3) -> None:
        """Keep only the last N checkpoints to save space."""
        checkpoint_files = sorted(self.checkpoint_dir.glob("checkpoint_*.pt"))
        if len(checkpoint_files) > keep_last_n:
            for old_ckpt in checkpoint_files[:-keep_last_n]:
                old_ckpt.unlink()
                print(f"Cleaned up old checkpoint: {old_ckpt.name}")
