from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class BackboneNotes:
    name: str
    objective: str
    strengths: list[str]
    caveats: list[str]


class SSLBackboneAdapter(ABC):
    @abstractmethod
    def load(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def notes(self) -> BackboneNotes:
        raise NotImplementedError


class DINOv2BackboneAdapter(SSLBackboneAdapter):
    def __init__(self, model_name: str = "dinov2_vitb14"):
        self.model_name = model_name

    def load(self) -> Any:
        import torch
        return torch.hub.load("facebookresearch/dinov2", self.model_name)

    def notes(self) -> BackboneNotes:
        return BackboneNotes(
            name=f"DINOv2::{self.model_name}",
            objective="Self-distilled visual foundation features for patch- and pixel-level transfer.",
            strengths=[
                "Strong off-the-shelf dense visual features.",
                "Good first choice for healthy-feature modeling before segmentation fine-tuning.",
                "Official repo already includes semantic segmentation examples.",
            ],
            caveats=[
                "Heavy models can exceed GTX 1650 memory if fine-tuned end-to-end.",
                "Prefer frozen feature extraction first, then lightweight heads.",
            ],
        )


class MAEBackboneAdapter(SSLBackboneAdapter):
    def __init__(self, checkpoint_path: str | None = None):
        self.checkpoint_path = checkpoint_path

    def load(self) -> Any:
        raise NotImplementedError(
            "MAE is included as a planned adapter. Provide a local checkpoint and "
            "training recipe before enabling it in this repo."
        )

    def notes(self) -> BackboneNotes:
        return BackboneNotes(
            name="MAE",
            objective="Masked image reconstruction for domain-adapted self-supervised pretraining.",
            strengths=[
                "Good option when you want domain-specific pretraining on your lettuce images.",
                "Can learn strong local texture structure when augmentation is tuned well.",
            ],
            caveats=[
                "Requires a dedicated pretraining run and checkpoint management.",
                "More engineering overhead than starting from frozen DINOv2 features.",
            ],
        )
