from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RltGateTrainConfig:
    dataset: Path = Path("datasets/lerobot-export(3)")
    output_dir: Path = Path("outputs/rlt_gate")
    camera: str = "both"  # front, wrist, or both
    model: str = "tiny"  # tiny or resnet18
    image_size: int = 160
    batch_size: int = 96
    num_workers: int = 4
    epochs: int = 40
    lr: float = 3e-4
    weight_decay: float = 1e-4
    val_ratio: float = 0.2
    seed: int = 42
    label_smoothing: float = 0.02
    sample_stride: int = 1
    cache_images: str = "ram"  # ram or none
    device: str = "auto"
    amp: bool = True
    threshold: float = 0.5
    positive_threshold: float = 0.6
    negative_threshold: float = 0.4


CFG = RltGateTrainConfig()
