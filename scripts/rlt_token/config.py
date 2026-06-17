from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RLTokenExtractConfig:
    policy_path: Path = REPO_ROOT / "outputs/rlt_vla/ur3e_smolvla_0614/checkpoints/030000/pretrained_model"
    output_dir: Path = REPO_ROOT / "outputs/rlt_token"
    device: str = "cuda"
    batch_size: int = 16
    num_workers: int = 8
    video_backend: str = "pyav"
    token_source: str = "context"
    token_name: str = "rl_token"
    amp: bool = True
    offline: bool = True
    save_prefix_sequence: bool = False


CFG = RLTokenExtractConfig()


@dataclass(frozen=True)
class RLTStage1TrainConfig:
    policy_path: Path = REPO_ROOT / "outputs/rlt_vla/ur3e_smolvla_0614/checkpoints/030000/pretrained_model"
    output_dir: Path = REPO_ROOT / "outputs/rlt_stage1"
    device: str = "cuda"
    batch_size: int = 16
    num_workers: int = 8
    steps: int = 10000
    val_fraction: float = 0.05
    val_batches: int = 32
    log_freq: int = 20
    save_freq: int = 1000
    seed: int = 1000
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    hidden_dim: int = 512
    z_dim: int = 256
    architecture: str = "sequence"
    encoder_layers: int = 2
    encoder_heads: int = 8
    dropout: float = 0.1
    expert_loss_weight: float = 1.0
    video_backend: str = "pyav"
    amp: bool = True
    offline: bool = True


STAGE1_CFG = RLTStage1TrainConfig()
