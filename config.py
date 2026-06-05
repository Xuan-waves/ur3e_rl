from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SmolVLATrainConfig:
    # Dataset and output.
    dataset: Path = REPO_ROOT / "datasets" / "ur3e_lerobot_vr_impedance_20260531_172425"
    repo_id: str | None = None
    output_root: Path = REPO_ROOT / "outputs" / "train"
    job_name: str = "ur3e_smolvla"

    # 24 GB GPU defaults. Reduce batch_size first if CUDA OOM appears.
    steps: int = 100000
    batch_size: int = 32
    num_workers: int = 32
    log_freq: int = 10000
    save_freq: int = 10000
    seed: int = 1000

    # Runtime.
    device: str = "cuda"
    amp: bool = True
    use_imagenet_stats: bool = True
    tolerance_s: float = 1e-4

    # SmolVLA model.
    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    load_vlm_weights: bool = False
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True

    # Temporal/action setup. The UR3e collector runs at 30 Hz, so chunk=50 is ~1.67 s.
    n_obs_steps: int = 2
    chunk_size: int = 50
    n_action_steps: int = 50

    # Data/model dimensions.
    max_state_dim: int = 32
    max_action_dim: int = 32
    image_size: int = 256
    num_vlm_layers: int = 16
    num_expert_layers: int = -1
    expert_width_multiplier: float = 0.5

    # Cache inside the repo so training does not write to a read-only home cache.
    hf_home: Path = REPO_ROOT / ".cache" / "huggingface"


CFG = SmolVLATrainConfig()
