from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class NoRotvecSmolVLATrainConfig:
    # Dataset produced by strip_fixed_rotvec_dataset.py.
    dataset: Path = REPO_ROOT / "datasets" / "ur3e_lerobot_vr_impedance_20260605_170753_no_rotvec"
    repo_id: str | None = None
    output_root: Path = REPO_ROOT / "outputs" / "train"
    job_name: str = "ur3e_smolvla_no_rotvec"

    # 24 GB GPU defaults. Reduce batch_size first if CUDA OOM appears.
    steps: int = 20000
    batch_size: int = 16
    num_workers: int = 8
    log_freq: int = 1000
    save_freq: int = 1000
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

    # Collector runs at 30 Hz. chunk=50 is about 1.67 s.
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    # The dataset is 5D, but SmolVLA can keep the same padded dimensions as
    # the servoJ/impedance variants so checkpoints stay easy to compare.
    max_state_dim: int = 32
    max_action_dim: int = 32
    image_size: int = 256
    num_vlm_layers: int = 16
    num_expert_layers: int = -1
    expert_width_multiplier: float = 0.75

    # Keep model/cache writes inside the repo.
    hf_home: Path = REPO_ROOT / ".cache" / "huggingface"


CFG = NoRotvecSmolVLATrainConfig()
