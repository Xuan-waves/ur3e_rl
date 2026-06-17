#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import warnings
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

warnings.filterwarnings(
    "ignore",
    message="The video decoding and encoding capabilities of torchvision are deprecated.*",
    category=UserWarning,
)

from scripts.rlt_token.config import STAGE1_CFG  # noqa: E402
from scripts.rlt_token.extract_rl_token import (  # noqa: E402
    jsonable_config,
    load_train_dataset_config,
    normalize_policy_path,
)


def set_offline_env() -> None:
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache/huggingface"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(REPO_ROOT / ".cache/huggingface/datasets"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_fps(dataset_root: Path) -> float:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing dataset metadata: {info_path}")
    info = load_json(info_path)
    fps = float(info.get("fps", 30.0))
    if fps <= 0:
        raise ValueError(f"Invalid dataset fps={fps} in {info_path}")
    return fps


def default_output_dir(policy_path: Path, output_root: Path) -> Path:
    step = policy_path.parent.name
    run_name = policy_path.parents[2].name if len(policy_path.parents) > 2 else policy_path.parent.name
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_root / f"rlt_stage1_{run_name}_{step}_{stamp}"


def stage1_config_jsonable() -> dict[str, Any]:
    out = {}
    for key, value in asdict(STAGE1_CFG).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


class RLTStage1AutoEncoder(nn.Module):
    """Small encoder-decoder that compresses frozen VLA embeddings into z_rl."""

    def __init__(
        self,
        *,
        vlm_dim: int,
        expert_dim: int,
        hidden_dim: int,
        z_dim: int,
        encoder_layers: int,
        encoder_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.vlm_dim = int(vlm_dim)
        self.expert_dim = int(expert_dim)
        self.hidden_dim = int(hidden_dim)
        self.z_dim = int(z_dim)

        self.vlm_in = nn.Sequential(nn.LayerNorm(vlm_dim), nn.Linear(vlm_dim, hidden_dim))
        self.expert_in = nn.Sequential(nn.LayerNorm(expert_dim), nn.Linear(expert_dim, hidden_dim))
        self.rl_query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.rl_query, mean=0.0, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=encoder_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=encoder_layers)
        self.to_z = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, z_dim))
        self.from_z = nn.Sequential(nn.Linear(z_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.vlm_out = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, vlm_dim))
        self.expert_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, expert_dim),
        )

    def encode(self, vlm: torch.Tensor, expert: torch.Tensor) -> torch.Tensor:
        bsz = vlm.shape[0]
        tokens = torch.stack([self.vlm_in(vlm), self.expert_in(expert)], dim=1)
        query = self.rl_query.expand(bsz, -1, -1)
        tokens = torch.cat([tokens, query], dim=1)
        encoded = self.encoder(tokens)
        return self.to_z(encoded[:, -1])

    def forward(self, vlm: torch.Tensor, expert: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encode(vlm, expert)
        hidden = self.from_z(z)
        return {
            "z_rl": z,
            "vlm_recon": self.vlm_out(hidden),
            "expert_recon": self.expert_out(hidden),
        }


class RLTStage1SequenceAutoEncoder(nn.Module):
    """Bottleneck encoder-decoder over frozen VLA token sequences.

    The encoder reads the VLA prefix tokens, expert/action suffix tokens, and a
    learnable RL query. The decoder reconstructs the token sequences from z_rl
    plus learned position queries only, keeping z_rl as the information
    bottleneck.
    """

    def __init__(
        self,
        *,
        vlm_dim: int,
        expert_dim: int,
        max_vlm_tokens: int,
        max_expert_tokens: int,
        hidden_dim: int,
        z_dim: int,
        encoder_layers: int,
        encoder_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.vlm_dim = int(vlm_dim)
        self.expert_dim = int(expert_dim)
        self.max_vlm_tokens = int(max_vlm_tokens)
        self.max_expert_tokens = int(max_expert_tokens)
        self.hidden_dim = int(hidden_dim)
        self.z_dim = int(z_dim)

        self.vlm_in = nn.Sequential(nn.LayerNorm(vlm_dim), nn.Linear(vlm_dim, hidden_dim))
        self.expert_in = nn.Sequential(nn.LayerNorm(expert_dim), nn.Linear(expert_dim, hidden_dim))
        self.type_embed = nn.Parameter(torch.zeros(1, 3, hidden_dim))
        self.rl_query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.type_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.rl_query, mean=0.0, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=encoder_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=encoder_layers)
        self.to_z = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, z_dim))
        self.from_z = nn.Sequential(nn.Linear(z_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))

        self.vlm_pos = nn.Parameter(torch.zeros(1, self.max_vlm_tokens, hidden_dim))
        self.expert_pos = nn.Parameter(torch.zeros(1, self.max_expert_tokens, hidden_dim))
        nn.init.normal_(self.vlm_pos, mean=0.0, std=0.02)
        nn.init.normal_(self.expert_pos, mean=0.0, std=0.02)
        self.vlm_out = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, vlm_dim))
        self.expert_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, expert_dim),
        )

    def encode(
        self,
        vlm_seq: torch.Tensor,
        expert_seq: torch.Tensor,
        vlm_mask: torch.Tensor | None = None,
        expert_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz = vlm_seq.shape[0]
        if vlm_seq.shape[1] > self.max_vlm_tokens or expert_seq.shape[1] > self.max_expert_tokens:
            raise ValueError(
                f"Stage1 sequence length exceeds checkpoint capacity: "
                f"vlm={vlm_seq.shape[1]}/{self.max_vlm_tokens}, "
                f"expert={expert_seq.shape[1]}/{self.max_expert_tokens}"
            )
        vlm_tokens = self.vlm_in(vlm_seq) + self.type_embed[:, 0:1, :]
        expert_tokens = self.expert_in(expert_seq) + self.type_embed[:, 1:2, :]
        query = self.rl_query.expand(bsz, -1, -1) + self.type_embed[:, 2:3, :]
        tokens = torch.cat([vlm_tokens, expert_tokens, query], dim=1)
        if vlm_mask is not None and expert_mask is not None:
            valid = torch.cat(
                [
                    vlm_mask.to(dtype=torch.bool),
                    expert_mask.to(dtype=torch.bool),
                    torch.ones((bsz, 1), dtype=torch.bool, device=tokens.device),
                ],
                dim=1,
            )
            padding_mask = ~valid
        else:
            padding_mask = None
        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)
        return self.to_z(encoded[:, -1])

    def forward(
        self,
        vlm_seq: torch.Tensor,
        expert_seq: torch.Tensor,
        vlm_mask: torch.Tensor | None = None,
        expert_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        z = self.encode(vlm_seq, expert_seq, vlm_mask, expert_mask)
        hidden = self.from_z(z).unsqueeze(1)
        vlm_len = vlm_seq.shape[1]
        expert_len = expert_seq.shape[1]
        vlm_hidden = hidden + self.vlm_pos[:, :vlm_len, :]
        expert_hidden = hidden + self.expert_pos[:, :expert_len, :]
        return {
            "z_rl": z,
            "vlm_seq_recon": self.vlm_out(vlm_hidden),
            "expert_seq_recon": self.expert_out(expert_hidden),
        }


class FrozenSmolVLAEmbeddingReader:
    def __init__(self, policy_path: Path, device: str, *, amp: bool) -> None:
        import lerobot.policies.smolvla.processor_smolvla  # noqa: F401
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.utils.constants import (
            ACTION,
            OBS_LANGUAGE_ATTENTION_MASK,
            OBS_LANGUAGE_TOKENS,
            POLICY_PREPROCESSOR_DEFAULT_NAME,
        )

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.amp = bool(amp and device == "cuda")
        self.action_key = ACTION
        self.obs_language_tokens = OBS_LANGUAGE_TOKENS
        self.obs_language_attention_mask = OBS_LANGUAGE_ATTENTION_MASK
        self.make_att_2d_masks = make_att_2d_masks

        self.policy = SmolVLAPolicy.from_pretrained(
            policy_path,
            cli_overrides=[f"--device={device}"],
            local_files_only=True,
        )
        self.policy.eval()
        for param in self.policy.parameters():
            param.requires_grad_(False)

        self.preprocessor = PolicyProcessorPipeline.from_pretrained(
            policy_path,
            config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
            local_files_only=True,
            overrides={"device_processor": {"device": device}},
        )
        self.input_keys = {
            *self.policy.config.input_features.keys(),
            OBS_LANGUAGE_TOKENS,
            OBS_LANGUAGE_ATTENTION_MASK,
            ACTION,
        }

    @staticmethod
    def masked_mean(embs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(dtype=embs.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (embs * weights).sum(dim=1) / denom

    def read(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        autocast = torch.autocast(device_type="cuda") if self.amp else torch.no_grad()
        with torch.no_grad(), autocast:
            processed = self.preprocessor(batch)
            processed = {key: value for key, value in processed.items() if key in self.input_keys}
            processed = self.policy._prepare_batch(processed)

            images, img_masks = self.policy.prepare_images(processed)
            state = self.policy.prepare_state(processed)
            actions = self.policy.prepare_action(processed)
            if actions.ndim != 3:
                raise ValueError(
                    f"Stage 1 requires action chunks shaped [B, chunk, action_dim], got {tuple(actions.shape)}. "
                    "Check delta_timestamps when constructing LeRobotDataset."
                )
            if actions.shape[1] != int(self.policy.config.chunk_size):
                raise ValueError(
                    f"Action chunk length {actions.shape[1]} does not match policy chunk_size={self.policy.config.chunk_size}."
                )

            lang_tokens = processed[self.obs_language_tokens]
            lang_masks = processed[self.obs_language_attention_mask]
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.policy.model.embed_prefix(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state=state,
            )
            timestep = torch.zeros(actions.shape[0], dtype=torch.float32, device=actions.device)
            suffix_embs, suffix_pad_masks, suffix_att_masks = self.policy.model.embed_suffix(actions, timestep)

            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
            att_2d_masks = self.make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            (prefix_out, suffix_out), _ = self.policy.model.vlm_with_expert.forward(
                attention_mask=att_2d_masks,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                fill_kv_cache=False,
            )
            vlm = self.masked_mean(prefix_out.float(), prefix_pad_masks)
            expert = self.masked_mean(suffix_out.float(), suffix_pad_masks)
        return {
            "vlm": vlm.detach().clone(),
            "expert": expert.detach().clone(),
            "vlm_seq": prefix_out.float().detach().clone(),
            "expert_seq": suffix_out.float().detach().clone(),
            "vlm_mask": prefix_pad_masks.detach().clone().to(dtype=torch.bool),
            "expert_mask": suffix_pad_masks.detach().clone().to(dtype=torch.bool),
        }


def split_indices(length: int, val_fraction: float, seed: int, max_frames: int | None) -> tuple[list[int], list[int]]:
    indices = list(range(length))
    if max_frames is not None:
        indices = indices[: max(1, min(int(max_frames), len(indices)))]
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_count = max(1, int(round(len(indices) * val_fraction))) if len(indices) > 1 else 0
    val = indices[:val_count]
    train = indices[val_count:] or indices
    return train, val


def cycle(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def mse_pair(
    out: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    *,
    expert_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if "vlm_seq_recon" in out:
        vlm_loss = masked_token_mse(out["vlm_seq_recon"], target["vlm_seq"], target["vlm_mask"])
        expert_loss = masked_token_mse(out["expert_seq_recon"], target["expert_seq"], target["expert_mask"])
    else:
        vlm_loss = F.mse_loss(out["vlm_recon"], target["vlm"])
        expert_loss = F.mse_loss(out["expert_recon"], target["expert"])
    loss = vlm_loss + float(expert_loss_weight) * expert_loss
    return loss, {
        "loss": float(loss.detach().cpu()),
        "vlm": float(vlm_loss.detach().cpu()),
        "expert": float(expert_loss.detach().cpu()),
        "z_norm": float(out["z_rl"].detach().norm(dim=-1).mean().cpu()),
    }


def masked_token_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = (pred - target).pow(2).mean(dim=-1)
    weights = mask.to(dtype=loss.dtype)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def make_stage1_model(cfg: dict[str, Any]) -> nn.Module:
    architecture = str(cfg.get("architecture", "pooled"))
    if architecture == "sequence":
        return RLTStage1SequenceAutoEncoder(
            vlm_dim=int(cfg["vlm_dim"]),
            expert_dim=int(cfg["expert_dim"]),
            max_vlm_tokens=int(cfg["max_vlm_tokens"]),
            max_expert_tokens=int(cfg["max_expert_tokens"]),
            hidden_dim=int(cfg["hidden_dim"]),
            z_dim=int(cfg["z_dim"]),
            encoder_layers=int(cfg["encoder_layers"]),
            encoder_heads=int(cfg["encoder_heads"]),
            dropout=float(cfg.get("dropout", 0.0)),
        )
    if architecture == "pooled":
        return RLTStage1AutoEncoder(
            vlm_dim=int(cfg["vlm_dim"]),
            expert_dim=int(cfg["expert_dim"]),
            hidden_dim=int(cfg["hidden_dim"]),
            z_dim=int(cfg["z_dim"]),
            encoder_layers=int(cfg["encoder_layers"]),
            encoder_heads=int(cfg["encoder_heads"]),
            dropout=float(cfg.get("dropout", 0.0)),
        )
    raise ValueError(f"Unknown Stage1 architecture: {architecture}")


def encode_stage1(model: nn.Module, target: dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(model, RLTStage1SequenceAutoEncoder):
        return model.encode(target["vlm_seq"], target["expert_seq"], target["vlm_mask"], target["expert_mask"])
    return model.encode(target["vlm"], target["expert"])


def forward_stage1(model: nn.Module, target: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if isinstance(model, RLTStage1SequenceAutoEncoder):
        return model(target["vlm_seq"], target["expert_seq"], target["vlm_mask"], target["expert_mask"])
    return model(target["vlm"], target["expert"])


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    reader: FrozenSmolVLAEmbeddingReader,
    loader: DataLoader,
    *,
    device: str,
    batches: int,
    expert_loss_weight: float,
) -> dict[str, float]:
    model.eval()
    acc: dict[str, float] = {"loss": 0.0, "vlm": 0.0, "expert": 0.0, "z_norm": 0.0}
    seen = 0
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= batches:
            break
        target = reader.read(batch)
        target = {key: value.to(device) for key, value in target.items()}
        out = forward_stage1(model, target)
        _loss, metrics = mse_pair(out, target, expert_loss_weight=expert_loss_weight)
        for key in acc:
            acc[key] += metrics[key]
        seen += 1
    if seen == 0:
        return acc
    return {key: value / seen for key, value in acc.items()}


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_val: float,
    config: dict[str, Any],
    dims: dict[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(step),
            "best_val": float(best_val),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config": config,
            "embedding_dims": dims,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RLT Stage 1 encoder-decoder on frozen SmolVLA embeddings.")
    parser.add_argument("--policy-path", type=Path, default=STAGE1_CFG.policy_path)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=STAGE1_CFG.output_dir)
    parser.add_argument("--device", default=STAGE1_CFG.device)
    parser.add_argument("--batch-size", type=int, default=STAGE1_CFG.batch_size)
    parser.add_argument("--num-workers", type=int, default=STAGE1_CFG.num_workers)
    parser.add_argument("--steps", type=int, default=STAGE1_CFG.steps)
    parser.add_argument("--val-fraction", type=float, default=STAGE1_CFG.val_fraction)
    parser.add_argument("--val-batches", type=int, default=STAGE1_CFG.val_batches)
    parser.add_argument("--log-freq", type=int, default=STAGE1_CFG.log_freq)
    parser.add_argument("--save-freq", type=int, default=STAGE1_CFG.save_freq)
    parser.add_argument("--seed", type=int, default=STAGE1_CFG.seed)
    parser.add_argument("--lr", type=float, default=STAGE1_CFG.lr)
    parser.add_argument("--weight-decay", type=float, default=STAGE1_CFG.weight_decay)
    parser.add_argument("--grad-clip-norm", type=float, default=STAGE1_CFG.grad_clip_norm)
    parser.add_argument("--hidden-dim", type=int, default=STAGE1_CFG.hidden_dim)
    parser.add_argument("--z-dim", type=int, default=STAGE1_CFG.z_dim)
    parser.add_argument("--architecture", choices=("sequence", "pooled"), default=STAGE1_CFG.architecture)
    parser.add_argument("--encoder-layers", type=int, default=STAGE1_CFG.encoder_layers)
    parser.add_argument("--encoder-heads", type=int, default=STAGE1_CFG.encoder_heads)
    parser.add_argument("--dropout", type=float, default=STAGE1_CFG.dropout)
    parser.add_argument("--expert-loss-weight", type=float, default=STAGE1_CFG.expert_loss_weight)
    parser.add_argument("--video-backend", default=STAGE1_CFG.video_backend)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=STAGE1_CFG.amp)
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=STAGE1_CFG.offline)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="Debug limit overriding --steps.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.offline:
        set_offline_env()
    else:
        os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache/huggingface"))
        os.environ.setdefault("HF_DATASETS_CACHE", str(REPO_ROOT / ".cache/huggingface/datasets"))
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    policy_path = normalize_policy_path(args.policy_path)
    repo_id, dataset_root = load_train_dataset_config(policy_path)
    if args.repo_id:
        repo_id = args.repo_id
    if args.dataset_root is not None:
        dataset_root = args.dataset_root.expanduser()
        if not dataset_root.is_absolute():
            dataset_root = (REPO_ROOT / dataset_root).resolve()

    fps = dataset_fps(dataset_root)
    policy_config = load_json(policy_path / "config.json")
    chunk_size = int(policy_config.get("chunk_size", 50))
    delta_timestamps = {"action": [idx / fps for idx in range(chunk_size)]}

    output_root = args.output_root.expanduser()
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()
    output_dir = args.output_dir.expanduser() if args.output_dir is not None else default_output_dir(policy_path, output_root)
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[stage1] policy={policy_path}")
    print(f"[stage1] dataset={dataset_root} repo_id={repo_id} fps={fps:.3f} chunk={chunk_size}")
    print(f"[stage1] output={output_dir}")

    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_root,
        delta_timestamps=delta_timestamps,
        tolerance_s=1e-4,
        video_backend=args.video_backend,
        download_videos=False,
    )
    train_indices, val_indices = split_indices(len(dataset), args.val_fraction, args.seed, args.max_frames)
    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices) if val_indices else Subset(dataset, train_indices[: min(len(train_indices), 128)])
    train_loader = DataLoader(
        train_ds,
        batch_size=max(1, int(args.batch_size)),
        shuffle=True,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=args.device == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, min(int(args.num_workers), 2)),
        pin_memory=args.device == "cuda",
        drop_last=False,
    )

    reader = FrozenSmolVLAEmbeddingReader(policy_path, args.device, amp=args.amp)
    first_batch = next(iter(train_loader))
    first_target = reader.read(first_batch)
    vlm_dim = int(first_target["vlm"].shape[-1])
    expert_dim = int(first_target["expert"].shape[-1])
    max_vlm_tokens = int(first_target["vlm_seq"].shape[1])
    max_expert_tokens = int(first_target["expert_seq"].shape[1])
    dims = {
        "vlm_dim": vlm_dim,
        "expert_dim": expert_dim,
        "z_dim": int(args.z_dim),
        "max_vlm_tokens": max_vlm_tokens,
        "max_expert_tokens": max_expert_tokens,
    }
    print(
        f"[stage1] embedding dims: vlm={vlm_dim} expert={expert_dim} z={args.z_dim} "
        f"tokens=({max_vlm_tokens},{max_expert_tokens}) architecture={args.architecture}"
    )

    model_cfg = {
        "architecture": args.architecture,
        "vlm_dim": vlm_dim,
        "expert_dim": expert_dim,
        "max_vlm_tokens": max_vlm_tokens,
        "max_expert_tokens": max_expert_tokens,
        "hidden_dim": args.hidden_dim,
        "z_dim": args.z_dim,
        "encoder_layers": args.encoder_layers,
        "encoder_heads": args.encoder_heads,
        "dropout": args.dropout,
        "expert_loss_weight": args.expert_loss_weight,
        "policy_path": str(policy_path),
        "dataset_root": str(dataset_root),
        "repo_id": repo_id,
        "fps": fps,
        "chunk_size": chunk_size,
        "delta_timestamps": delta_timestamps,
    }
    model = make_stage1_model(model_cfg).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    (output_dir / "stage1_config.json").write_text(json.dumps(model_cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    train_iter = cycle(train_loader)
    total_steps = int(args.max_steps or args.steps)
    best_val = math.inf
    start = time.time()
    running = {"loss": 0.0, "vlm": 0.0, "expert": 0.0, "z_norm": 0.0}
    running_count = 0
    for step in range(1, total_steps + 1):
        model.train()
        batch = next(train_iter)
        target = reader.read(batch)
        target = {key: value.to(args.device) for key, value in target.items()}
        out = forward_stage1(model, target)
        loss, metrics = mse_pair(out, target, expert_loss_weight=args.expert_loss_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optimizer.step()

        for key in running:
            running[key] += metrics[key]
        running_count += 1

        should_log = step == 1 or step % args.log_freq == 0 or step == total_steps
        should_save = step % args.save_freq == 0 or step == total_steps
        if should_log or should_save:
            val = evaluate(
                model,
                reader,
                val_loader,
                device=args.device,
                batches=max(1, int(args.val_batches)),
                expert_loss_weight=args.expert_loss_weight,
            )
            if val["loss"] < best_val:
                best_val = val["loss"]
                save_checkpoint(
                    output_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    best_val=best_val,
                    config=model_cfg,
                    dims=dims,
                )
            if should_save:
                save_checkpoint(
                    output_dir / "last.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    best_val=best_val,
                    config=model_cfg,
                    dims=dims,
                )
            if should_log:
                elapsed = max(time.time() - start, 1e-6)
                avg = {key: value / max(running_count, 1) for key, value in running.items()}
                print(
                    f"step={step:06d}/{total_steps} "
                    f"train_loss={avg['loss']:.6f} train_vlm={avg['vlm']:.6f} train_exp={avg['expert']:.6f} "
                    f"val_loss={val['loss']:.6f} val_vlm={val['vlm']:.6f} val_exp={val['expert']:.6f} "
                    f"z={avg['z_norm']:.3f} best={best_val:.6f} steps/s={step/elapsed:.3f}",
                    flush=True,
                )
                running = {key: 0.0 for key in running}
                running_count = 0

    summary = {
        "kind": "rlt_stage1_autoencoder",
        "note": "Stage 1 freezes SmolVLA, reads contextual VLM prefix and action-expert embeddings, trains an encoder to produce z_rl, and trains a decoder to reconstruct those embeddings.",
        "best_val": best_val,
        "steps": total_steps,
        "model_config": model_cfg,
        "defaults": stage1_config_jsonable(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] best={output_dir / 'best.pt'} last={output_dir / 'last.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
