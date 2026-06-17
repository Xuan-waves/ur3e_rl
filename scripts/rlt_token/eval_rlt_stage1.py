#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

warnings.filterwarnings(
    "ignore",
    message="The video decoding and encoding capabilities of torchvision are deprecated.*",
    category=UserWarning,
)

from scripts.rlt_token.extract_rl_token import load_train_dataset_config, normalize_policy_path  # noqa: E402
from scripts.rlt_token.train_rlt_stage1 import (  # noqa: E402
    FrozenSmolVLAEmbeddingReader,
    RLTStage1AutoEncoder,
    dataset_fps,
    forward_stage1,
    load_json,
    make_stage1_model,
    set_offline_env,
)


def latest_stage1_checkpoint(output_root: Path) -> Path:
    candidates = sorted(output_root.expanduser().glob("*/best.pt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No Stage1 best.pt found under {output_root}")
    return candidates[-1].resolve()


def scalar_array(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def per_item_mse(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    loss = F.mse_loss(pred, target, reduction="none")
    if loss.ndim <= 2:
        return loss.mean(dim=-1).detach().cpu().numpy()
    dims = tuple(range(1, loss.ndim))
    return loss.mean(dim=dims).detach().cpu().numpy()


def per_item_masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    loss = (pred - target).pow(2).mean(dim=-1)
    weights = mask.to(dtype=loss.dtype)
    return ((loss * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)).detach().cpu().numpy()


def pca2(x: np.ndarray) -> np.ndarray:
    if x.shape[0] < 2:
        return np.zeros((x.shape[0], 2), dtype=np.float32)
    centered = x - x.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    comps = vh[:2].T
    out = centered @ comps
    if out.shape[1] == 1:
        out = np.pad(out, ((0, 0), (0, 1)))
    return out.astype(np.float32)


def save_plots(path: Path, arrays: dict[str, np.ndarray], episode: int) -> None:
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib unavailable, skip plots: {exc}")
        return

    frame = arrays["frame_index"]
    z_norm = arrays["z_norm"]
    dz_norm = arrays["dz_norm"]
    vlm_loss = arrays["vlm_loss"]
    expert_loss = arrays["expert_loss"]
    pca = arrays["z_pca2"]

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), constrained_layout=True)
    axes[0].plot(frame, z_norm)
    axes[0].set_title(f"episode {episode}: z norm")
    axes[0].set_xlabel("frame")
    axes[0].set_ylabel("||z||")

    axes[1].plot(frame, dz_norm)
    axes[1].set_title("temporal token step")
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("||z_t - z_{t-1}||")

    axes[2].plot(frame, vlm_loss, label="vlm")
    axes[2].plot(frame, expert_loss, label="expert")
    axes[2].set_title("reconstruction loss per frame")
    axes[2].set_xlabel("frame")
    axes[2].legend()

    scatter = axes[3].scatter(pca[:, 0], pca[:, 1], c=frame, s=8, cmap="viridis")
    axes[3].set_title("z_rl PCA trajectory")
    axes[3].set_xlabel("PC1")
    axes[3].set_ylabel("PC2")
    fig.colorbar(scatter, ax=axes[3], label="frame")

    fig.savefig(path, dpi=140)
    plt.close(fig)


def load_stage1_model(checkpoint_path: Path, device: str) -> tuple[RLTStage1AutoEncoder, dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt["model_config"]
    model = make_stage1_model(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, cfg


def resolve_eval_policy_path(cli_path: Path | None, stage1_cfg: dict[str, Any]) -> Path:
    if cli_path is not None:
        return normalize_policy_path(cli_path)
    saved_path = Path(stage1_cfg["policy_path"])
    try:
        return normalize_policy_path(saved_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "The Stage1 checkpoint records a SmolVLA policy path that is not available on this machine: "
            f"{saved_path}\n"
            "Pass the current SmolVLA checkpoint explicitly, for example:\n"
            "  --policy-path outputs/rlt_vla/ur3e_smolvla_0610/checkpoints/050000/pretrained_model"
        ) from exc


def summarize_episode(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    dz = arrays["dz_norm"]
    return {
        "frames": int(arrays["z"].shape[0]),
        "z_norm_mean": float(arrays["z_norm"].mean()),
        "z_norm_std": float(arrays["z_norm"].std()),
        "dz_mean": float(dz[1:].mean()) if dz.size > 1 else 0.0,
        "dz_p95": float(np.percentile(dz[1:], 95)) if dz.size > 1 else 0.0,
        "vlm_loss_mean": float(arrays["vlm_loss"].mean()),
        "expert_loss_mean": float(arrays["expert_loss"].mean()),
        "loss_mean": float(arrays["loss"].mean()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline diagnostics for trained RLT Stage1 token.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Stage1 best.pt/last.pt.")
    parser.add_argument("--stage1-root", type=Path, default=REPO_ROOT / "outputs/rlt_stage1")
    parser.add_argument("--policy-path", type=Path, default=None, help="Override policy path from Stage1 checkpoint.")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--max-episodes", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.offline:
        set_offline_env()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else latest_stage1_checkpoint(args.stage1_root)
    model, stage1_cfg = load_stage1_model(checkpoint, args.device)

    policy_path = resolve_eval_policy_path(args.policy_path, stage1_cfg)
    repo_id, dataset_root = load_train_dataset_config(policy_path)
    if args.repo_id:
        repo_id = args.repo_id
    if args.dataset_root is not None:
        dataset_root = args.dataset_root.expanduser()
        if not dataset_root.is_absolute():
            dataset_root = (REPO_ROOT / dataset_root).resolve()

    fps = dataset_fps(dataset_root)
    policy_config = load_json(policy_path / "config.json")
    chunk_size = int(policy_config.get("chunk_size", stage1_cfg.get("chunk_size", 50)))
    delta_timestamps = {"action": [idx / fps for idx in range(chunk_size)]}

    if args.episodes is None:
        info = load_json(dataset_root / "meta" / "info.json")
        total_eps = int(info.get("total_episodes", 1))
        episodes = list(range(min(args.max_episodes, total_eps)))
    else:
        episodes = args.episodes

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (REPO_ROOT / "outputs/rlt_stage1_eval" / f"{checkpoint.parent.name}_{stamp}")
    output_dir = output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval-stage1] checkpoint={checkpoint}")
    print(f"[eval-stage1] policy={policy_path}")
    print(f"[eval-stage1] dataset={dataset_root} episodes={episodes}")
    print(f"[eval-stage1] output={output_dir}")

    reader = FrozenSmolVLAEmbeddingReader(policy_path, args.device, amp=False)
    all_summary: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "policy_path": str(policy_path),
        "dataset_root": str(dataset_root),
        "episodes": {},
    }

    for ep in episodes:
        dataset = LeRobotDataset(
            repo_id=repo_id,
            root=dataset_root,
            episodes=[int(ep)],
            delta_timestamps=delta_timestamps,
            tolerance_s=1e-4,
            video_backend=args.video_backend,
            download_videos=False,
        )
        if args.max_frames is not None:
            indices = list(range(min(int(args.max_frames), len(dataset))))
            dataset = torch.utils.data.Subset(dataset, indices)

        loader = DataLoader(
            dataset,
            batch_size=max(1, int(args.batch_size)),
            shuffle=False,
            num_workers=max(0, int(args.num_workers)),
            pin_memory=args.device == "cuda",
        )
        chunks: dict[str, list[np.ndarray]] = {
            "z": [],
            "vlm_loss": [],
            "expert_loss": [],
            "loss": [],
            "frame_index": [],
            "timestamp": [],
        }
        with torch.no_grad():
            for batch in loader:
                target = reader.read(batch)
                target = {key: value.to(args.device) for key, value in target.items()}
                out = forward_stage1(model, target)
                if "vlm_seq_recon" in out:
                    vlm_loss = per_item_masked_mse(out["vlm_seq_recon"], target["vlm_seq"], target["vlm_mask"])
                    expert_loss = per_item_masked_mse(
                        out["expert_seq_recon"],
                        target["expert_seq"],
                        target["expert_mask"],
                    )
                else:
                    vlm_loss = per_item_mse(out["vlm_recon"], target["vlm"])
                    expert_loss = per_item_mse(out["expert_recon"], target["expert"])
                chunks["z"].append(out["z_rl"].detach().cpu().numpy())
                chunks["vlm_loss"].append(vlm_loss)
                chunks["expert_loss"].append(expert_loss)
                chunks["loss"].append(vlm_loss + expert_loss)
                chunks["frame_index"].append(scalar_array(batch["frame_index"]).reshape(-1))
                chunks["timestamp"].append(scalar_array(batch["timestamp"]).reshape(-1))

        arrays = {key: np.concatenate(values, axis=0) for key, values in chunks.items()}
        z = arrays["z"]
        dz = np.zeros(z.shape[0], dtype=np.float32)
        if z.shape[0] > 1:
            dz[1:] = np.linalg.norm(np.diff(z, axis=0), axis=1)
        arrays["z_norm"] = np.linalg.norm(z, axis=1).astype(np.float32)
        arrays["dz_norm"] = dz
        arrays["z_pca2"] = pca2(z)

        ep_dir = output_dir / f"episode_{int(ep):06d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ep_dir / "stage1_eval.npz", **arrays)
        save_plots(ep_dir / "stage1_eval.png", arrays, int(ep))
        summary = summarize_episode(arrays)
        all_summary["episodes"][str(ep)] = summary
        print(
            f"[episode {ep}] frames={summary['frames']} loss={summary['loss_mean']:.6f} "
            f"vlm={summary['vlm_loss_mean']:.6f} expert={summary['expert_loss_mean']:.6f} "
            f"z={summary['z_norm_mean']:.3f}+/-{summary['z_norm_std']:.3f} "
            f"dz_mean={summary['dz_mean']:.3f} dz_p95={summary['dz_p95']:.3f}"
        )

    (output_dir / "summary.json").write_text(json.dumps(all_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] summary={output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
