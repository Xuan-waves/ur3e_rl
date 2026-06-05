#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_smolvla.config import CFG


def str_bool(value: bool) -> str:
    return "true" if value else "false"


def check_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_dataset(root: Path) -> None:
    info_path = root / "meta" / "info.json"
    stats_path = root / "meta" / "stats.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")

    info = load_json(info_path)
    print(
        "[dataset] "
        f"root={root} episodes={info.get('total_episodes')} frames={info.get('total_frames')} "
        f"fps={info.get('fps')} robot={info.get('robot_type')}"
    )
    features = info.get("features", {})
    for key in ("observation.images.cam_front", "observation.images.cam_wrist", "observation.state", "action"):
        feature = features.get(key, {})
        print(f"[dataset] {key}: dtype={feature.get('dtype')} shape={feature.get('shape')}")

    if stats_path.exists():
        stats = load_json(stats_path)
        for key, label in (("observation.state", "state"), ("action", "action")):
            values = stats.get(key, {})
            mins = values.get("min") or []
            maxs = values.get("max") or []
            if len(mins) >= 2 and len(maxs) >= 2:
                rl_idx = min(len(mins), len(maxs)) - 1
                rl_min = float(mins[rl_idx])
                rl_max = float(maxs[rl_idx])
                print(f"[dataset] {label}.RL_mark[{rl_idx}] min={rl_min:.3f} max={rl_max:.3f}")
                if abs(rl_max - rl_min) < 1e-6:
                    print(f"[warn] {label}.RL_mark has no variation in this dataset.")


def ensure_training_dependencies(*, strict: bool) -> None:
    required = ["torch", "lerobot", "accelerate", "av"]
    missing = [name for name in required if not check_module(name)]
    if missing:
        message = f"Missing required training modules: {', '.join(missing)}"
        if strict:
            raise RuntimeError(message)
        print(f"[warn] {message}")

    missing_smolvla = [name for name in ["transformers"] if not check_module(name)]
    if missing_smolvla:
        message = (
            "SmolVLA dependency is missing: transformers. "
            "Install LeRobot SmolVLA extras in the ur3e_rlt env, for example: "
            "pip install 'lerobot[smolvla]'"
        )
        if strict:
            raise RuntimeError(message)
        print(f"[warn] {message}")


def make_train_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    lerobot_train = shutil.which("lerobot-train")
    if lerobot_train is None:
        candidate = Path(sys.executable).with_name("lerobot-train")
        lerobot_train = str(candidate if candidate.exists() else "lerobot-train")

    dataset_root = args.dataset.resolve()
    repo_id = args.repo_id or f"local/{dataset_root.name}"

    cmd = [
        lerobot_train,
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={dataset_root}",
        "--dataset.video_backend=pyav",
        f"--dataset.use_imagenet_stats={str_bool(args.use_imagenet_stats)}",
        "--policy.type=smolvla",
        f"--policy.push_to_hub={str_bool(False)}",
        f"--policy.device={args.device}",
        f"--policy.use_amp={str_bool(args.amp)}",
        f"--policy.vlm_model_name={args.vlm_model_name}",
        f"--policy.load_vlm_weights={str_bool(args.load_vlm_weights)}",
        f"--policy.freeze_vision_encoder={str_bool(args.freeze_vision_encoder)}",
        f"--policy.train_expert_only={str_bool(args.train_expert_only)}",
        f"--policy.train_state_proj={str_bool(args.train_state_proj)}",
        f"--policy.n_obs_steps={args.n_obs_steps}",
        f"--policy.chunk_size={args.chunk_size}",
        f"--policy.n_action_steps={args.n_action_steps}",
        f"--policy.max_state_dim={args.max_state_dim}",
        f"--policy.max_action_dim={args.max_action_dim}",
        f"--policy.num_vlm_layers={args.num_vlm_layers}",
        f"--policy.num_expert_layers={args.num_expert_layers}",
        f"--policy.expert_width_multiplier={args.expert_width_multiplier}",
        f"--policy.resize_imgs_with_padding=[{args.image_size},{args.image_size}]",
        f"--batch_size={args.batch_size}",
        f"--steps={args.steps}",
        f"--num_workers={args.num_workers}",
        f"--log_freq={args.log_freq}",
        f"--save_freq={args.save_freq}",
        "--eval_freq=0",
        f"--seed={args.seed}",
        f"--output_dir={output_dir}",
        f"--job_name={args.job_name}",
        "--wandb.enable=false",
        f"--tolerance_s={args.tolerance_s}",
    ]
    if args.extra:
        cmd.extend(args.extra)
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SmolVLA on the UR3e VR impedance LeRobot dataset.")
    parser.add_argument("--dataset", type=Path, default=CFG.dataset)
    parser.add_argument("--repo-id", default=CFG.repo_id, help="LeRobot repo_id label for local loading.")
    parser.add_argument("--output-root", type=Path, default=CFG.output_root)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--job-name", default=CFG.job_name)
    parser.add_argument("--steps", type=int, default=CFG.steps)
    parser.add_argument("--batch-size", type=int, default=CFG.batch_size)
    parser.add_argument("--num-workers", type=int, default=CFG.num_workers)
    parser.add_argument("--log-freq", type=int, default=CFG.log_freq)
    parser.add_argument("--save-freq", type=int, default=CFG.save_freq)
    parser.add_argument("--seed", type=int, default=CFG.seed)
    parser.add_argument("--device", default=CFG.device)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=CFG.amp)
    parser.add_argument("--use-imagenet-stats", action=argparse.BooleanOptionalAction, default=CFG.use_imagenet_stats)
    parser.add_argument("--vlm-model-name", default=CFG.vlm_model_name)
    parser.add_argument("--load-vlm-weights", action=argparse.BooleanOptionalAction, default=CFG.load_vlm_weights)
    parser.add_argument("--freeze-vision-encoder", action=argparse.BooleanOptionalAction, default=CFG.freeze_vision_encoder)
    parser.add_argument("--train-expert-only", action=argparse.BooleanOptionalAction, default=CFG.train_expert_only)
    parser.add_argument("--train-state-proj", action=argparse.BooleanOptionalAction, default=CFG.train_state_proj)
    parser.add_argument("--n-obs-steps", type=int, default=CFG.n_obs_steps)
    parser.add_argument("--chunk-size", type=int, default=CFG.chunk_size)
    parser.add_argument("--n-action-steps", type=int, default=CFG.n_action_steps)
    parser.add_argument("--max-state-dim", type=int, default=CFG.max_state_dim)
    parser.add_argument("--max-action-dim", type=int, default=CFG.max_action_dim)
    parser.add_argument("--image-size", type=int, default=CFG.image_size)
    parser.add_argument("--num-vlm-layers", type=int, default=CFG.num_vlm_layers)
    parser.add_argument("--num-expert-layers", type=int, default=CFG.num_expert_layers)
    parser.add_argument("--expert-width-multiplier", type=float, default=CFG.expert_width_multiplier)
    parser.add_argument("--tolerance-s", type=float, default=CFG.tolerance_s)
    parser.add_argument("--dry-run", action="store_true", help="Print command and validate dataset without training.")
    parser.add_argument(
        "--skip-dependency-check",
        action="store_true",
        help="Skip SmolVLA dependency checks before invoking lerobot-train.",
    )
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args forwarded to lerobot-train after --.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.extra and args.extra[0] == "--":
        args.extra = args.extra[1:]
    dataset = args.dataset.expanduser()
    if not dataset.is_absolute():
        dataset = (REPO_ROOT / dataset).resolve()
    args.dataset = dataset

    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = args.output_root / f"{args.job_name}_{stamp}"
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()

    os.environ.setdefault("HF_HOME", str(CFG.hf_home))
    os.environ.setdefault("HF_DATASETS_CACHE", str(CFG.hf_home / "datasets"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    summarize_dataset(dataset)
    ensure_training_dependencies(strict=not (args.dry_run or args.skip_dependency_check))

    cmd = make_train_command(args, output_dir)
    print("[train] command:")
    print(" ".join(cmd))
    if args.dry_run:
        return 0
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.run(cmd, cwd=REPO_ROOT, env=os.environ.copy(), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
