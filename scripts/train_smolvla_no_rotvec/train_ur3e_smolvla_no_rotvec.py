#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_smolvla.train_ur3e_smolvla import (  # noqa: E402
    ensure_training_dependencies,
    make_train_command,
    summarize_dataset,
)
from scripts.train_smolvla_no_rotvec.config import CFG  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SmolVLA on the UR3e no-rotvec impedance dataset.")
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
