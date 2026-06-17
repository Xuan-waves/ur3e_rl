#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rlt_token.config import CFG  # noqa: E402


def normalize_policy_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    else:
        path = path.resolve()
    if (path / "model.safetensors").exists():
        return path
    if (path / "pretrained_model" / "model.safetensors").exists():
        return (path / "pretrained_model").resolve()
    for rel in (
        "checkpoints/last/pretrained_model",
        "checkpoints/050000/pretrained_model",
        "checkpoints/020000/pretrained_model",
    ):
        candidate = path / rel
        if (candidate / "model.safetensors").exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve a pretrained_model directory from: {path}")


def load_train_dataset_config(policy_path: Path) -> tuple[str, Path]:
    train_config_path = policy_path / "train_config.json"
    if not train_config_path.exists():
        raise FileNotFoundError(f"Missing train_config.json in policy path: {policy_path}")
    train_config = json.loads(train_config_path.read_text(encoding="utf-8"))
    dataset_cfg = train_config.get("dataset")
    if not isinstance(dataset_cfg, dict):
        raise ValueError(f"train_config.json does not contain a dataset config: {train_config_path}")
    repo_id = str(dataset_cfg.get("repo_id") or "")
    root_value = dataset_cfg.get("root")
    if not repo_id or not root_value:
        raise ValueError(f"dataset.repo_id/root are required in: {train_config_path}")
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        root = (REPO_ROOT / root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root from train_config does not exist: {root}")
    return repo_id, root


def default_output_dir(policy_path: Path, output_root: Path) -> Path:
    step = policy_path.parent.name if policy_path.parent.name != "checkpoints" else policy_path.name
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = policy_path.parents[2].name if len(policy_path.parents) > 2 else policy_path.parent.name
    return output_root / f"rlt_token_{run_name}_{step}_{stamp}"


def jsonable_config() -> dict[str, Any]:
    out = {}
    for key, value in asdict(CFG).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def scalar_to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def collate_metadata(batch: dict[str, Any]) -> dict[str, np.ndarray]:
    meta = {}
    for key in ("index", "episode_index", "frame_index", "timestamp", "task_index"):
        if key in batch:
            meta[key] = scalar_to_numpy(batch[key])
    return meta


class PrefixTokenExtractor:
    def __init__(self, policy_path: Path, device: str, *, amp: bool) -> None:
        import lerobot.policies.smolvla.processor_smolvla  # noqa: F401
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.utils.constants import (
            OBS_LANGUAGE_ATTENTION_MASK,
            OBS_LANGUAGE_TOKENS,
            POLICY_PREPROCESSOR_DEFAULT_NAME,
        )

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.amp = bool(amp and device == "cuda")
        self.obs_language_tokens = OBS_LANGUAGE_TOKENS
        self.obs_language_attention_mask = OBS_LANGUAGE_ATTENTION_MASK

        self.policy = SmolVLAPolicy.from_pretrained(
            policy_path,
            cli_overrides=[f"--device={device}"],
            local_files_only=True,
        )
        self.policy.eval()
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
        }
        self.make_att_2d_masks = make_att_2d_masks

    @staticmethod
    def masked_mean(embs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(dtype=embs.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (embs * weights).sum(dim=1) / denom

    @torch.inference_mode()
    def extract(
        self,
        batch: dict[str, Any],
        *,
        token_source: str,
        save_prefix_sequence: bool,
    ) -> dict[str, torch.Tensor]:
        autocast = torch.autocast(device_type="cuda") if self.amp else torch.no_grad()
        with autocast:
            processed = self.preprocessor(batch)
            processed = {key: value for key, value in processed.items() if key in self.input_keys}
            processed = self.policy._prepare_batch(processed)
            images, img_masks = self.policy.prepare_images(processed)
            state = self.policy.prepare_state(processed)
            lang_tokens = processed[self.obs_language_tokens]
            lang_masks = processed[self.obs_language_attention_mask]
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.policy.model.embed_prefix(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state=state,
            )
            prefix_embed_pooled = self.masked_mean(prefix_embs.float(), prefix_pad_masks)
            prefix_context = None
            prefix_context_pooled = None
            if token_source in {"context", "both"}:
                prefix_att_2d_masks = self.make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
                prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
                (prefix_context, _), _past_key_values = self.policy.model.vlm_with_expert.forward(
                    attention_mask=prefix_att_2d_masks,
                    position_ids=prefix_position_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs, None],
                    use_cache=self.policy.config.use_cache,
                    fill_kv_cache=True,
                )
                prefix_context_pooled = self.masked_mean(prefix_context.float(), prefix_pad_masks)
            rl_token = prefix_embed_pooled if token_source == "embedding" else prefix_context_pooled
            if rl_token is None:
                raise RuntimeError(f"Failed to extract token for token_source={token_source!r}")

        out = {
            "rl_token": rl_token.detach().cpu(),
            "prefix_embed_pooled": prefix_embed_pooled.detach().cpu(),
            "prefix_pad_count": prefix_pad_masks.sum(dim=1).detach().cpu().to(torch.int32),
        }
        if prefix_context_pooled is not None:
            out["prefix_context_pooled"] = prefix_context_pooled.detach().cpu()
        if save_prefix_sequence:
            out["prefix_sequence"] = prefix_embs.detach().cpu().to(torch.float16)
            out["prefix_pad_mask"] = prefix_pad_masks.detach().cpu()
            out["prefix_attention_mask"] = prefix_att_masks.detach().cpu()
            if prefix_context is not None:
                out["prefix_context_sequence"] = prefix_context.detach().cpu().to(torch.float16)
        return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract frozen SmolVLA prefix embeddings as the first-stage RLT token source. "
            "The saved rl_token tensor is the VLA-derived token used by later RL-token encoder/RL heads."
        )
    )
    parser.add_argument("--policy-path", type=Path, default=CFG.policy_path)
    parser.add_argument("--dataset-root", type=Path, default=None, help="Override dataset root from train_config.json.")
    parser.add_argument("--repo-id", default=None, help="Override dataset repo_id from train_config.json.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=CFG.output_dir)
    parser.add_argument("--device", default=CFG.device)
    parser.add_argument("--batch-size", type=int, default=CFG.batch_size)
    parser.add_argument("--num-workers", type=int, default=CFG.num_workers)
    parser.add_argument("--episodes", type=int, nargs="*", default=None, help="Optional episode ids to extract.")
    parser.add_argument("--video-backend", default=CFG.video_backend)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=CFG.amp)
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=CFG.offline)
    parser.add_argument(
        "--token-source",
        choices=("context", "embedding", "both"),
        default=CFG.token_source,
        help=(
            "context runs prefix embeddings through the frozen SmolVLA transformer and pools the contextual output. "
            "embedding pools the input prefix embeddings before transformer. both saves both and uses context as rl_token."
        ),
    )
    parser.add_argument(
        "--save-prefix-sequence",
        action=argparse.BooleanOptionalAction,
        default=CFG.save_prefix_sequence,
        help="Also save the full prefix token sequence. This is large; pooled tokens are enough for the first pass.",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Debug limit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache/huggingface"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(REPO_ROOT / ".cache/huggingface/datasets"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    policy_path = normalize_policy_path(args.policy_path)
    repo_id, dataset_root = load_train_dataset_config(policy_path)
    if args.repo_id:
        repo_id = args.repo_id
    if args.dataset_root is not None:
        dataset_root = args.dataset_root.expanduser()
        if not dataset_root.is_absolute():
            dataset_root = (REPO_ROOT / dataset_root).resolve()

    output_root = args.output_root.expanduser()
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()
    output_dir = args.output_dir.expanduser() if args.output_dir is not None else default_output_dir(policy_path, output_root)
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[rlt-token] policy={policy_path}")
    print(f"[rlt-token] dataset={dataset_root} repo_id={repo_id}")
    print(f"[rlt-token] output={output_dir}")

    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_root,
        episodes=args.episodes,
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
        pin_memory=args.device == "cuda" and torch.cuda.is_available(),
    )
    extractor = PrefixTokenExtractor(policy_path, args.device, amp=args.amp)

    chunks: dict[str, list[torch.Tensor]] = {
        "rl_token": [],
        "prefix_embed_pooled": [],
        "prefix_pad_count": [],
    }
    if args.token_source in {"context", "both"}:
        chunks["prefix_context_pooled"] = []
    if args.save_prefix_sequence:
        chunks.update({"prefix_sequence": [], "prefix_pad_mask": [], "prefix_attention_mask": []})
        if args.token_source in {"context", "both"}:
            chunks["prefix_context_sequence"] = []
    meta_chunks: dict[str, list[np.ndarray]] = {}

    total = len(dataset)
    seen = 0
    for batch_idx, batch in enumerate(loader, start=1):
        tokens = extractor.extract(
            batch,
            token_source=args.token_source,
            save_prefix_sequence=args.save_prefix_sequence,
        )
        for key, value in tokens.items():
            chunks[key].append(value)
        for key, value in collate_metadata(batch).items():
            meta_chunks.setdefault(key, []).append(value)
        seen += int(tokens["rl_token"].shape[0])
        if batch_idx == 1 or batch_idx % 10 == 0 or seen >= total:
            shape = tuple(tokens["rl_token"].shape)
            print(f"\r[rlt-token] {seen}/{total} frames rl_token{shape}", end="", flush=True)
    print()

    tensors = {key: torch.cat(values, dim=0) for key, values in chunks.items()}
    torch.save(tensors, output_dir / "tokens.pt")
    metadata = {key: np.concatenate(values, axis=0) for key, values in meta_chunks.items()}
    np.savez_compressed(output_dir / "metadata.npz", **metadata)

    summary = {
        "kind": "smolvla_prefix_token",
        "note": "rl_token is pooled from the frozen SmolVLA prefix representation. token_source=context means the prefix was passed through the VLA transformer before pooling; train a token encoder before treating it as a compressed 256D RL token.",
        "policy_path": str(policy_path),
        "dataset_root": str(dataset_root),
        "repo_id": repo_id,
        "frames": int(seen),
        "episodes": args.episodes,
        "device": extractor.device,
        "amp": bool(extractor.amp),
        "token_source": args.token_source,
        "rl_token_shape": list(tensors["rl_token"].shape),
        "save_prefix_sequence": bool(args.save_prefix_sequence),
        "config": jsonable_config(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] tokens={output_dir / 'tokens.pt'} metadata={output_dir / 'metadata.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
