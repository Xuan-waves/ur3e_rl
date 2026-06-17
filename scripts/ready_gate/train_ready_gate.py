#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rlt_gate.train_rlt_gate import build_model


@dataclass(frozen=True)
class ReadyGateTrainConfig:
    dataset: Path = REPO_ROOT / "datasets/ready_gate/ready_gate_20260617_145611"
    output_dir: Path = REPO_ROOT / "outputs/ready_gate"
    camera: str = "both"
    model: str = "tiny"
    image_size: int = 128
    batch_size: int = 128
    num_workers: int = 0
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    val_ratio: float = 0.2
    seed: int = 7
    label_smoothing: float = 0.02
    sample_stride: int = 1
    split_mode: str = "stratified_frame"
    cache_images: str = "ram"
    device: str = "auto"
    amp: bool = True
    threshold: float = 0.5


CFG = ReadyGateTrainConfig()


class ReadyGateDataset(Dataset):
    def __init__(
        self,
        root: Path,
        labels: pd.DataFrame,
        *,
        camera: str,
        image_size: int,
        augment: bool,
        cache_images: str,
        progress: bool,
        name: str,
    ) -> None:
        self.root = root
        self.labels = labels.reset_index(drop=True)
        self.camera = camera
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self._ram_images: list[np.ndarray] | None = None
        if cache_images == "ram":
            self._ram_images = self._build_ram_cache(progress=progress, name=name)
        elif cache_images != "none":
            raise ValueError(f"Unsupported cache mode: {cache_images!r}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.labels.iloc[idx]
        if self._ram_images is not None:
            image = self._ram_images[idx]
        else:
            image = self._read_row(row)
        return self._preprocess(image), torch.tensor(float(row.ready_gate), dtype=torch.float32)

    def _build_ram_cache(self, *, progress: bool, name: str) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        iterator = range(len(self.labels))
        if progress and tqdm is not None:
            iterator = tqdm(iterator, desc=f"cache {name}", leave=False, dynamic_ncols=True)
        for idx in iterator:
            out.append(self._read_row(self.labels.iloc[idx]))
        return out

    def _read_row(self, row: pd.Series) -> np.ndarray:
        if self.camera == "both":
            front = self._read_image(self.root / str(row.front_path))
            wrist = self._read_image(self.root / str(row.wrist_path))
            return np.concatenate([front, wrist], axis=2)
        if self.camera == "wrist":
            return self._read_image(self.root / str(row.wrist_path))
        return self._read_image(self.root / str(row.front_path))

    @staticmethod
    def _read_image(path: Path) -> np.ndarray:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Could not read image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
            if image.shape[2] > 4:
                parts = [
                    cv2.resize(image[:, :, start : start + 3], (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
                    for start in range(0, image.shape[2], 3)
                ]
                image = np.concatenate(parts, axis=2)
            else:
                image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        if self.augment:
            if random.random() < 0.5:
                factor = random.uniform(0.85, 1.15)
                image = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
            if random.random() < 0.25:
                noise = np.random.normal(0.0, 3.0, image.shape).astype(np.float32)
                image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        x = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().div(255.0)
        return (x - 0.5) / 0.5


def load_labels(root: Path, sample_stride: int) -> pd.DataFrame:
    rows: list[dict] = []
    for ep_dir in sorted(root.glob("episode_*")):
        labels_path = ep_dir / "labels.jsonl"
        if not labels_path.exists():
            continue
        for line in labels_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row["frame_index"]) % max(1, int(sample_stride)) != 0:
                continue
            front_path = root / row["front_path"]
            wrist_path = root / row["wrist_path"]
            if front_path.exists() and wrist_path.exists():
                rows.append(row)
    if not rows:
        raise RuntimeError(f"No ready-gate labels found in {root}")
    return pd.DataFrame(rows).reset_index(drop=True)


def split_labels(labels: pd.DataFrame, val_ratio: float, seed: int, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    if mode == "episode":
        episodes = np.array(sorted(labels.episode_index.unique()))
        rng.shuffle(episodes)
        val_count = max(1, int(round(len(episodes) * val_ratio)))
        val_eps = set(int(ep) for ep in episodes[:val_count])
        val = labels[labels.episode_index.isin(val_eps)]
        train = labels[~labels.episode_index.isin(val_eps)]
        return train.reset_index(drop=True), val.reset_index(drop=True)
    if mode != "stratified_frame":
        raise ValueError(f"Unsupported split mode: {mode!r}")
    train_parts = []
    val_parts = []
    for label, group in labels.groupby("ready_gate"):
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        val_count = max(1, int(round(len(idx) * val_ratio)))
        val_parts.append(labels.loc[idx[:val_count]])
        train_parts.append(labels.loc[idx[val_count:]])
    train = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val = pd.concat(val_parts).sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)
    return train, val


def binary_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    pred = probs >= threshold
    target = labels >= 0.5
    tp = torch.logical_and(pred, target).sum().item()
    tn = torch.logical_and(~pred, ~target).sum().item()
    fp = torch.logical_and(pred, ~target).sum().item()
    fn = torch.logical_and(~pred, target).sum().item()
    total = max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
    return {
        "acc": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    *,
    pos_weight: torch.Tensor,
    label_smoothing: float,
    threshold: float,
    amp: bool,
    progress: bool,
    desc: str,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    seen = 0
    all_logits = []
    all_labels = []
    iterator = loader
    pbar = None
    if progress and tqdm is not None:
        pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
        iterator = pbar
    for images, labels in iterator:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        smooth = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing
        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits = model(images).view(-1)
                loss = F.binary_cross_entropy_with_logits(logits, smooth, pos_weight=pos_weight)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        total_loss += float(loss.item()) * int(labels.numel())
        seen += int(labels.numel())
        all_logits.append(logits.detach().float().cpu())
        all_labels.append(labels.detach().float().cpu())
        if pbar is not None:
            pbar.set_postfix(loss=f"{total_loss / max(seen, 1):.4f}")
    logits_cpu = torch.cat(all_logits)
    labels_cpu = torch.cat(all_labels)
    metrics = binary_metrics(logits_cpu, labels_cpu, threshold)
    metrics["loss"] = total_loss / max(seen, 1)
    return metrics


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small image ready-gate classifier.")
    parser.add_argument("--dataset", type=Path, default=CFG.dataset)
    parser.add_argument("--output-dir", type=Path, default=CFG.output_dir)
    parser.add_argument("--camera", choices=("front", "wrist", "both"), default=CFG.camera)
    parser.add_argument("--model", choices=("tiny", "resnet18"), default=CFG.model)
    parser.add_argument("--image-size", type=int, default=CFG.image_size)
    parser.add_argument("--batch-size", type=int, default=CFG.batch_size)
    parser.add_argument("--num-workers", type=int, default=CFG.num_workers)
    parser.add_argument("--epochs", type=int, default=CFG.epochs)
    parser.add_argument("--lr", type=float, default=CFG.lr)
    parser.add_argument("--weight-decay", type=float, default=CFG.weight_decay)
    parser.add_argument("--val-ratio", type=float, default=CFG.val_ratio)
    parser.add_argument("--seed", type=int, default=CFG.seed)
    parser.add_argument("--label-smoothing", type=float, default=CFG.label_smoothing)
    parser.add_argument("--sample-stride", type=int, default=CFG.sample_stride)
    parser.add_argument("--split-mode", choices=("stratified_frame", "episode"), default=CFG.split_mode)
    parser.add_argument("--cache-images", choices=("ram", "none"), default=CFG.cache_images)
    parser.add_argument("--device", default=CFG.device)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--threshold", type=float, default=CFG.threshold)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = ReadyGateTrainConfig(
        dataset=args.dataset,
        output_dir=args.output_dir,
        camera=args.camera,
        model=args.model,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_ratio=args.val_ratio,
        seed=args.seed,
        label_smoothing=args.label_smoothing,
        sample_stride=args.sample_stride,
        split_mode=args.split_mode,
        cache_images=args.cache_images,
        device=args.device,
        amp=not bool(args.no_amp),
        threshold=args.threshold,
    )
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = choose_device(cfg.device)
    root = cfg.dataset.expanduser().resolve()
    labels = load_labels(root, cfg.sample_stride)
    train_labels, val_labels = split_labels(labels, cfg.val_ratio, cfg.seed, cfg.split_mode)
    channels = 6 if cfg.camera == "both" else 3
    model = build_model(cfg.model, channels).to(device)
    positives = float(train_labels.ready_gate.sum())
    negatives = float(len(train_labels) - positives)
    pos_weight = torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = (cfg.output_dir / f"ready_gate_{stamp}").expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    config_payload = asdict(cfg)
    config_payload["dataset"] = str(root)
    config_payload["output_dir"] = str(cfg.output_dir)
    config_payload["input_channels"] = channels
    (out / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    print(f"[data] root={root} rows={len(labels)} train={len(train_labels)} val={len(val_labels)} split={cfg.split_mode}")
    print(
        f"[data] labels total_pos={int(labels.ready_gate.sum())} total_neg={len(labels)-int(labels.ready_gate.sum())} "
        f"train_pos={int(positives)} train_neg={int(negatives)} pos_weight={float(pos_weight.item()):.3f}"
    )
    print(f"[model] name={cfg.model} camera={cfg.camera} channels={channels} params={sum(p.numel() for p in model.parameters())} device={device}")
    print(f"[output] {out}")

    train_ds = ReadyGateDataset(root, train_labels, camera=cfg.camera, image_size=cfg.image_size, augment=True, cache_images=cfg.cache_images, progress=not args.no_progress, name="train")
    val_ds = ReadyGateDataset(root, val_labels, camera=cfg.camera, image_size=cfg.image_size, augment=False, cache_images=cfg.cache_images, progress=not args.no_progress, name="val")
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=device.type == "cuda")

    best_f1 = -math.inf
    best_path = out / "best.pt"
    last_path = out / "last.pt"
    history = []
    for epoch in range(1, cfg.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            pos_weight=pos_weight,
            label_smoothing=cfg.label_smoothing,
            threshold=cfg.threshold,
            amp=cfg.amp,
            progress=not args.no_progress,
            desc=f"train {epoch:03d}/{cfg.epochs:03d}",
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                None,
                None,
                device,
                pos_weight=pos_weight,
                label_smoothing=0.0,
                threshold=cfg.threshold,
                amp=False,
                progress=not args.no_progress,
                desc=f"val   {epoch:03d}/{cfg.epochs:03d}",
            )
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} train_f1={train_metrics['f1']:.3f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.3f} "
            f"val_p={val_metrics['precision']:.3f} val_r={val_metrics['recall']:.3f} val_f1={val_metrics['f1']:.3f}"
        )
        ckpt = {
            "model": model.state_dict(),
            "config": config_payload,
            "epoch": epoch,
            "metrics": row,
        }
        torch.save(ckpt, last_path)
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(ckpt, best_path)
    pd.DataFrame(history).to_csv(out / "metrics.csv", index=False)
    train_labels.to_parquet(out / "train_labels.parquet")
    val_labels.to_parquet(out / "val_labels.parquet")
    print(f"[done] best_f1={best_f1:.3f} best={best_path} last={last_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
