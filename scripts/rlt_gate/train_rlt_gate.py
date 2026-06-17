from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
import sys

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rlt_gate.config import CFG, RltGateTrainConfig


VIDEO_KEYS = {
    "front": "observation.images.cam_front",
    "wrist": "observation.images.cam_wrist",
}


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyRltGateNet(nn.Module):
    """Small image classifier for frame-level RLT gate prediction."""

    def __init__(self, input_channels: int = 3, width: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, width, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
            DepthwiseSeparableBlock(width, width * 2, stride=2),
            DepthwiseSeparableBlock(width * 2, width * 2, stride=1),
            DepthwiseSeparableBlock(width * 2, width * 4, stride=2),
            DepthwiseSeparableBlock(width * 4, width * 4, stride=1),
            DepthwiseSeparableBlock(width * 4, width * 6, stride=2),
            DepthwiseSeparableBlock(width * 6, width * 6, stride=1),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(width * 6, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(-1)


def build_model(model_name: str, input_channels: int) -> nn.Module:
    if model_name == "tiny":
        return TinyRltGateNet(input_channels=input_channels)
    if model_name == "resnet18":
        try:
            from torchvision.models import resnet18
        except ImportError as exc:
            raise RuntimeError("torchvision is required for --model resnet18") from exc
        model = resnet18(weights=None)
        if input_channels != 3:
            old_conv = model.conv1
            model.conv1 = nn.Conv2d(
                input_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model
    raise ValueError(f"Unsupported model: {model_name!r}")


class RltGateDataset(Dataset):
    def __init__(
        self,
        root: Path,
        labels: pd.DataFrame,
        camera: str,
        image_size: int,
        augment: bool,
        cache_images: str,
        progress: bool,
        name: str,
    ):
        self.root = root
        self.labels = labels.reset_index(drop=True)
        self.camera = camera
        self.image_size = int(image_size)
        self.augment = augment
        self._caps: dict[tuple[str, int], cv2.VideoCapture] = {}
        self._ram_images: list[np.ndarray] | None = None
        if cache_images == "ram":
            self._ram_images = self._build_ram_cache(progress=progress, name=name)
        elif cache_images != "none":
            raise ValueError(f"Unsupported cache mode: {cache_images!r}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.labels.iloc[idx]
        label = float(row.rlt_phase)
        if self._ram_images is not None:
            image = self._ram_images[idx]
        else:
            ep = int(row.episode_index)
            frame_idx = int(row.frame_index)
            if self.camera == "both":
                image = np.concatenate(
                    [self._read_image("front", ep, frame_idx), self._read_image("wrist", ep, frame_idx)],
                    axis=2,
                )
            else:
                image = self._read_image(self.camera, ep, frame_idx)
        image = self._preprocess(image)
        return image, torch.tensor(label, dtype=torch.float32)

    def _build_ram_cache(self, progress: bool, name: str) -> list[np.ndarray]:
        required_cameras = ("front", "wrist") if self.camera == "both" else (self.camera,)
        camera_images = [self._decode_camera_to_ram(camera, progress=progress, name=name) for camera in required_cameras]
        if len(camera_images) == 1:
            return camera_images[0]
        return [np.concatenate([images[i] for images in camera_images], axis=2) for i in range(len(self.labels))]

    def _decode_camera_to_ram(self, camera: str, progress: bool, name: str) -> list[np.ndarray]:
        images: list[np.ndarray | None] = [None] * len(self.labels)
        groups = list(self.labels.groupby("episode_index", sort=True))
        iterator = groups
        if progress and tqdm is not None:
            iterator = tqdm(groups, desc=f"cache {name}:{camera}", leave=False, dynamic_ncols=True)
        for ep, group in iterator:
            ep_i = int(ep)
            path = self.root / "videos" / VIDEO_KEYS[camera] / "chunk-000" / f"file-{ep_i:03d}.mp4"
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {path}")
            wanted: dict[int, list[int]] = {}
            for row_idx, frame_idx in zip(group.index.to_numpy(), group["frame_index"].to_numpy()):
                wanted.setdefault(int(frame_idx), []).append(int(row_idx))
            max_frame = max(wanted) if wanted else -1
            frame_idx = 0
            while frame_idx <= max_frame:
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    cap.release()
                    raise RuntimeError(f"Could not read {camera} ep={ep_i} frame={frame_idx}")
                if frame_idx in wanted:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    rgb = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
                    for row_idx in wanted[frame_idx]:
                        images[row_idx] = rgb
                frame_idx += 1
            cap.release()
        missing = [i for i, image in enumerate(images) if image is None]
        if missing:
            raise RuntimeError(f"RAM cache missing {len(missing)} frames for {camera}; first={missing[0]}")
        return images  # type: ignore[return-value]

    def _read_image(self, camera: str, ep: int, frame_idx: int) -> np.ndarray:
        key = (camera, ep)
        cap = self._caps.get(key)
        if cap is None:
            path = self.root / "videos" / VIDEO_KEYS[camera] / "chunk-000" / f"file-{ep:03d}.mp4"
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {path}")
            self._caps[key] = cap
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, bgr = cap.read()
        if not ok or bgr is None:
            raise RuntimeError(f"Could not read {camera} ep={ep} frame={frame_idx}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        if self.augment:
            if random.random() < 0.5:
                factor = random.uniform(0.85, 1.15)
                image = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
            if random.random() < 0.25:
                noise = np.random.normal(0.0, 3.0, image.shape).astype(np.float32)
                image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        x = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)
        return (x - 0.5) / 0.5


def load_labels(root: Path, camera: str, sample_stride: int) -> pd.DataFrame:
    labels_path = root / "meta" / "rlt_gate_labels.parquet"
    episodes_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels: {labels_path}")
    labels = pd.read_parquet(labels_path)
    episodes = pd.read_parquet(episodes_path)
    labels = labels.merge(episodes[["episode_index", "length"]], on="episode_index", how="left")
    labels = labels[labels["frame_index"] < labels["length"]].copy()

    required_cameras = ("front", "wrist") if camera == "both" else (camera,)
    valid_eps = []
    for ep, group in labels.groupby("episode_index"):
        ep_i = int(ep)
        paths = [root / "videos" / VIDEO_KEYS[cam] / "chunk-000" / f"file-{ep_i:03d}.mp4" for cam in required_cameras]
        if all(path.exists() for path in paths):
            valid_eps.append(ep_i)
    labels = labels[labels["episode_index"].isin(valid_eps)].copy()
    if sample_stride > 1:
        labels = labels[labels["frame_index"] % int(sample_stride) == 0].copy()
    return labels.drop(columns=["length"]).reset_index(drop=True)


def split_by_episode(labels: pd.DataFrame, val_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    episodes = sorted(int(ep) for ep in labels.episode_index.unique())
    rng = random.Random(seed)
    rng.shuffle(episodes)
    val_count = max(1, int(round(len(episodes) * val_ratio)))
    val_eps = set(episodes[:val_count])
    train = labels[~labels.episode_index.isin(val_eps)].copy()
    val = labels[labels.episode_index.isin(val_eps)].copy()
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
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
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
    pos_weight: torch.Tensor,
    label_smoothing: float,
    threshold: float,
    amp: bool,
    desc: str,
    progress: bool,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    seen = 0
    all_logits = []
    all_labels = []
    iterator = loader
    progress_bar = None
    if progress and tqdm is not None:
        progress_bar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
        iterator = progress_bar
    for images, labels in iterator:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        smooth_labels = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing
        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits = model(images).view(-1)
                loss = F.binary_cross_entropy_with_logits(logits, smooth_labels, pos_weight=pos_weight)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        total_loss += float(loss.item()) * len(labels)
        seen += len(labels)
        all_logits.append(logits.detach().float().cpu())
        all_labels.append(labels.detach().float().cpu())
        if progress_bar is not None:
            progress_bar.set_postfix(loss=f"{total_loss / max(seen, 1):.4f}")
    logits_cpu = torch.cat(all_logits)
    labels_cpu = torch.cat(all_labels)
    metrics = binary_metrics(logits_cpu, labels_cpu, threshold)
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    return metrics


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def config_to_jsonable(cfg: RltGateTrainConfig, *, dataset: Path, input_channels: int | None = None) -> dict:
    payload = asdict(cfg)
    payload["dataset"] = str(dataset)
    payload["output_dir"] = str(cfg.output_dir)
    if input_channels is not None:
        payload["input_channels"] = int(input_channels)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small image-to-RLT-gate classifier.")
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
    parser.add_argument("--cache-images", choices=("ram", "none"), default=CFG.cache_images)
    parser.add_argument("--device", default=CFG.device)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--threshold", type=float, default=CFG.threshold)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = RltGateTrainConfig(
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
        cache_images=args.cache_images,
        device=args.device,
        amp=not bool(args.no_amp),
        threshold=args.threshold,
    )
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = choose_device(cfg.device)
    root = cfg.dataset.resolve()
    labels = load_labels(root, cfg.camera, cfg.sample_stride)
    train_labels, val_labels = split_by_episode(labels, cfg.val_ratio, cfg.seed)
    channels = 6 if cfg.camera == "both" else 3
    model = build_model(cfg.model, input_channels=channels).to(device)
    positives = float(train_labels.rlt_phase.sum())
    negatives = float(len(train_labels) - positives)
    pos_weight = torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = (cfg.output_dir / f"rlt_gate_{stamp}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config_to_jsonable(cfg, dataset=root), indent=2), encoding="utf-8")

    cache_channels = 6 if cfg.camera == "both" else 3
    cache_gib = len(labels) * cfg.image_size * cfg.image_size * cache_channels / (1024**3)
    print(f"[data] root={root} camera={cfg.camera} rows={len(labels)} train={len(train_labels)} val={len(val_labels)}")
    print(f"[data] train_pos={int(positives)} train_neg={int(negatives)} pos_weight={float(pos_weight.item()):.3f}")
    print(f"[model] name={cfg.model} params={sum(p.numel() for p in model.parameters())} device={device} output={out}")
    print(f"[cache] mode={cfg.cache_images} estimated_uint8={cache_gib:.2f} GiB")
    train_ds = RltGateDataset(
        root,
        train_labels,
        cfg.camera,
        cfg.image_size,
        augment=True,
        cache_images=cfg.cache_images,
        progress=not args.no_progress,
        name="train",
    )
    val_ds = RltGateDataset(
        root,
        val_labels,
        cfg.camera,
        cfg.image_size,
        augment=False,
        cache_images=cfg.cache_images,
        progress=not args.no_progress,
        name="val",
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

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
            pos_weight,
            cfg.label_smoothing,
            cfg.threshold,
            cfg.amp,
            desc=f"train {epoch:03d}/{cfg.epochs:03d}",
            progress=not args.no_progress,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                None,
                None,
                device,
                pos_weight,
                0.0,
                cfg.threshold,
                False,
                desc=f"val   {epoch:03d}/{cfg.epochs:03d}",
                progress=not args.no_progress,
            )
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} train_f1={train_metrics['f1']:.3f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.3f} "
            f"val_p={val_metrics['precision']:.3f} val_r={val_metrics['recall']:.3f} val_f1={val_metrics['f1']:.3f}"
        )
        ckpt = {
            "model": model.state_dict(),
            "config": config_to_jsonable(cfg, dataset=root, input_channels=channels),
            "epoch": epoch,
            "metrics": row,
        }
        torch.save(ckpt, last_path)
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(ckpt, best_path)
    pd.DataFrame(history).to_csv(out / "metrics.csv", index=False)
    print(f"[done] best_f1={best_f1:.3f} best={best_path} last={last_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
