from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")
if "QT_QPA_FONTDIR" not in os.environ:
    for font_dir in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts"):
        if Path(font_dir).exists():
            os.environ["QT_QPA_FONTDIR"] = font_dir
            break

import cv2
import numpy as np
import pandas as pd
import torch

try:
    import av
except ImportError:
    av = None

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rlt_gate.train_rlt_gate import VIDEO_KEYS, binary_metrics, build_model


class PreviewWriter:
    def __init__(self, path: Path, fps: float, width: int, height: int):
        self.path = path
        self._cv_writer = None
        self._container = None
        self._stream = None
        if av is not None:
            self._container = av.open(str(path), mode="w", options={"movflags": "faststart"})
            self._stream = self._container.add_stream("h264", rate=max(1, int(round(fps))))
            self._stream.width = int(width)
            self._stream.height = int(height)
            self._stream.pix_fmt = "yuv420p"
        else:
            self._cv_writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not self._cv_writer.isOpened():
                raise RuntimeError(f"Could not open output video writer: {path}")

    def write(self, bgr: np.ndarray) -> None:
        if self._container is not None:
            frame = av.VideoFrame.from_ndarray(bgr, format="bgr24")
            for packet in self._stream.encode(frame):
                self._container.mux(packet)
        else:
            self._cv_writer.write(bgr)

    def close(self) -> None:
        if self._container is not None:
            for packet in self._stream.encode():
                self._container.mux(packet)
            self._container.close()
        elif self._cv_writer is not None:
            self._cv_writer.release()


@dataclass
class GateResult:
    name: str
    path: Path
    camera: str
    model: str
    image_size: int
    threshold: float
    positive_threshold: float
    negative_threshold: float
    hold_frames: int
    acc: float
    precision: float
    recall: float
    f1: float
    hysteresis_acc: float
    hysteresis_precision: float
    hysteresis_recall: float
    hysteresis_f1: float


def load_checkpoint(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt.get("config")
    if not isinstance(cfg, dict):
        raise ValueError(f"Checkpoint has no config dict: {path}")
    camera = cfg.get("camera", "front")
    input_channels = int(cfg.get("input_channels", 6 if camera == "both" else 3))
    model = build_model(str(cfg.get("model", "tiny")), input_channels=input_channels)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, cfg


def load_labels(root: Path) -> pd.DataFrame:
    labels_path = root / "meta" / "rlt_gate_labels.parquet"
    episodes_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    labels = pd.read_parquet(labels_path)
    episodes = pd.read_parquet(episodes_path)
    labels = labels.merge(episodes[["episode_index", "length"]], on="episode_index", how="left")
    labels = labels[labels["frame_index"] < labels["length"]].drop(columns=["length"])
    return labels.reset_index(drop=True)


def video_path(root: Path, camera: str, episode: int) -> Path:
    return root / "videos" / VIDEO_KEYS[camera] / "chunk-000" / f"file-{episode:03d}.mp4"


def valid_episode_labels(root: Path, labels: pd.DataFrame, camera: str) -> pd.DataFrame:
    required = ("front", "wrist") if camera == "both" else (camera,)
    valid_eps = []
    for ep in sorted(int(ep) for ep in labels.episode_index.unique()):
        if all(video_path(root, cam, ep).exists() for cam in required):
            valid_eps.append(ep)
    return labels[labels.episode_index.isin(valid_eps)].reset_index(drop=True)


def read_episode_frames(root: Path, camera: str, episode: int, frame_indices: list[int], image_size: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path(root, camera, episode)))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path(root, camera, episode)}")
    wanted = {int(frame_idx) for frame_idx in frame_indices}
    max_frame = max(wanted) if wanted else -1
    frames: dict[int, np.ndarray] = {}
    frame_idx = 0
    while frame_idx <= max_frame:
        ok, bgr = cap.read()
        if not ok or bgr is None:
            cap.release()
            raise RuntimeError(f"Could not read {camera} episode={episode} frame={frame_idx}")
        if frame_idx in wanted:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
            frames[frame_idx] = rgb
        frame_idx += 1
    cap.release()
    return [frames[int(frame_idx)] for frame_idx in frame_indices]


def make_inputs(root: Path, camera: str, episode: int, frame_indices: list[int], image_size: int) -> torch.Tensor:
    if camera == "both":
        front = read_episode_frames(root, "front", episode, frame_indices, image_size)
        wrist = read_episode_frames(root, "wrist", episode, frame_indices, image_size)
        images = [np.concatenate([front[i], wrist[i]], axis=2) for i in range(len(frame_indices))]
    else:
        images = read_episode_frames(root, camera, episode, frame_indices, image_size)
    arr = np.stack(images).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(0, 3, 1, 2)
    return (x - 0.5) / 0.5


def predict_labels(
    root: Path,
    labels: pd.DataFrame,
    model: torch.nn.Module,
    camera: str,
    image_size: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    probs = np.zeros(len(labels), dtype=np.float32)
    offset = 0
    for episode, group in labels.groupby("episode_index", sort=True):
        frame_indices = [int(v) for v in group.frame_index.to_numpy()]
        for start in range(0, len(frame_indices), batch_size):
            chunk = frame_indices[start : start + batch_size]
            x = make_inputs(root, camera, int(episode), chunk, image_size).to(device)
            with torch.no_grad():
                logits = model(x).view(-1)
                pred = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
            probs[offset + start : offset + start + len(chunk)] = pred
        offset += len(group)
    return probs


def hysteresis_phase(probs: np.ndarray, pos_threshold: float, neg_threshold: float, hold_frames: int) -> np.ndarray:
    phase = 0
    pos_count = 0
    neg_count = 0
    out = np.zeros(len(probs), dtype=np.int64)
    for i, prob in enumerate(probs):
        if prob >= pos_threshold:
            pos_count += 1
            neg_count = 0
        elif prob <= neg_threshold:
            neg_count += 1
            pos_count = 0
        else:
            pos_count = 0
            neg_count = 0
        if phase == 0 and pos_count >= hold_frames:
            phase = 1
        elif phase == 1 and neg_count >= hold_frames:
            phase = 0
        out[i] = phase
    return out


def metrics_from_arrays(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    logits = torch.from_numpy(np.log(np.clip(probs, 1e-6, 1 - 1e-6) / np.clip(1 - probs, 1e-6, 1)))
    return binary_metrics(logits.float(), torch.from_numpy(labels.astype(np.float32)), threshold)


def phase_metrics(pred_phase: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    pred = torch.from_numpy(pred_phase.astype(bool))
    target = torch.from_numpy(labels.astype(bool))
    tp = torch.logical_and(pred, target).sum().item()
    tn = torch.logical_and(~pred, ~target).sum().item()
    fp = torch.logical_and(pred, ~target).sum().item()
    fn = torch.logical_and(~pred, target).sum().item()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "acc": (tp + tn) / max(tp + tn + fp + fn, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def draw_overlay(frame: np.ndarray, text_lines: list[str], gt: int, phase: int, prob: float) -> np.ndarray:
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    panel_h = 92 + 24 * max(0, len(text_lines) - 3)
    cv2.rectangle(canvas, (0, 0), (w, panel_h), (0, 0, 0), -1)
    for i, line in enumerate(text_lines):
        cv2.putText(canvas, line, (12, 26 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    bar_x, bar_y, bar_w, bar_h = 12, panel_h - 26, min(w - 24, 420), 12
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (70, 70, 70), -1)
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + int(bar_w * prob), bar_y + bar_h), (0, 210, 255), -1)
    color = (0, 220, 0) if gt == phase else (0, 0, 255)
    cv2.circle(canvas, (w - 32, 32), 14, color, -1)
    return canvas


def read_display_frame(caps: dict[str, cv2.VideoCapture], camera: str) -> tuple[bool, np.ndarray | None]:
    if camera != "both":
        return caps[camera].read()
    ok_front, front = caps["front"].read()
    ok_wrist, wrist = caps["wrist"].read()
    if not ok_front or front is None or not ok_wrist or wrist is None:
        return False, None
    if wrist.shape[:2] != front.shape[:2]:
        wrist = cv2.resize(wrist, (front.shape[1], front.shape[0]), interpolation=cv2.INTER_AREA)
    cv2.putText(front, "front", (12, front.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(wrist, "wrist", (12, wrist.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return True, np.hstack([front, wrist])


def preview_episode(
    root: Path,
    checkpoint: Path,
    model: torch.nn.Module,
    cfg: dict,
    episode: int,
    device: torch.device,
    batch_size: int,
    pos_threshold: float,
    neg_threshold: float,
    hold_frames: int,
    output_video: Path | None,
    show: bool,
) -> None:
    camera = str(cfg.get("camera", "front"))
    image_size = int(cfg.get("image_size", 160))
    labels = load_labels(root)
    labels = valid_episode_labels(root, labels, camera)
    ep_labels = labels[labels.episode_index == episode].reset_index(drop=True)
    if ep_labels.empty:
        raise ValueError(f"No labels/video for episode {episode}")
    print(f"[preview] episode={episode:03d} frames={len(ep_labels)} show={show} output={output_video}")
    probs = predict_labels(root, ep_labels, model, camera, image_size, device, batch_size)
    phase = hysteresis_phase(probs, pos_threshold, neg_threshold, hold_frames)

    display_cameras = ("front", "wrist") if camera == "both" else (camera,)
    caps = {cam: cv2.VideoCapture(str(video_path(root, cam, episode))) for cam in display_cameras}
    for cam, cap in caps.items():
        if not cap.isOpened():
            raise RuntimeError(f"Could not open preview video: {video_path(root, cam, episode)}")
    fps = caps[display_cameras[0]].get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    if output_video is not None:
        output_video.parent.mkdir(parents=True, exist_ok=True)
        width = int(caps[display_cameras[0]].get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(caps[display_cameras[0]].get(cv2.CAP_PROP_FRAME_HEIGHT))
        if camera == "both":
            width *= 2
        writer = PreviewWriter(output_video, fps, width, height)
    if show:
        cv2.namedWindow("RLT gate preview", cv2.WINDOW_NORMAL)
    frame_to_row = {int(row.frame_index): i for i, row in ep_labels.iterrows()}
    i = 0
    while True:
        ok, frame = read_display_frame(caps, camera)
        if not ok or frame is None:
            break
        if i in frame_to_row:
            row_i = frame_to_row[i]
            gt = int(ep_labels.iloc[row_i].rlt_phase)
            prob = float(probs[row_i])
            pred = int(prob >= float(cfg.get("threshold", 0.5)))
            ph = int(phase[row_i])
            frame = draw_overlay(
                frame,
                [
                    f"episode={episode:03d} frame={i:04d} model={cfg.get('model')} camera={camera}",
                    f"GT={gt} pred={pred} prob={prob:.3f} hyst={ph}",
                    f"ckpt={checkpoint.parent.name}",
                ],
                gt,
                ph,
                prob,
            )
        if writer is not None:
            writer.write(frame)
        if show:
            cv2.imshow("RLT gate preview", frame)
            key = cv2.waitKey(max(1, int(1000 / fps))) & 0xFF
            if key in (ord("q"), 27):
                break
        i += 1
    for cap in caps.values():
        cap.release()
    if writer is not None:
        writer.close()
        print(f"[preview] saved {output_video}")
    if show:
        cv2.destroyAllWindows()
    print(f"[preview] done episode={episode:03d}")


def episode_result(
    root: Path,
    checkpoint: Path,
    model: torch.nn.Module,
    cfg: dict,
    episode: int,
    device: torch.device,
    batch_size: int,
    pos_threshold: float | None,
    neg_threshold: float | None,
    hold_frames: int,
) -> GateResult:
    camera = str(cfg.get("camera", "front"))
    image_size = int(cfg.get("image_size", 160))
    labels = valid_episode_labels(root, load_labels(root), camera)
    labels = labels[labels.episode_index == episode].reset_index(drop=True)
    if labels.empty:
        raise ValueError(f"No labels/video for episode {episode}")
    probs = predict_labels(root, labels, model, camera, image_size, device, batch_size)
    y = labels.rlt_phase.to_numpy(dtype=np.int64)
    threshold = float(cfg.get("threshold", 0.5))
    pos_t = float(pos_threshold if pos_threshold is not None else cfg.get("positive_threshold", 0.6))
    neg_t = float(neg_threshold if neg_threshold is not None else cfg.get("negative_threshold", 0.4))
    base = metrics_from_arrays(probs, y, threshold)
    phase = hysteresis_phase(probs, pos_t, neg_t, hold_frames)
    hys = phase_metrics(phase, y)
    return GateResult(
        name=checkpoint.parent.name,
        path=checkpoint,
        camera=camera,
        model=str(cfg.get("model", "tiny")),
        image_size=image_size,
        threshold=threshold,
        positive_threshold=pos_t,
        negative_threshold=neg_t,
        hold_frames=hold_frames,
        acc=base["acc"],
        precision=base["precision"],
        recall=base["recall"],
        f1=base["f1"],
        hysteresis_acc=hys["acc"],
        hysteresis_precision=hys["precision"],
        hysteresis_recall=hys["recall"],
        hysteresis_f1=hys["f1"],
    )


def eval_checkpoint(
    root: Path,
    checkpoint: Path,
    device: torch.device,
    batch_size: int,
    pos_threshold: float | None,
    neg_threshold: float | None,
    hold_frames: int,
) -> tuple[GateResult, pd.DataFrame]:
    model, cfg = load_checkpoint(checkpoint, device)
    camera = str(cfg.get("camera", "front"))
    image_size = int(cfg.get("image_size", 160))
    labels = valid_episode_labels(root, load_labels(root), camera)
    probs = predict_labels(root, labels, model, camera, image_size, device, batch_size)
    y = labels.rlt_phase.to_numpy(dtype=np.int64)
    threshold = float(cfg.get("threshold", 0.5))
    pos_t = float(pos_threshold if pos_threshold is not None else cfg.get("positive_threshold", 0.6))
    neg_t = float(neg_threshold if neg_threshold is not None else cfg.get("negative_threshold", 0.4))
    base = metrics_from_arrays(probs, y, threshold)
    phase = hysteresis_phase(probs, pos_t, neg_t, hold_frames)
    hys = phase_metrics(phase, y)
    name = checkpoint.parent.name
    result = GateResult(
        name=name,
        path=checkpoint,
        camera=camera,
        model=str(cfg.get("model", "tiny")),
        image_size=image_size,
        threshold=threshold,
        positive_threshold=pos_t,
        negative_threshold=neg_t,
        hold_frames=hold_frames,
        acc=base["acc"],
        precision=base["precision"],
        recall=base["recall"],
        f1=base["f1"],
        hysteresis_acc=hys["acc"],
        hysteresis_precision=hys["precision"],
        hysteresis_recall=hys["recall"],
        hysteresis_f1=hys["f1"],
    )
    pred_rows = labels[["episode_index", "frame_index", "index", "timestamp", "rlt_phase"]].copy()
    pred_rows["prob"] = probs
    pred_rows["pred"] = (probs >= threshold).astype(np.int64)
    pred_rows["hysteresis_phase"] = phase
    pred_rows["checkpoint"] = name
    return result, pred_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and preview RLT gate checkpoints.")
    parser.add_argument("--dataset", type=Path, default=Path("datasets/lerobot-export(2)"))
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--positive-threshold", type=float, default=None)
    parser.add_argument("--negative-threshold", type=float, default=None)
    parser.add_argument("--hold-frames", type=int, default=3)
    parser.add_argument("--save-csv", type=Path, default=None)
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--full-eval", action="store_true", help="Run full-dataset evaluation before episode preview.")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--output-video", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    root = args.dataset.resolve()
    results = []
    all_preds = []
    for checkpoint in args.checkpoint:
        checkpoint = checkpoint.expanduser().resolve()
        model, cfg = load_checkpoint(checkpoint, device)
        if args.episode is not None and not args.full_eval:
            result = episode_result(
                root,
                checkpoint,
                model,
                cfg,
                args.episode,
                device,
                args.batch_size,
                args.positive_threshold,
                args.negative_threshold,
                args.hold_frames,
            )
        else:
            result, pred_rows = eval_checkpoint(
                root,
                checkpoint,
                device,
                args.batch_size,
                args.positive_threshold,
                args.negative_threshold,
                args.hold_frames,
            )
            results.append(result)
            all_preds.append(pred_rows)
        print(
            f"{result.name}: model={result.model} camera={result.camera} image={result.image_size} "
            f"raw_f1={result.f1:.4f} raw_p={result.precision:.4f} raw_r={result.recall:.4f} "
            f"hys_f1={result.hysteresis_f1:.4f} hys_p={result.hysteresis_precision:.4f} hys_r={result.hysteresis_recall:.4f}"
        )
        if args.episode is not None:
            output_video = args.output_video
            if output_video is not None and len(args.checkpoint) > 1:
                output_video = output_video.with_name(f"{output_video.stem}_{result.name}{output_video.suffix}")
            preview_episode(
                root,
                checkpoint,
                model,
                cfg,
                args.episode,
                device,
                args.batch_size,
                result.positive_threshold,
                result.negative_threshold,
                result.hold_frames,
                output_video,
                args.show,
            )
        if args.episode is not None and not args.full_eval:
            results.append(result)
    summary = pd.DataFrame([result.__dict__ for result in results])
    if args.save_csv is not None:
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.save_csv, index=False)
        if all_preds:
            pred_path = args.save_csv.with_name(args.save_csv.stem + "_predictions.parquet")
            pd.concat(all_preds, ignore_index=True).to_parquet(pred_path, index=False)
            print(f"saved summary={args.save_csv} predictions={pred_path}")
        else:
            print(f"saved summary={args.save_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
