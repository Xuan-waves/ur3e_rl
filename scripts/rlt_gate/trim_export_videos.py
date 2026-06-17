from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

import av
import cv2
import pandas as pd


VIDEO_KEYS = (
    "observation.images.cam_front",
    "observation.images.cam_wrist",
)


def _frame_count(path: Path) -> tuple[int, float, tuple[int, int]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video: {path}")
    frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return frames, fps, (width, height)


def _trim_video(src: Path, dst: Path, expected_frames: int, fps: float, size: tuple[int, int]) -> int:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(dst), mode="w", options={"movflags": "faststart"})
    stream = container.add_stream("h264", rate=int(round(fps)))
    stream.width = int(size[0])
    stream.height = int(size[1])
    stream.pix_fmt = "yuv420p"
    written = 0
    try:
        while written < expected_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
            written += 1
        for packet in stream.encode():
            container.mux(packet)
    finally:
        cap.release()
        container.close()
    return written


def trim_dataset(root: Path, *, execute: bool, backup_dir: Path | None) -> bool:
    if hasattr(cv2, "setLogLevel"):
        cv2.setLogLevel(0)

    episodes_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if not episodes_path.exists():
        raise FileNotFoundError(f"Missing episode metadata: {episodes_path}")
    episodes = pd.read_parquet(episodes_path).sort_values("episode_index")

    ok = True
    if backup_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        backup_dir = root / "meta" / f"video_trim_backup_{stamp}"

    for _, row in episodes.iterrows():
        ep = int(row["episode_index"])
        expected = int(row["length"])
        for video_key in VIDEO_KEYS:
            src = root / "videos" / video_key / "chunk-000" / f"file-{ep:03d}.mp4"
            if not src.exists():
                print(f"[missing] ep={ep:06d} {video_key}: {src}")
                continue
            frames, fps, size = _frame_count(src)
            if frames == expected:
                print(f"[ok] ep={ep:06d} {video_key} frames={frames}")
                continue
            if frames < expected:
                print(f"[short] ep={ep:06d} {video_key} frames={frames} expected={expected}; not modified")
                ok = False
                continue

            print(f"[trim] ep={ep:06d} {video_key} frames={frames} -> {expected} fps={fps:.3f}")
            if not execute:
                continue

            relative = src.relative_to(root)
            backup_path = backup_dir / relative
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            if not backup_path.exists():
                shutil.copy2(src, backup_path)

            tmp = src.with_suffix(".trim.tmp.mp4")
            written = _trim_video(src, tmp, expected, fps, size)
            if written != expected:
                tmp.unlink(missing_ok=True)
                print(f"[error] ep={ep:06d} {video_key}: wrote {written}/{expected}")
                ok = False
                continue
            new_frames, _, _ = _frame_count(tmp)
            if new_frames != expected:
                tmp.unlink(missing_ok=True)
                print(f"[error] ep={ep:06d} {video_key}: encoded frame count {new_frames}/{expected}")
                ok = False
                continue
            tmp.replace(src)

    if execute:
        print(f"[backup] {backup_dir}")
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trim exported per-episode LeRobot videos to metadata length.")
    parser.add_argument("--dataset", type=Path, default=Path("datasets/lerobot-export(2)"))
    parser.add_argument("--execute", action="store_true", help="Actually rewrite videos. Omit for dry run.")
    parser.add_argument("--backup-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ok = trim_dataset(args.dataset.resolve(), execute=bool(args.execute), backup_dir=args.backup_dir)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
