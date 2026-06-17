from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import imageio_ffmpeg
import pandas as pd


VIDEO_KEYS = ("observation.images.cam_front", "observation.images.cam_wrist")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Trim split-per-episode LeRobot mp4 files to the frame count recorded in meta/episodes."
    )
    parser.add_argument("dataset", type=Path, help="LeRobot dataset root.")
    parser.add_argument("--fps", type=float, default=None, help="Override dataset fps.")
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="Backup directory for the original videos. Default: <dataset>/videos_backup_before_trim_<stamp>",
    )
    parser.add_argument("--no-backup", action="store_true", help="Do not copy the original videos before trimming.")
    parser.add_argument("--dry-run", action="store_true", help="Only report mismatches.")
    args = parser.parse_args()

    root = args.dataset.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    fps = float(args.fps or _read_dataset_fps(root))
    episodes_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if not episodes_path.exists():
        raise FileNotFoundError(f"Missing episodes parquet: {episodes_path}")

    episodes = pd.read_parquet(episodes_path)
    mismatches = _scan(root, episodes)
    report = {
        "dataset": str(root),
        "fps": fps,
        "total_videos": int(len(episodes) * len(VIDEO_KEYS)),
        "mismatch_videos": len(mismatches),
        "mismatches": mismatches,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "mismatches"}, indent=2))
    if mismatches:
        print("Top mismatches:")
        for item in sorted(mismatches, key=lambda x: abs(x["extra_frames"]), reverse=True)[:20]:
            print(
                f"  ep={item['episode_index']:03d} {item['video_key']} "
                f"need={item['expected_frames']} got={item['actual_frames']} extra={item['extra_frames']}"
            )

    if args.dry_run:
        _write_report(root, report, suffix="dry_run")
        return 0
    if not mismatches:
        _update_episode_video_timestamps(episodes, fps)
        episodes.to_parquet(episodes_path, index=False)
        _write_report(root, report, suffix="already_ok")
        return 0

    if not args.no_backup:
        backup = args.backup_dir.expanduser().resolve() if args.backup_dir else _default_backup_dir(root)
        if backup.exists():
            raise FileExistsError(f"Backup directory already exists: {backup}")
        shutil.copytree(root / "videos", backup)
        report["backup_dir"] = str(backup)
        print(f"Backed up original videos to: {backup}")

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    for item in mismatches:
        expected = int(item["expected_frames"])
        actual = int(item["actual_frames"])
        if actual < expected:
            raise RuntimeError(f"Cannot trim short video: {item}")
        _trim_video(Path(item["path"]), expected, fps, ffmpeg)

    _update_episode_video_timestamps(episodes, fps)
    episodes.to_parquet(episodes_path, index=False)

    post = _scan(root, episodes)
    report["post_mismatch_videos"] = len(post)
    report["post_mismatches"] = post
    _write_report(root, report, suffix="trim")
    if post:
        raise RuntimeError(f"Some videos still have wrong frame counts; see trim report. count={len(post)}")
    print("Trim complete. All split episode videos match meta/episodes length.")
    return 0


def _read_dataset_fps(root: Path) -> float:
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    return float(info["fps"])


def _scan(root: Path, episodes: pd.DataFrame) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for _, row in episodes.iterrows():
        episode = int(row["episode_index"])
        expected = int(row["length"])
        for video_key in VIDEO_KEYS:
            file_index_col = f"videos/{video_key}/file_index"
            chunk_index_col = f"videos/{video_key}/chunk_index"
            file_index = int(row[file_index_col]) if file_index_col in row else episode
            chunk_index = int(row[chunk_index_col]) if chunk_index_col in row else 0
            path = root / "videos" / video_key / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"
            actual, fps = _video_info(path)
            if actual != expected:
                mismatches.append(
                    {
                        "episode_index": episode,
                        "video_key": video_key,
                        "path": str(path),
                        "expected_frames": expected,
                        "actual_frames": actual,
                        "extra_frames": actual - expected,
                        "fps": fps,
                    }
                )
    return mismatches


def _video_info(path: Path) -> tuple[int, float]:
    if not path.exists():
        return -1, 0.0
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return -1, 0.0
    frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return frames, fps


def _trim_video(path: Path, frames: int, fps: float, ffmpeg: str) -> None:
    tmp = path.with_suffix(".trim_tmp.mp4")
    if tmp.exists():
        tmp.unlink()
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-frames:v",
        str(frames),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-r",
        f"{fps:g}",
        str(tmp),
    ]
    subprocess.run(cmd, check=True)
    actual, _ = _video_info(tmp)
    if actual != frames:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Trimmed video has wrong frame count: {tmp} expected={frames} actual={actual}")
    tmp.replace(path)


def _update_episode_video_timestamps(episodes: pd.DataFrame, fps: float) -> None:
    duration = episodes["length"].astype(float) / float(fps)
    for video_key in VIDEO_KEYS:
        from_col = f"videos/{video_key}/from_timestamp"
        to_col = f"videos/{video_key}/to_timestamp"
        if from_col in episodes.columns:
            episodes[from_col] = 0.0
        if to_col in episodes.columns:
            episodes[to_col] = duration


def _default_backup_dir(root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / f"videos_backup_before_trim_{stamp}"


def _write_report(root: Path, report: dict[str, Any], *, suffix: str) -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = root / "meta" / f"video_trim_report_{suffix}_{stamp}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote report: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
