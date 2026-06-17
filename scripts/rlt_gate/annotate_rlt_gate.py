from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

import cv2
import numpy as np
import pandas as pd


VIDEO_KEYS = {
    "front": "observation.images.cam_front",
    "wrist": "observation.images.cam_wrist",
}


@dataclass
class Interval:
    start: int
    end: int

    def as_dict(self) -> dict[str, int]:
        return {"start": int(self.start), "end": int(self.end)}


@dataclass
class EpisodeAnnotation:
    length: int
    intervals: list[Interval] = field(default_factory=list)


class RltGateAnnotator:
    def __init__(self, cfg: argparse.Namespace):
        self.cfg = cfg
        self.root = cfg.dataset.resolve()
        self.fps = float(cfg.fps)
        self.camera = cfg.camera
        self.window = "RLT gate annotator"
        self.episodes_df = self._load_episodes()
        self.data_df = self._load_data()
        self.annotations_path = cfg.output_json or (self.root / "meta" / "rlt_gate_annotations.json")
        self.labels_path = cfg.output_parquet or (self.root / "meta" / "rlt_gate_labels.parquet")
        self.annotations: dict[int, EpisodeAnnotation] = self._load_annotations()
        self.active_start: int | None = None
        self.playing = True
        self.current_frame = 0
        self.episode_pos = 0
        if hasattr(cv2, "setLogLevel"):
            cv2.setLogLevel(0)

    def run(self) -> int:
        if self.cfg.check_videos:
            ok = self.check_videos()
            return 0 if ok else 2

        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        try:
            while 0 <= self.episode_pos < len(self.episodes_df):
                if not self._run_episode(self.episode_pos):
                    break
        finally:
            cv2.destroyAllWindows()
            self.save()
        return 0

    def check_videos(self) -> bool:
        ok = True
        for _, row in self.episodes_df.iterrows():
            ep = int(row["episode_index"])
            expected = int(row["length"])
            paths = self._video_paths(ep)
            for label, path in paths.items():
                if not path.exists():
                    print(f"[missing] ep={ep:06d} {label}: {path}")
                    ok = False
                    continue
                cap = cv2.VideoCapture(str(path))
                frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                fps = float(cap.get(cv2.CAP_PROP_FPS))
                cap.release()
                status = "ok" if frames == expected else "mismatch"
                print(f"[{status}] ep={ep:06d} {label} frames={frames} expected={expected} fps={fps:.3f}")
                ok = ok and frames == expected
        return ok

    def save(self) -> None:
        self.annotations_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "dataset_root": str(self.root),
            "fps": self.fps,
            "camera": self.camera,
            "semantics": "rlt_phase=1 means the frame should be considered part of the future RL/refinement phase.",
            "episodes": {
                str(ep): {
                    "length": ann.length,
                    "intervals": [interval.as_dict() for interval in ann.intervals],
                }
                for ep, ann in sorted(self.annotations.items())
            },
            "updated_at_unix": time.time(),
        }
        self.annotations_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        labels = self._make_labels_df()
        labels.to_parquet(self.labels_path, index=False)
        print(f"[save] json={self.annotations_path}")
        print(f"[save] labels={self.labels_path} rows={len(labels)} positives={int(labels['rlt_phase'].sum())}")

    def _run_episode(self, episode_pos: int) -> bool:
        row = self.episodes_df.iloc[episode_pos]
        ep = int(row["episode_index"])
        length = int(row["length"])
        self.annotations.setdefault(ep, EpisodeAnnotation(length=length))
        self.annotations[ep].length = length
        self.active_start = None
        self.current_frame = min(max(self.current_frame, 0), length - 1)
        if not self._episode_videos_available(ep):
            print(f"[skip] ep={ep:06d} missing one or more videos; labels remain zero unless already annotated.")
            if self.episode_pos >= len(self.episodes_df) - 1:
                return False
            self.episode_pos += 1
            self.current_frame = 0
            return True
        caps = self._open_caps(ep)
        try:
            while True:
                frame = self._read_display_frame(caps, ep, self.current_frame, length)
                cv2.imshow(self.window, frame)
                delay = max(1, int(1000.0 / self.fps)) if self.playing else 0
                key = cv2.waitKey(delay) & 0xFF
                action = self._handle_key(key, ep, length)
                if action == "quit":
                    return False
                if action == "next":
                    self.episode_pos = min(self.episode_pos + 1, len(self.episodes_df) - 1)
                    self.current_frame = 0
                    return True
                if action == "prev":
                    self.episode_pos = max(self.episode_pos - 1, 0)
                    self.current_frame = 0
                    return True
                if self.playing:
                    self.current_frame += 1
                    if self.current_frame >= length:
                        if self.active_start is not None:
                            self._finish_interval(ep, length - 1)
                        self.current_frame = length - 1
                        self.playing = False
        finally:
            for cap in caps.values():
                cap.release()
        return True

    def _handle_key(self, key: int, ep: int, length: int) -> str | None:
        if key in (255, -1):
            return None
        char = chr(key).lower() if 0 <= key < 128 else ""
        if char == "q":
            return "quit"
        if char == "s":
            self.save()
            return None
        if char == " ":
            self.playing = not self.playing
            return None
        if char == "r":
            if self.active_start is None:
                self.active_start = self.current_frame
                print(f"[mark-start] ep={ep:06d} frame={self.active_start}")
            else:
                self._finish_interval(ep, self.current_frame)
            return None
        if char == "z":
            self._undo(ep)
            return None
        if char == "n":
            return "next"
        if char == "p":
            return "prev"
        if char == "a":
            self.current_frame = max(0, self.current_frame - 1)
            self.playing = False
            return None
        if char == "d":
            self.current_frame = min(length - 1, self.current_frame + 1)
            self.playing = False
            return None
        if char == "j":
            self.current_frame = max(0, self.current_frame - int(round(self.fps)))
            self.playing = False
            return None
        if char == "l":
            self.current_frame = min(length - 1, self.current_frame + int(round(self.fps)))
            self.playing = False
            return None
        if char == "0":
            self.current_frame = 0
            self.playing = False
            return None
        if char == "b":
            self._replay_last_interval(ep)
            return None
        if char == "e":
            self.current_frame = 0
            self.playing = True
            return None
        return None

    def _finish_interval(self, ep: int, end_frame: int) -> None:
        assert self.active_start is not None
        start = min(self.active_start, end_frame)
        end = max(self.active_start, end_frame)
        ann = self.annotations[ep]
        ann.intervals.append(Interval(start=start, end=end))
        ann.intervals = self._merge_intervals(ann.intervals, ann.length)
        print(f"[mark-end] ep={ep:06d} interval=[{start}, {end}] intervals={len(ann.intervals)}")
        self.active_start = None

    def _undo(self, ep: int) -> None:
        if self.active_start is not None:
            print(f"[undo] canceled active start ep={ep:06d} frame={self.active_start}")
            self.active_start = None
            return
        ann = self.annotations[ep]
        if not ann.intervals:
            print(f"[undo] ep={ep:06d} has no intervals")
            return
        removed = ann.intervals.pop()
        print(f"[undo] ep={ep:06d} removed interval=[{removed.start}, {removed.end}]")

    def _replay_last_interval(self, ep: int) -> None:
        ann = self.annotations[ep]
        if not ann.intervals:
            print(f"[replay] ep={ep:06d} has no interval")
            return
        interval = ann.intervals[-1]
        self.current_frame = interval.start
        self.playing = True
        print(f"[replay] ep={ep:06d} interval=[{interval.start}, {interval.end}]")

    def _read_display_frame(
        self,
        caps: dict[str, cv2.VideoCapture],
        ep: int,
        frame_idx: int,
        length: int,
    ) -> np.ndarray:
        images = []
        for label, cap in caps.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, image = cap.read()
            if not ok or image is None:
                image = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(image, f"{label}: read failed", (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            images.append(image)
        display = images[0] if len(images) == 1 else np.hstack(images)
        return self._draw_overlay(display, ep, frame_idx, length)

    def _draw_overlay(self, image: np.ndarray, ep: int, frame_idx: int, length: int) -> np.ndarray:
        out = image.copy()
        ann = self.annotations[ep]
        marked = self._is_marked(ann, frame_idx)
        color = (0, 220, 0) if marked or self.active_start is not None else (240, 240, 240)
        state = "RLT=1" if marked else "RLT=0"
        if self.active_start is not None:
            state += f" ACTIVE from {self.active_start}"
        header = (
            f"ep {self.episode_pos + 1}/{len(self.episodes_df)} id={ep:06d} "
            f"frame {frame_idx + 1}/{length} t={frame_idx / self.fps:.2f}s {state}"
        )
        help_text = "R toggle  Z undo  Space play/pause  A/D step  J/L +/-1s  B replay mark  E replay ep  N/P ep  S save  Q quit"
        cv2.rectangle(out, (0, 0), (out.shape[1], 78), (0, 0, 0), -1)
        cv2.putText(out, header, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)
        cv2.putText(out, help_text, (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1)

        bar_y = out.shape[0] - 24
        cv2.rectangle(out, (0, bar_y), (out.shape[1], out.shape[0]), (0, 0, 0), -1)
        for interval in ann.intervals:
            x1 = int(out.shape[1] * interval.start / max(length - 1, 1))
            x2 = int(out.shape[1] * interval.end / max(length - 1, 1))
            cv2.rectangle(out, (x1, bar_y + 4), (max(x2, x1 + 2), out.shape[0] - 5), (0, 180, 0), -1)
        if self.active_start is not None:
            x1 = int(out.shape[1] * self.active_start / max(length - 1, 1))
            x2 = int(out.shape[1] * frame_idx / max(length - 1, 1))
            cv2.rectangle(out, (min(x1, x2), bar_y + 4), (max(x1, x2) + 2, out.shape[0] - 5), (0, 220, 220), -1)
        x = int(out.shape[1] * frame_idx / max(length - 1, 1))
        cv2.line(out, (x, bar_y), (x, out.shape[0]), (255, 255, 255), 2)
        return out

    def _open_caps(self, ep: int) -> dict[str, cv2.VideoCapture]:
        paths = self._video_paths(ep)
        caps = {}
        selected = paths if self.camera == "both" else {self.camera: paths[self.camera]}
        for label, path in selected.items():
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open {label} video for episode {ep}: {path}")
            caps[label] = cap
        return caps

    def _video_paths(self, ep: int) -> dict[str, Path]:
        return {
            label: self.root / "videos" / video_key / "chunk-000" / f"file-{ep:03d}.mp4"
            for label, video_key in VIDEO_KEYS.items()
        }

    def _episode_videos_available(self, ep: int) -> bool:
        paths = self._video_paths(ep)
        selected = paths if self.camera == "both" else {self.camera: paths[self.camera]}
        return all(path.exists() for path in selected.values())

    def _make_labels_df(self) -> pd.DataFrame:
        rows = []
        for _, ep_row in self.episodes_df.iterrows():
            ep = int(ep_row["episode_index"])
            length = int(ep_row["length"])
            start_index = int(ep_row["dataset_from_index"])
            labels = np.zeros(length, dtype=np.uint8)
            ann = self.annotations.get(ep, EpisodeAnnotation(length=length))
            for interval in self._merge_intervals(ann.intervals, length):
                labels[interval.start : interval.end + 1] = 1
            for frame_idx in range(length):
                rows.append(
                    {
                        "episode_index": ep,
                        "frame_index": frame_idx,
                        "index": start_index + frame_idx,
                        "timestamp": frame_idx / self.fps,
                        "rlt_phase": int(labels[frame_idx]),
                    }
                )
        return pd.DataFrame(rows)

    def _load_episodes(self) -> pd.DataFrame:
        path = self.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing LeRobot episode metadata: {path}")
        df = pd.read_parquet(path)
        required = {"episode_index", "length", "dataset_from_index", "dataset_to_index"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Episode metadata missing columns: {sorted(missing)}")
        return df.sort_values("episode_index").reset_index(drop=True)

    def _load_data(self) -> pd.DataFrame:
        path = self.root / "data" / "chunk-000" / "file-000.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return pd.DataFrame()

    def _load_annotations(self) -> dict[int, EpisodeAnnotation]:
        annotations = {
            int(row["episode_index"]): EpisodeAnnotation(length=int(row["length"]))
            for _, row in self.episodes_df.iterrows()
        }
        if not self.annotations_path.exists():
            return annotations
        payload = json.loads(self.annotations_path.read_text(encoding="utf-8"))
        for ep_str, item in payload.get("episodes", {}).items():
            ep = int(ep_str)
            length = int(item.get("length", annotations.get(ep, EpisodeAnnotation(0)).length))
            intervals = [
                Interval(start=int(interval["start"]), end=int(interval["end"]))
                for interval in item.get("intervals", [])
            ]
            annotations[ep] = EpisodeAnnotation(length=length, intervals=self._merge_intervals(intervals, length))
        print(f"[load] existing annotations: {self.annotations_path}")
        return annotations

    @staticmethod
    def _merge_intervals(intervals: list[Interval], length: int) -> list[Interval]:
        cleaned = []
        for interval in intervals:
            start = max(0, min(int(interval.start), length - 1))
            end = max(0, min(int(interval.end), length - 1))
            if end < start:
                start, end = end, start
            cleaned.append(Interval(start, end))
        if not cleaned:
            return []
        cleaned.sort(key=lambda item: item.start)
        merged = [cleaned[0]]
        for interval in cleaned[1:]:
            last = merged[-1]
            if interval.start <= last.end + 1:
                last.end = max(last.end, interval.end)
            else:
                merged.append(interval)
        return merged

    @staticmethod
    def _is_marked(ann: EpisodeAnnotation, frame_idx: int) -> bool:
        return any(interval.start <= frame_idx <= interval.end for interval in ann.intervals)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manually annotate RLT/RL gate phases on exported LeRobot videos.")
    parser.add_argument("--dataset", type=Path, default=Path("datasets/lerobot-export(3)"))
    parser.add_argument("--camera", choices=("front", "wrist", "both"), default="both")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-parquet", type=Path)
    parser.add_argument("--check-videos", action="store_true", help="Only check per-episode mp4 frame counts.")
    return parser.parse_args()


def main() -> int:
    return RltGateAnnotator(parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(main())
