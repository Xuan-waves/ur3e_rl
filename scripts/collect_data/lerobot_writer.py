from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import CollectConfig


@dataclass(slots=True)
class CollectedFrame:
    front_rgb: np.ndarray
    wrist_rgb: np.ndarray
    state: np.ndarray
    action: np.ndarray
    timestamp: float


class LeRobotVrEpisodeWriter:
    """Small LeRobot writer for TCP-pose VR impedance demonstrations."""

    def __init__(self, cfg: CollectConfig):
        self.cfg = cfg
        self.dataset = None
        self.root = self._make_unused_root()
        self.episode_frame_count = 0
        self.total_saved_episodes = 0
        self.finalized = False
        self._lock = threading.RLock()

    @property
    def is_ready(self) -> bool:
        return self.dataset is not None

    def add_frame(self, frame: CollectedFrame) -> None:
        with self._lock:
            if self.dataset is None:
                self._create_dataset(frame)

            lerobot_frame = {
                "observation.images.cam_front": frame.front_rgb,
                "observation.images.cam_wrist": frame.wrist_rgb,
                "observation.state": frame.state.astype(np.float32),
                "action": frame.action.astype(np.float32),
                "task": self.cfg.task,
            }
            self.dataset.add_frame(lerobot_frame)
            self.episode_frame_count += 1

    def save_episode(self) -> bool:
        with self._lock:
            if self.dataset is None or self.episode_frame_count <= 0:
                return False
            self._repair_episode_buffer()
            try:
                self.dataset.save_episode(parallel_encoding=False)
            except TypeError as exc:
                if "parallel_encoding" not in str(exc):
                    raise
                self.dataset.save_episode()
            self.total_saved_episodes += 1
            self.episode_frame_count = 0
            return True

    def discard_episode(self) -> bool:
        with self._lock:
            if self.dataset is None or self.episode_frame_count <= 0:
                return False
            clear_buffer = getattr(self.dataset, "clear_episode_buffer", None)
            if callable(clear_buffer):
                clear_buffer()
            else:
                create_buffer = getattr(self.dataset, "create_episode_buffer", None)
                if callable(create_buffer):
                    self.dataset.episode_buffer = create_buffer()
                else:
                    episode_buffer = getattr(self.dataset, "episode_buffer", None)
                    if isinstance(episode_buffer, dict):
                        episode_buffer.clear()
            self.episode_frame_count = 0
            return True

    def finalize(self) -> None:
        with self._lock:
            if self.dataset is None or self.finalized:
                return
            finalize = getattr(self.dataset, "finalize", None)
            if callable(finalize):
                finalize()
            else:
                consolidate = getattr(self.dataset, "consolidate", None)
                if callable(consolidate):
                    consolidate()
            self.finalized = True

    def _repair_episode_buffer(self) -> None:
        episode_buffer = getattr(self.dataset, "episode_buffer", None)
        if not isinstance(episode_buffer, dict):
            return
        if "size" in episode_buffer and "task" in episode_buffer:
            return
        size = int(self.episode_frame_count)
        if size <= 0:
            return
        if "size" not in episode_buffer:
            episode_buffer["size"] = size
        if "task" not in episode_buffer:
            episode_buffer["task"] = [self.cfg.task] * size

    def _create_dataset(self, first_frame: CollectedFrame) -> None:
        front_h, front_w, front_c = first_frame.front_rgb.shape
        wrist_h, wrist_w, wrist_c = first_frame.wrist_rgb.shape
        state_names = [
            "tcp_x",
            "tcp_y",
            "tcp_z",
            "gripper",
        ]
        if self.cfg.action_position_mode == "relative":
            action_pos_names = ["delta_tcp_x", "delta_tcp_y", "delta_tcp_z"]
        else:
            action_pos_names = ["target_tcp_x", "target_tcp_y", "target_tcp_z"]
        action_names = action_pos_names + ["target_gripper"]
        features = {
            "observation.images.cam_front": {
                "dtype": "video" if self.cfg.use_videos else "image",
                "shape": (front_h, front_w, front_c),
                "names": ["height", "width", "channel"],
            },
            "observation.images.cam_wrist": {
                "dtype": "video" if self.cfg.use_videos else "image",
                "shape": (wrist_h, wrist_w, wrist_c),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (4,),
                "names": {"motors": state_names},
            },
            "action": {
                "dtype": "float32",
                "shape": (4,),
                "names": {"motors": action_names},
            },
        }
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError:
            from lerobot.datasets import LeRobotDataset

        kwargs = {
            "repo_id": self.cfg.repo_id,
            "root": self.root,
            "fps": int(round(self.cfg.fps)),
            "robot_type": self.cfg.robot_type,
            "features": features,
            "use_videos": self.cfg.use_videos,
            "image_writer_threads": self.cfg.image_writer_threads,
            "batch_encoding_size": 1,
            "metadata_buffer_size": 1,
            "vcodec": self.cfg.video_codec,
        }
        try:
            self.dataset = LeRobotDataset.create(**kwargs)
        except TypeError as exc:
            if "vcodec" not in str(exc):
                raise
            kwargs.pop("vcodec", None)
            self.dataset = LeRobotDataset.create(**kwargs)

    def _make_unused_root(self) -> Path:
        self.cfg.dataset_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = self.cfg.dataset_root / f"{self.cfg.dataset_name}_{stamp}"
        if not base.exists():
            return base
        suffix = 1
        while True:
            candidate = self.cfg.dataset_root / f"{self.cfg.dataset_name}_{stamp}_{suffix:02d}"
            if not candidate.exists():
                return candidate
            suffix += 1
