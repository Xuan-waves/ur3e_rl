from __future__ import annotations

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

    @property
    def is_ready(self) -> bool:
        return self.dataset is not None

    def add_frame(self, frame: CollectedFrame) -> None:
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
        if self.dataset is None or self.episode_frame_count <= 0:
            return False
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
        if self.dataset is None or self.episode_frame_count <= 0:
            return False
        clear_buffer = getattr(self.dataset, "clear_episode_buffer", None)
        if callable(clear_buffer):
            clear_buffer()
        else:
            episode_buffer = getattr(self.dataset, "episode_buffer", None)
            if isinstance(episode_buffer, dict):
                for value in episode_buffer.values():
                    if hasattr(value, "clear"):
                        value.clear()
        self.episode_frame_count = 0
        return True

    def finalize(self) -> None:
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

    def _create_dataset(self, first_frame: CollectedFrame) -> None:
        front_h, front_w, front_c = first_frame.front_rgb.shape
        wrist_h, wrist_w, wrist_c = first_frame.wrist_rgb.shape
        state_names = [
            "tcp_x",
            "tcp_y",
            "tcp_z",
            "tcp_rx",
            "tcp_ry",
            "tcp_rz",
            "gripper",
            "RL_mark",
        ]
        action_names = [
            "target_tcp_x",
            "target_tcp_y",
            "target_tcp_z",
            "target_tcp_rx",
            "target_tcp_ry",
            "target_tcp_rz",
            "target_gripper",
            "RL_mark",
        ]
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
                "shape": (8,),
                "names": {"motors": state_names},
            },
            "action": {
                "dtype": "float32",
                "shape": (8,),
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

