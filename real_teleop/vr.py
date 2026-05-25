from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from .config import TeleopConfig


class XrobotVrReader:
    def __init__(self, cfg: TeleopConfig):
        self.cfg = cfg
        xrobot_root = Path(__file__).resolve().parents[1] / "Xrobot_tool"
        if str(xrobot_root) not in sys.path:
            sys.path.insert(0, str(xrobot_root))
        from xrobotoolkit_teleop.common.xr_client import XrClient

        self.xr = XrClient()
        self._headset_rot = R.from_matrix(cfg.headset_to_world)

    def read(self) -> dict:
        raw_pose = self._safe(lambda: self.xr.get_pose_by_name("right_controller"))
        grip = float(self._safe(lambda: self.xr.get_key_value_by_name("right_grip") or 0.0, 0.0))
        trigger = float(self._safe(lambda: self.xr.get_key_value_by_name("right_trigger") or 0.0, 0.0))
        a_btn = bool(self._safe(lambda: self.xr.get_button_state_by_name("A"), False))

        pose = None
        if raw_pose is not None and len(raw_pose) >= 7:
            q = np.asarray(raw_pose[3:7], dtype=float)
            if np.linalg.norm(q) > 1e-6:
                pos = self.cfg.headset_to_world @ np.asarray(raw_pose[:3], dtype=float)
                quat = (self._headset_rot * R.from_quat(q) * self._headset_rot.inv()).as_quat()
                pose = np.concatenate([pos, quat]).tolist()

        return {
            "pose": pose,
            "enable": grip > self.cfg.enable_threshold,
            "enable_value": float(np.clip(grip, 0.0, 1.0)),
            "gripper": float(np.clip(trigger, 0.0, 1.0)),
            "home": a_btn,
        }

    @staticmethod
    def _safe(fn, default=None):
        try:
            return fn()
        except Exception:
            return default

