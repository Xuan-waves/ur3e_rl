from __future__ import annotations

import sys
import time
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
        self._headset_rot = R.from_matrix(cfg.headset_orientation_to_world)
        self._collection_input_period = 1.0 / max(float(cfg.collection_input_hz), 1e-6)
        self._last_collection_input_time = 0.0
        self._collection_inputs = {
            "record_start": False,
            "record_stop": False,
            "rl_toggle": False,
            "cancel_record": False,
            "left_trigger": 0.0,
            "stop_collection": False,
            "left_grip": 0.0,
        }

    def read(self) -> dict:
        raw_pose = self._safe(lambda: self.xr.get_pose_by_name("right_controller"))
        grip = float(self._safe(lambda: self.xr.get_key_value_by_name("right_grip") or 0.0, 0.0))
        trigger = float(self._safe(lambda: self.xr.get_key_value_by_name("right_trigger") or 0.0, 0.0))
        a_btn = bool(self._safe(lambda: self.xr.get_button_state_by_name("A"), False))
        self._update_collection_inputs()

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
            "gripper": float(np.clip(trigger, 0.0, self.cfg.gripper_command_max)),
            "home": a_btn,
            **self._collection_inputs,
        }

    def _update_collection_inputs(self) -> None:
        stamp = time.monotonic()
        if stamp - self._last_collection_input_time < self._collection_input_period:
            return
        self._last_collection_input_time = stamp
        left_trigger = float(self._safe(lambda: self.xr.get_key_value_by_name("left_trigger") or 0.0, 0.0))
        left_grip = float(self._safe(lambda: self.xr.get_key_value_by_name("left_grip") or 0.0, 0.0))
        self._collection_inputs = {
            "record_start": bool(self._safe(lambda: self.xr.get_button_state_by_name("X"), False)),
            "record_stop": bool(self._safe(lambda: self.xr.get_button_state_by_name("B"), False)),
            "rl_toggle": bool(self._safe(lambda: self.xr.get_button_state_by_name("Y"), False)),
            "cancel_record": left_trigger >= 0.95,
            "left_trigger": float(np.clip(left_trigger, 0.0, 1.0)),
            "stop_collection": left_grip >= 0.95,
            "left_grip": float(np.clip(left_grip, 0.0, 1.0)),
        }

    @staticmethod
    def _safe(fn, default=None):
        try:
            return fn()
        except Exception:
            return default
