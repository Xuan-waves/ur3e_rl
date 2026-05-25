from __future__ import annotations

import json
import time
from typing import Any

import numpy as np


def now() -> float:
    return time.monotonic()


def dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"))


def loads(text: str) -> dict[str, Any]:
    return json.loads(text)


def as_vec(value: Any, size: int, default: float = 0.0) -> np.ndarray:
    arr = np.asarray(value if value is not None else [default] * size, dtype=float)
    if arr.shape != (size,):
        raise ValueError(f"Expected vector of length {size}, got {arr.shape}")
    return arr


REASON_CODES = {
    "tracking": 0,
    "no_target": 1,
    "stale_vr": 2,
    "home": 3,
    "disabled": 4,
    "anchored": 5,
    "hold": 6,
}
REASON_TEXT = {value: key for key, value in REASON_CODES.items()}


def reason_code(reason: str) -> float:
    return float(REASON_CODES.get(reason, REASON_CODES["hold"]))


def reason_text(code: float | int) -> str:
    return REASON_TEXT.get(int(code), "hold")


def make_vr_command(payload: dict[str, Any]) -> list[float]:
    pose = payload.get("pose")
    has_pose = pose is not None
    if has_pose:
        pose_vec = as_vec(pose, 7)
    else:
        pose_vec = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=float)
    return [
        float(payload.get("stamp", now())),
        1.0 if has_pose else 0.0,
        1.0 if bool(payload.get("enable", False)) else 0.0,
        float(np.clip(payload.get("gripper", 0.0), 0.0, 1.0)),
        1.0 if bool(payload.get("home", False)) else 0.0,
        *pose_vec.tolist(),
    ]


def parse_vr_command(data: Any) -> dict[str, Any]:
    arr = as_vec(data, 12)
    pose = arr[5:12].copy() if arr[1] > 0.5 else None
    return {
        "stamp": float(arr[0]),
        "pose": pose,
        "enable": bool(arr[2] > 0.5),
        "gripper": float(np.clip(arr[3], 0.0, 1.0)),
        "home": bool(arr[4] > 0.5),
    }


def make_robot_state(
    *,
    q: np.ndarray,
    tcp_pos: np.ndarray,
    tcp_quat: np.ndarray,
    gripper: float,
    servo_active: bool,
    homing: bool,
    target_tracking: bool,
    target_age: float,
) -> list[float]:
    return [
        now(),
        *as_vec(q, 6).tolist(),
        *as_vec(tcp_pos, 3).tolist(),
        *_norm_quat(tcp_quat).tolist(),
        float(np.clip(gripper, 0.0, 1.0)),
        1.0 if servo_active else 0.0,
        1.0 if homing else 0.0,
        1.0 if target_tracking else 0.0,
        float(target_age),
    ]


def parse_robot_state(data: Any) -> dict[str, Any]:
    arr = as_vec(data, 19)
    return {
        "stamp": float(arr[0]),
        "q": arr[1:7].copy(),
        "tcp_pos": arr[7:10].copy(),
        "tcp_quat": _norm_quat(arr[10:14]),
        "gripper": float(np.clip(arr[14], 0.0, 1.0)),
        "servo_active": bool(arr[15] > 0.5),
        "homing": bool(arr[16] > 0.5),
        "target_tracking": bool(arr[17] > 0.5),
        "target_age": float(arr[18]),
    }


def make_joint_target(
    *,
    tracking: bool,
    q: np.ndarray | None = None,
    gripper: float = 0.0,
    reason: str = "tracking",
    ok: bool = True,
    q_delta: float = 0.0,
) -> list[float]:
    q_vec = np.zeros(6, dtype=float) if q is None else as_vec(q, 6)
    return [
        now(),
        1.0 if tracking else 0.0,
        *q_vec.tolist(),
        float(np.clip(gripper, 0.0, 1.0)),
        reason_code(reason),
        1.0 if ok else 0.0,
        float(q_delta),
    ]


def parse_joint_target(data: Any) -> dict[str, Any]:
    arr = as_vec(data, 12)
    tracking = bool(arr[1] > 0.5)
    return {
        "stamp": float(arr[0]),
        "tracking": tracking,
        "q": arr[2:8].copy() if tracking else None,
        "gripper": float(np.clip(arr[8], 0.0, 1.0)),
        "reason": reason_text(arr[9]),
        "ok": bool(arr[10] > 0.5),
        "q_delta": float(arr[11]),
    }


def make_ik_target(pos: np.ndarray, quat: np.ndarray) -> list[float]:
    return [now(), *as_vec(pos, 3).tolist(), *_norm_quat(quat).tolist()]


def parse_ik_target(data: Any) -> dict[str, Any]:
    arr = as_vec(data, 8)
    return {"stamp": float(arr[0]), "pos": arr[1:4].copy(), "quat": _norm_quat(arr[4:8])}


def _norm_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return q / n
