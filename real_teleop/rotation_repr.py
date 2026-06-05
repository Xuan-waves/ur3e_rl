from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def continuous_rotvec_from_quat(quat: np.ndarray, previous: np.ndarray | None = None) -> np.ndarray:
    """Return a rotvec representation that is continuous with the previous sample.

    Rotation vectors have a branch cut around pi radians.  Near that cut, the
    same physical orientation may appear as either roughly +pi * axis or
    -pi * axis.  For learning, choose the equivalent representation closest to
    the previous frame.
    """
    rotvec = R.from_quat(_norm_quat(quat)).as_rotvec()
    return unwrap_rotvec(rotvec, previous)


def unwrap_rotvec(rotvec: np.ndarray, previous: np.ndarray | None = None) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=float)
    if previous is None:
        return rotvec.astype(np.float32)

    previous = np.asarray(previous, dtype=float)
    if previous.shape != (3,) or not np.all(np.isfinite(previous)):
        return rotvec.astype(np.float32)

    angle = float(np.linalg.norm(rotvec))
    candidates = [rotvec]
    if angle > 1e-9:
        axis = rotvec / angle
        candidates.append(rotvec - 2.0 * np.pi * axis)
        candidates.append(rotvec + 2.0 * np.pi * axis)

    best = min(candidates, key=lambda value: float(np.linalg.norm(value - previous)))
    return np.asarray(best, dtype=np.float32)


def _norm_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=float)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return quat / norm
