from __future__ import annotations

import numpy as np


class PoseFilter:
    def __init__(self, alpha_pos: float, alpha_rot: float):
        self.alpha_pos = alpha_pos
        self.alpha_rot = alpha_rot
        self._pos: np.ndarray | None = None
        self._quat: np.ndarray | None = None

    def reset(self) -> None:
        self._pos = None
        self._quat = None

    def __call__(self, pos: np.ndarray, quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._pos is None:
            self._pos = np.asarray(pos, dtype=float).copy()
            self._quat = _norm_quat(quat)
            return self._pos.copy(), self._quat.copy()
        self._pos = self.alpha_pos * pos + (1.0 - self.alpha_pos) * self._pos
        q = self.alpha_rot * _same_hemi(quat, self._quat) + (1.0 - self.alpha_rot) * self._quat
        self._quat = _norm_quat(q)
        return self._pos.copy(), self._quat.copy()


def _norm_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return q / n


def _same_hemi(q: np.ndarray, ref: np.ndarray) -> np.ndarray:
    q = _norm_quat(q)
    return -q if float(np.dot(q, ref)) < 0.0 else q

