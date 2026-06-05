from __future__ import annotations

import numpy as np

from .config import TeleopConfig


class SafetyLimiter:
    def __init__(self, cfg: TeleopConfig):
        self.cfg = cfg

    def clamp_workspace(self, pos: np.ndarray) -> np.ndarray:
        return np.clip(pos, self.cfg.workspace_min, self.cfg.workspace_max)

    def clamp_impedance_workspace(self, pos: np.ndarray) -> np.ndarray:
        return np.clip(pos, self.cfg.impedance_workspace_min, self.cfg.impedance_workspace_max)

    def check_workspace(self, pos: np.ndarray) -> bool:
        return bool(np.all(pos >= self.cfg.workspace_min) and np.all(pos <= self.cfg.workspace_max))

    def clamp_joints(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self.cfg.joint_limits[:, 0], self.cfg.joint_limits[:, 1])

    def check_joints(self, q: np.ndarray) -> bool:
        return bool(
            q.shape == (6,)
            and np.all(np.isfinite(q))
            and np.all(q >= self.cfg.joint_limits[:, 0])
            and np.all(q <= self.cfg.joint_limits[:, 1])
        )

    def limit_step(
        self,
        current_q: np.ndarray,
        target_q: np.ndarray,
        dt: float,
        *,
        max_joint_step: float | None = None,
        max_joint_speed: float | None = None,
    ) -> np.ndarray:
        step_limit = self.cfg.max_joint_step if max_joint_step is None else float(max_joint_step)
        speed_limit = self.cfg.max_joint_speed if max_joint_speed is None else float(max_joint_speed)
        max_step = min(step_limit, speed_limit * dt)
        delta = np.clip(target_q - current_q, -max_step, max_step)
        return self.clamp_joints(current_q + delta)
