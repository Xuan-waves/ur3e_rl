from __future__ import annotations

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

from .config import TeleopConfig


class RobotKinematics:
    def __init__(self, cfg: TeleopConfig):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(cfg.xml_path)
        self.data = mujoco.MjData(self.model)
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, cfg.ee_frame)
        if self.site_id < 0:
            raise ValueError(f"MuJoCo site not found: {cfg.ee_frame}")

    def forward(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.data.qpos[:6] = q
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        pos = self.data.site_xpos[self.site_id].copy()
        mat = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        return pos, R.from_matrix(mat).as_quat()


class MinkIkSolver:
    def __init__(self, cfg: TeleopConfig):
        import mink

        self.cfg = cfg
        self.mink = mink
        self.model = mujoco.MjModel.from_xml_path(cfg.xml_path)
        self.data = mujoco.MjData(self.model)
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, cfg.ee_frame)
        if self.site_id < 0:
            raise ValueError(f"MuJoCo site not found: {cfg.ee_frame}")
        self.configuration = self.mink.Configuration(self.model)
        self.ee_task = self.mink.FrameTask(
            frame_name=cfg.ee_frame,
            frame_type="site",
            position_cost=cfg.ik_position_cost,
            orientation_cost=cfg.ik_orientation_cost,
            lm_damping=cfg.ik_lm_damping,
        )
        self.posture_task = self.mink.PostureTask(model=self.model, cost=cfg.ik_posture_cost)
        self.damping_task = self.mink.DampingTask(model=self.model, cost=cfg.ik_damping_cost)

    def forward(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.data.qpos[:6] = q
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        pos = self.data.site_xpos[self.site_id].copy()
        mat = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        return pos, R.from_matrix(mat).as_quat()

    def solve(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        q_init: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, bool]:
        q = self.configuration.q.copy()
        q[:6] = q_init
        self.configuration.update(q)
        self.posture_task.set_target_from_configuration(self.configuration)
        target = self.mink.SE3.from_rotation_and_translation(
            self.mink.SO3.from_matrix(R.from_quat(target_quat).as_matrix()),
            target_pos,
        )
        self.ee_task.set_target(target)

        ok = True
        for _ in range(max(1, self.cfg.ik_iters)):
            try:
                vel = self.mink.solve_ik(
                    self.configuration,
                    [self.ee_task, self.posture_task, self.damping_task],
                    dt,
                    "daqp",
                    damping=self.cfg.ik_solve_damping,
                )
            except Exception:
                ok = False
                break
            self.configuration.integrate_inplace(vel, dt)
        return self.configuration.q[:6].copy(), ok
