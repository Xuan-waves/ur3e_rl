"""
UR3e 运动学与动力学工具类

涵盖：
  - 正运动学（FK）
  - 逆运动学（IK）：差分法 / pyroboplan 带碰撞版
  - 雅可比矩阵及其伪逆
  - 关节空间动力学：质量矩阵、重力项、科氏力项
  - 操作空间动力学矩阵（阻抗控制核心）
  - 关节空间 / 任务空间阻抗控制力矩
  - 辅助工具：可操作度、条件数、重力补偿

依赖：pinocchio, numpy
用法示例见文件末尾 __main__ 块。
"""

import numpy as np
import pinocchio


# UR3e 六个关节的名称（顺序与 MuJoCo ctrl[0:6] 一致）
ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


class UR3eKinematics:
    """
    UR3e + AG95 运动学与动力学工具类（基于 pinocchio）

    坐标约定
    --------
    所有输出均在 pinocchio 世界系（与 URDF 根节点对齐）。
    MuJoCo 的 data.qpos[:6] 与 pinocchio 关节角顺序一致。

    Parameters
    ----------
    model_pin : pinocchio.Model
        由 load_models() 返回的 pinocchio 模型。
    data_pin  : pinocchio.Data
        由 model_pin.createData() 创建的数据对象。
    eef_frame : str
        末端执行器 frame 名称，默认 "grasp_center"。
    """

    def __init__(
        self,
        model_pin: pinocchio.Model,
        data_pin: pinocchio.Data,
        eef_frame: str = "grasp_center",
    ):
        self.model = model_pin
        self.data  = data_pin
        self.eef_frame    = eef_frame
        self.eef_frame_id = model_pin.getFrameId(eef_frame)

        # 构建手臂 6 关节在 pinocchio q / v 向量中的索引
        self._arm_qidx = []
        self._arm_vidx = []
        for name in ARM_JOINT_NAMES:
            jid = model_pin.getJointId(name)
            self._arm_qidx.append(model_pin.joints[jid].idx_q)
            self._arm_vidx.append(model_pin.joints[jid].idx_v)
        self._arm_qidx = np.array(self._arm_qidx, dtype=int)
        self._arm_vidx = np.array(self._arm_vidx, dtype=int)

        self._q_neutral = pinocchio.neutral(model_pin)

    # ------------------------------------------------------------------ #
    #  私有工具                                                             #
    # ------------------------------------------------------------------ #

    def _full_q(self, q6: np.ndarray) -> np.ndarray:
        """将 6 关节角填入完整 pinocchio q 向量，其余关节保持 neutral。"""
        q = self._q_neutral.copy()
        q[self._arm_qidx] = q6
        return q

    def _full_v(self, dq6: np.ndarray) -> np.ndarray:
        """将 6 关节速度填入完整 pinocchio v 向量。"""
        v = np.zeros(self.model.nv)
        v[self._arm_vidx] = dq6
        return v

    # ------------------------------------------------------------------ #
    #  正运动学（FK）                                                       #
    # ------------------------------------------------------------------ #

    def fk(self, q6: np.ndarray) -> pinocchio.SE3:
        """
        正运动学：6 关节角 → 末端位姿

        Parameters
        ----------
        q6 : (6,) 关节角，单位 rad

        Returns
        -------
        T : pinocchio.SE3
            T.translation : (3,) 末端位置（米）
            T.rotation    : (3,3) 旋转矩阵
        """
        q = self._full_q(q6)
        pinocchio.forwardKinematics(self.model, self.data, q)
        pinocchio.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.eef_frame_id].copy()

    def fk_pos(self, q6: np.ndarray) -> np.ndarray:
        """正运动学，仅返回末端位置 (3,)。"""
        return self.fk(q6).translation.copy()

    def fk_all_joints(self, q6: np.ndarray) -> dict:
        """
        返回所有关节 frame 的位姿，便于可视化或调试。

        Returns
        -------
        dict[frame_name -> pinocchio.SE3]
        """
        q = self._full_q(q6)
        pinocchio.forwardKinematics(self.model, self.data, q)
        pinocchio.updateFramePlacements(self.model, self.data)
        return {
            f.name: self.data.oMf[self.model.getFrameId(f.name)].copy()
            for f in self.model.frames
        }

    # ------------------------------------------------------------------ #
    #  雅可比矩阵                                                           #
    # ------------------------------------------------------------------ #

    def jacobian(
        self,
        q6: np.ndarray,
        local_frame: bool = False,
    ) -> np.ndarray:
        """
        末端执行器几何雅可比矩阵（6×6）

        Parameters
        ----------
        q6          : (6,) 关节角
        local_frame : True  → LOCAL 系（随末端旋转，适合力控）
                      False → LOCAL_WORLD_ALIGNED（世界对齐，常用）

        Returns
        -------
        J : (6, 6)  上 3 行为平移速度，下 3 行为角速度
        """
        q = self._full_q(q6)
        pinocchio.computeJointJacobians(self.model, self.data, q)
        pinocchio.updateFramePlacements(self.model, self.data)
        ref = (
            pinocchio.ReferenceFrame.LOCAL
            if local_frame
            else pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        J_full = pinocchio.getFrameJacobian(
            self.model, self.data, self.eef_frame_id, ref
        )
        return J_full[:, self._arm_vidx]   # (6, 6)

    def jacobian_linear(self, q6: np.ndarray) -> np.ndarray:
        """平移部分雅可比（3×6）：v_eef = J_lin @ dq6。"""
        return self.jacobian(q6)[:3, :]

    def jacobian_pinv(
        self, q6: np.ndarray, damping: float = 1e-6
    ) -> np.ndarray:
        """
        阻尼最小二乘伪逆 J†（6×6）

        J† = Jᵀ (J Jᵀ + λ²I)⁻¹

        接近奇异时 damping 越大越稳定，但精度越低。
        """
        J = self.jacobian(q6)
        return J.T @ np.linalg.inv(J @ J.T + damping * np.eye(6))

    # ------------------------------------------------------------------ #
    #  动力学量                                                             #
    # ------------------------------------------------------------------ #

    def mass_matrix(self, q6: np.ndarray) -> np.ndarray:
        """
        关节空间质量矩阵 M(q)，shape=(6,6)

        用于阻抗控制中的惯量补偿和操作空间变换：
          τ_inertia = M(q) · q̈
        """
        q = self._full_q(q6)
        M_full = pinocchio.crba(self.model, self.data, q)
        return M_full[np.ix_(self._arm_vidx, self._arm_vidx)].copy()

    def gravity(self, q6: np.ndarray) -> np.ndarray:
        """
        重力矩向量 g(q)，shape=(6,)

        补偿重力所需的关节力矩，可直接加到 ctrl 上实现重力补偿。
        """
        q = self._full_q(q6)
        g_full = pinocchio.computeGeneralizedGravity(self.model, self.data, q)
        return g_full[self._arm_vidx].copy()

    def coriolis_centrifugal(
        self, q6: np.ndarray, dq6: np.ndarray
    ) -> np.ndarray:
        """
        科氏力 + 向心力项 C(q,q̇)q̇，shape=(6,)

        推导方式：RNEA(q, q̇, 0) - g(q)
          RNEA 在 q̈=0 时给出：τ = C(q,q̇)q̇ + g(q)
          → C(q,q̇)q̇ = τ - g
        """
        q    = self._full_q(q6)
        v    = self._full_v(dq6)
        a    = np.zeros(self.model.nv)
        tau  = pinocchio.rnea(self.model, self.data, q, v, a)
        g    = pinocchio.computeGeneralizedGravity(self.model, self.data, q)
        return (tau - g)[self._arm_vidx].copy()

    def operational_space_matrices(
        self, q6: np.ndarray, dq6: np.ndarray
    ) -> tuple:
        """
        操作空间动力学矩阵（阻抗控制核心）

        推导
        ----
          Λ    = (J M⁻¹ Jᵀ)⁻¹          操作空间惯量矩阵
          J̄    = M⁻¹ Jᵀ Λ              动力学一致伪逆（满足 J̄ᵀ M = Jᵀ）
          μ    = Λ J M⁻¹ C q̇           操作空间科氏力项
          p    = Λ J M⁻¹ g             操作空间重力项

        阻抗控制力 F_task → 关节力矩：
          τ = Jᵀ (Λ a_des + μ + p)

        Returns
        -------
        Lambda : (6,6)
        mu     : (6,)
        p      : (6,)
        J      : (6,6)  雅可比（一并返回，避免重复计算）
        J_bar  : (6,6)  动力学一致伪逆
        """
        J = self.jacobian(q6)
        M = self.mass_matrix(q6)
        C = self.coriolis_centrifugal(q6, dq6)
        g = self.gravity(q6)

        M_inv  = np.linalg.inv(M)
        Lambda = np.linalg.inv(J @ M_inv @ J.T)
        J_bar  = M_inv @ J.T @ Lambda

        mu = Lambda @ J @ M_inv @ C   # (6,)
        p  = Lambda @ J @ M_inv @ g   # (6,)

        return Lambda, mu, p, J, J_bar

    # ------------------------------------------------------------------ #
    #  阻抗控制力矩                                                         #
    # ------------------------------------------------------------------ #

    def impedance_torque_joint_space(
        self,
        q6:   np.ndarray,
        dq6:  np.ndarray,
        qd:   np.ndarray,
        dqd:  np.ndarray = None,
        ddqd: np.ndarray = None,
        Kp:   np.ndarray = None,
        Kd:   np.ndarray = None,
    ) -> np.ndarray:
        """
        关节空间阻抗控制力矩

        τ = M(q)·(q̈d + Kd(q̇d−q̇) + Kp(qd−q)) + C(q,q̇)q̇ + g(q)

        Parameters
        ----------
        q6, dq6       : 当前关节角 (rad) 与速度 (rad/s)
        qd, dqd, ddqd : 目标角度 / 速度 / 加速度（后两者默认零）
        Kp, Kd        : (6,6) 增益矩阵，None 时使用内置默认值

        Returns
        -------
        tau : (6,) 关节力矩 (N·m)
        """
        if dqd  is None: dqd  = np.zeros(6)
        if ddqd is None: ddqd = np.zeros(6)
        if Kp   is None: Kp   = np.diag([100., 100., 100., 50.,  50.,  50.])
        if Kd   is None: Kd   = np.diag([20.,   20.,  20., 10.,  10.,  10.])

        M = self.mass_matrix(q6)
        C = self.coriolis_centrifugal(q6, dq6)
        g = self.gravity(q6)

        acc_des = ddqd + Kd @ (dqd - dq6) + Kp @ (qd - q6)
        return M @ acc_des + C + g

    def impedance_torque_task_space(
        self,
        q6:    np.ndarray,
        dq6:   np.ndarray,
        T_des: pinocchio.SE3,
        v_des: np.ndarray = None,
        a_des: np.ndarray = None,
        Kp:    np.ndarray = None,
        Kd:    np.ndarray = None,
    ) -> np.ndarray:
        """
        任务空间（笛卡尔）阻抗控制力矩

        τ = Jᵀ [Λ(ẍd + Kd(ẋd−ẋ) + Kp·err) + μ + p]

        误差定义
        --------
        位置误差：e_p = p_des − p_cur  (3,)
        旋转误差：e_R = log(R_des R_cur^T)  (3,)  —— 李群对数映射

        Parameters
        ----------
        q6, dq6 : 当前关节角与速度
        T_des   : 目标末端位姿 pinocchio.SE3
        v_des   : (6,) 目标末端速度，None→零
        a_des   : (6,) 目标末端加速度，None→零
        Kp, Kd  : (6,6) 任务空间增益矩阵

        Returns
        -------
        tau : (6,) 关节力矩 (N·m)
        """
        if v_des is None: v_des = np.zeros(6)
        if a_des is None: a_des = np.zeros(6)
        if Kp    is None: Kp    = np.diag([200., 200., 200., 20., 20., 20.])
        if Kd    is None: Kd    = np.diag([40.,   40.,  40.,  4.,  4.,  4.])

        T_cur = self.fk(q6)
        Lambda, mu, p, J, _ = self.operational_space_matrices(q6, dq6)

        # 位置误差
        e_p = T_des.translation - T_cur.translation

        # 旋转误差（SO3 对数映射）
        R_err = T_des.rotation @ T_cur.rotation.T
        e_R   = pinocchio.log3(R_err)

        err   = np.concatenate([e_p, e_R])      # (6,)
        v_cur = J @ dq6                          # (6,)

        a_cmd = a_des + Kd @ (v_des - v_cur) + Kp @ err
        F     = Lambda @ a_cmd + mu + p

        return J.T @ F

    # ------------------------------------------------------------------ #
    #  逆运动学（IK）                                                       #
    # ------------------------------------------------------------------ #

    def ik_differential(
        self,
        target:    pinocchio.SE3,
        init_q6:   np.ndarray = None,
        max_iters: int   = 500,
        tol:       float = 1e-4,
        step_size: float = 0.5,
        damping:   float = 1e-4,
    ) -> np.ndarray:
        """
        差分 IK（数值迭代，不做碰撞检测）

        适合快速求解或作为其他求解器的初始猜测。

        Parameters
        ----------
        target    : 目标末端位姿 pinocchio.SE3
        init_q6   : (6,) 初始关节角，None 时从全零开始
        max_iters : 最大迭代次数
        tol       : 收敛阈值（误差 L2 范数）
        step_size : 每步更新比例（0~1，过大可能振荡）
        damping   : 阻尼系数，防止雅可比奇异

        Returns
        -------
        q6_sol : (6,) 关节角解；收敛失败返回 None
        """
        q6 = np.zeros(6) if init_q6 is None else init_q6.copy()

        for _ in range(max_iters):
            T_cur = self.fk(q6)

            e_p  = target.translation - T_cur.translation
            R_err = target.rotation @ T_cur.rotation.T
            e_R  = pinocchio.log3(R_err)
            err  = np.concatenate([e_p, e_R])

            if np.linalg.norm(err) < tol:
                return q6

            J     = self.jacobian(q6)
            J_inv = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(6))
            q6    = q6 + step_size * (J_inv @ err)

        # 最终判断
        T_final = self.fk(q6)
        if np.linalg.norm(target.translation - T_final.translation) < tol * 10:
            return q6
        return None

    def ik_pyroboplan(
        self,
        env,
        target:  pinocchio.SE3,
        init_q6: np.ndarray = None,
    ) -> np.ndarray:
        """
        使用 pyroboplan DifferentialIk 求解（带碰撞检测，推荐使用）

        Parameters
        ----------
        env    : UR3eGraspEnv 实例（需已完成 reset()）
        target : 目标末端位姿 pinocchio.SE3

        Returns
        -------
        q6_sol : (6,) 关节角解；失败返回 None
        """
        if init_q6 is None:
            init_q6 = np.zeros(6)

        init_state = self._q_neutral.copy()
        init_state[self._arm_qidx] = init_q6

        q_sol = env.ik.solve(
            env.target_frame,
            target,
            init_state=init_state.copy(),
            verbose=False,
        )
        if q_sol is None:
            return None
        return q_sol[self._arm_qidx].copy()

    # ------------------------------------------------------------------ #
    #  辅助工具                                                             #
    # ------------------------------------------------------------------ #

    def eef_velocity(self, q6: np.ndarray, dq6: np.ndarray) -> np.ndarray:
        """末端执行器速度 ẋ = J q̇，shape=(6,)。"""
        return self.jacobian(q6) @ dq6

    def joint_torques_from_wrench(
        self, q6: np.ndarray, wrench: np.ndarray
    ) -> np.ndarray:
        """
        从末端力旋量映射到关节力矩（静力学）

        τ = Jᵀ F，不含动力学补偿。

        Parameters
        ----------
        wrench : (6,) [fx, fy, fz, tx, ty, tz]
        """
        return self.jacobian(q6).T @ wrench

    def gravity_compensation(self, q6: np.ndarray) -> np.ndarray:
        """纯重力补偿力矩 g(q)，shape=(6,)。直接叠加到 ctrl 即可悬停。"""
        return self.gravity(q6)

    def manipulability(self, q6: np.ndarray) -> float:
        """
        Yoshikawa 可操作度 w = sqrt(det(J Jᵀ))

        接近 0 表示当前构型趋近奇异，运动能力退化。
        """
        J = self.jacobian(q6)
        return float(np.sqrt(max(0.0, np.linalg.det(J @ J.T))))

    def condition_number(self, q6: np.ndarray) -> float:
        """
        雅可比矩阵条件数 σ_max / σ_min

        越大越接近奇异，通常 > 100 时需要注意。
        """
        s = np.linalg.svd(self.jacobian(q6), compute_uv=False)
        return float(s[0] / s[-1]) if s[-1] > 1e-10 else np.inf

    def singular_values(self, q6: np.ndarray) -> np.ndarray:
        """雅可比奇异值（从大到小），shape=(6,)。用于分析各方向运动能力。"""
        return np.linalg.svd(self.jacobian(q6), compute_uv=False)


# --------------------------------------------------------------------------- #
#  便捷构造函数                                                                 #
# --------------------------------------------------------------------------- #

def build_kinematics(urdf_path: str, srdf_path: str = None) -> "UR3eKinematics":
    """
    从 URDF 文件直接构建 UR3eKinematics 实例。

    Parameters
    ----------
    urdf_path : str  URDF 文件路径
    srdf_path : str  SRDF 文件路径（可选，用于排除自碰撞对）

    Returns
    -------
    kin : UR3eKinematics

    Example
    -------
    >>> from mujoco_scene.kinematics.ur3e_kinematics import build_kinematics
    >>> kin = build_kinematics("robot_description/urdf/ur3e_ag95.urdf")
    >>> T = kin.fk([0, -1.57, 0, -1.57, 0, 0])
    >>> print(T.translation)
    """
    import os
    from mujoco_scene.path_plan.set_model import (
        load_models,
        add_self_collisions,
        add_object_collisions,
    )

    model, collision_model, visual_model = load_models(urdf_path)
    if srdf_path and os.path.exists(srdf_path):
        add_self_collisions(model, collision_model, srdf_path)

    data = model.createData()
    return UR3eKinematics(model, data)


# --------------------------------------------------------------------------- #
#  用法示例                                                                     #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import os

    _THIS = os.path.dirname(os.path.abspath(__file__))
    _ROOT = os.path.dirname(_THIS)
    urdf = os.path.join(_ROOT, "robot_description", "urdf", "ur3e_ag95.urdf")
    srdf = os.path.join(_ROOT, "robot_description", "srdf", "ur3e_ag95.srdf")

    kin = build_kinematics(urdf, srdf)

    # UR3e home 位 [0,-π/2,0,-π/2,0,0] 是腕部奇异点（wrist_2=0 使 wrist_1/wrist_3 轴共线）。
    # FK 可用，但 Jacobian 的 rank=3，manipulability≈0，IK 会退化。
    # 演示 Jacobian/IK 时务必使用非奇异构型。
    q0   = np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0])  # FK only
    q_ns = np.array([0.0, -1.57, 1.0, -1.0, 1.57, 0.0])             # non-singular
    dq0  = np.zeros(6)

    # --- 正解 ---
    T    = kin.fk(q0)
    T_ns = kin.fk(q_ns)
    print("=== FK (home) ===")
    print(f"  position : {T.translation}")
    print("=== FK (non-singular) ===")
    print(f"  position : {T_ns.translation}")

    # --- 可操作度（用非奇异构型）---
    print(f"\n  manipulability : {kin.manipulability(q_ns):.4f}")
    print(f"  condition num  : {kin.condition_number(q_ns):.2f}")
    print(f"  singular values: {kin.singular_values(q_ns)}")

    # --- 动力学（用非奇异构型）---
    print("\n=== Dynamics (non-singular) ===")
    print(f"  gravity  (Nm) : {np.round(kin.gravity(q_ns), 3)}")
    print(f"  coriolis (0v) : {np.round(kin.coriolis_centrifugal(q_ns, dq0), 4)}")
    print(f"  mass matrix   :\n{np.round(kin.mass_matrix(q_ns), 4)}")

    # --- 差分 IK（从非奇异构型出发）---
    T_target = pinocchio.SE3(T_ns.rotation.copy(), T_ns.translation + np.array([0.05, 0, 0]))
    q_sol = kin.ik_differential(T_target, init_q6=q_ns)
    if q_sol is not None:
        print(f"\n=== IK (differential) ===")
        print(f"  solution : {np.round(q_sol, 4)}")
        T_check = kin.fk(q_sol)
        print(f"  FK error : {np.linalg.norm(T_check.translation - T_target.translation)*1000:.3f} mm")
    else:
        print("\nIK failed.")
