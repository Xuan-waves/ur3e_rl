"""
笛卡尔空间（任务空间）阻抗控制示例

控制律：
    τ = Jᵀ [ Λ(q) · (ẍd + Kd(ẋd − ẋ) + Kp · err) + μ(q,q̇) + p(q) ]

其中：
    Λ(q)  = (J M⁻¹ Jᵀ)⁻¹    操作空间惯量矩阵
    μ(q,q̇)= Λ J M⁻¹ C q̇    操作空间科氏项
    p(q)  = Λ J M⁻¹ g        操作空间重力项
    J     : 末端执行器雅可比（6×6）
    err   = [e_p ; e_R]       位置误差（3） + 旋转误差（SO3 对数映射，3）

Kp 前 3 分量控制位置刚度（N/m），后 3 分量控制旋转刚度（N·m/rad）。
Kd 前 3 分量控制位置阻尼（N·s/m），后 3 分量控制旋转阻尼（N·m·s/rad）。

力矩反解为等效 ctrl（位置 PD 伺服）：
    ctrl = q + (τ + kv × dq) / kp

用法：
    # 末端保持在初始位姿（柔顺悬停）
    python run_impedance_cartesian.py

    # 末端跟踪一个相对于初始位置的偏移目标（dx dy dz，单位 m）
    python run_impedance_cartesian.py --offset 0.05 0.0 0.0
"""

import os
import sys
import argparse
import importlib
import numpy as np
import pinocchio

# ── 路径设置（与 run_scene.py 保持一致）──────────────────────────────────────
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

_PKG_NAME = os.path.basename(_THIS_DIR)

env_module = importlib.import_module(f"{_PKG_NAME}.env.ur3e_grasp_env")
kin_module  = importlib.import_module(f"{_PKG_NAME}.kinematics.ur3e_kinematics")

UR3eGraspEnv   = env_module.UR3eGraspEnv
UR3eKinematics = kin_module.UR3eKinematics

# ── XML 中各关节伺服增益（与 ur3e_ag95.xml 保持一致）────────────────────────
#   关节顺序：shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
#
#   actuator 类别       kp      kv      forcerange
#   size2 (pan/lift)   1000    200     ±54 Nm
#   size1_limited (el)  500    100     ±28 Nm
#   size0 (wrist×3)    250     50      ±9  Nm
KP_SERVO = np.array([1000., 1000., 500., 250., 250., 250.])
KV_SERVO = np.array([ 200.,  200., 100.,  50.,  50.,  50.])

# forcerange 限幅（Nm）
FORCE_RANGE = np.array([54., 54., 28., 9., 9., 9.])

# 可操作度阈值：低于此值视为接近奇异，跳过力矩计算
MANIP_THRESHOLD = 0.01


def torque_to_ctrl(q6: np.ndarray, dq6: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """
    将期望关节力矩 τ 转换为等效 ctrl 位置命令。

    推导：
        F_actuator = kp × (ctrl − q) − kv × dq = τ
        →  ctrl = q + (τ + kv × dq) / kp

    先将 τ 限幅到各关节 forcerange，再计算 ctrl。
    """
    tau_clipped = np.clip(tau, -FORCE_RANGE, FORCE_RANGE)
    return q6 + (tau_clipped + KV_SERVO * dq6) / KP_SERVO


def run(env: UR3eGraspEnv, kin: UR3eKinematics, T_des: pinocchio.SE3):
    """
    笛卡尔空间阻抗控制主循环。

    Parameters
    ----------
    env   : 已完成 reset() 的仿真环境
    kin   : UR3eKinematics 实例
    T_des : 目标末端位姿 pinocchio.SE3
    """
    # ── 阻抗增益 ───────────────────────────────────────────────────────────────
    # Kp : 笛卡尔刚度
    #   前 3 分量 = 位置刚度 (N/m)；后 3 分量 = 旋转刚度 (N·m/rad)
    #   位置刚度通常比旋转刚度大一个量级
    # Kd : 笛卡尔阻尼
    #   建议临界阻尼：Kd ≈ 2 × sqrt(Kp)（各轴独立）
    Kp = np.diag([300., 300., 300., 30., 30., 30.])
    Kd = np.diag([ 35.,  35.,  35.,  11., 11., 11.])

    print("\n[笛卡尔阻抗控制] 已启动")
    print(f"  目标末端位置 (m)  : {np.round(T_des.translation, 4)}")
    print(f"  目标旋转矩阵      :\n{np.round(T_des.rotation, 3)}")
    print(f"  Kp 位置/旋转      : {np.diag(Kp)[:3]} / {np.diag(Kp)[3:]}")
    print(f"  Kd 位置/旋转      : {np.diag(Kd)[:3]} / {np.diag(Kd)[3:]}")
    print("  关闭 MuJoCo 窗口或按 Ctrl+C 退出\n")

    gripper_ctrl = env.data.ctrl[6]  # 保留夹爪当前状态

    while env.viewer.is_running():
        for _ in range(10):  # 每次渲染执行 10 个物理步
            # 1. 读取当前状态
            q6  = env.data.qpos[:6].copy()
            dq6 = env.data.qvel[:6].copy()

            # 2. 奇异性检查
            manip = kin.manipulability(q6)
            if manip < MANIP_THRESHOLD:
                # 接近奇异：仅做重力补偿，等待人工干预
                tau_grav = kin.gravity_compensation(q6)
                ctrl_arm = torque_to_ctrl(q6, dq6, tau_grav)
                ctrl_arm = np.clip(ctrl_arm, -6.2831, 6.2831)
                env.step(np.append(ctrl_arm, gripper_ctrl))
                print(f"  [警告] 接近奇异（manip={manip:.4f}），切换为重力补偿", end="\r")
                continue

            # 3. 计算笛卡尔阻抗力矩
            #    τ = Jᵀ [ Λ(a_des + Kd·Δẋ + Kp·err) + μ + p ]
            tau = kin.impedance_torque_task_space(
                q6, dq6,
                T_des=T_des,
                v_des=np.zeros(6),
                a_des=np.zeros(6),
                Kp=Kp,
                Kd=Kd,
            )

            # 4. 力矩 → 等效 ctrl 位置命令
            ctrl_arm = torque_to_ctrl(q6, dq6, tau)
            ctrl_arm = np.clip(ctrl_arm, -6.2831, 6.2831)

            # 5. 组装完整 ctrl 并执行
            ctrl = np.append(ctrl_arm, gripper_ctrl)
            env.step(ctrl)

        # 每帧打印末端位置误差（调试用，可注释掉）
        q6      = env.data.qpos[:6].copy()
        T_cur   = kin.fk(q6)
        pos_err = np.linalg.norm(T_des.translation - T_cur.translation)
        R_err   = T_des.rotation @ T_cur.rotation.T
        rot_err = np.linalg.norm(pinocchio.log3(R_err))
        print(
            f"  末端位置误差: {pos_err*1000:.2f} mm  "
            f"旋转误差: {np.degrees(rot_err):.2f} deg  "
            f"可操作度: {kin.manipulability(q6):.4f}",
            end="\r",
        )


def main():
    parser = argparse.ArgumentParser(description="UR3e 笛卡尔空间阻抗控制")
    parser.add_argument(
        "--offset", nargs=3, type=float,
        default=[0.0, 0.0, 0.0],
        metavar=("dx", "dy", "dz"),
        help="相对于初始末端位置的目标偏移（m），默认在原地柔顺悬停",
    )
    args = parser.parse_args()
    offset = np.array(args.offset)

    # ── 环境初始化 ────────────────────────────────────────────────────────────
    env = UR3eGraspEnv()
    env.reset()

    # 使用非奇异构型作为初始状态（home [0,-π/2,0,-π/2,0,0] 是腕部奇异点）
    q_start = np.array([0.0, -1.57, 1.0, -1.0, 1.57, 0.0])

    print(f"[初始化] 驱动机械臂到非奇异初始构型...")
    for _ in range(1000):
        env.data.ctrl[:6] = q_start
        env.step()

    # ── 构建运动学对象 ─────────────────────────────────────────────────────────
    kin = UR3eKinematics(env.model_roboplan, env.data_roboplan)

    # 读取稳定后的当前状态
    q_init  = env.data.qpos[:6].copy()
    T_init  = kin.fk(q_init)
    manip   = kin.manipulability(q_init)

    print(f"[初始状态]")
    print(f"  关节角 (deg)   : {np.round(np.degrees(q_init), 2)}")
    print(f"  末端位置 (m)   : {np.round(T_init.translation, 4)}")
    print(f"  可操作度       : {manip:.4f}")
    if manip < MANIP_THRESHOLD:
        print("  [错误] 初始构型接近奇异，无法启动笛卡尔阻抗控制！")
        env.close()
        return

    # 以初始末端位姿 + 用户偏移作为目标
    p_des  = T_init.translation + offset
    T_des  = pinocchio.SE3(T_init.rotation.copy(), p_des)

    if np.any(offset != 0):
        print(f"[目标] 在初始末端位置基础上偏移 {offset} m")
        # 验证目标可达性（用差分 IK 检查）
        q_check = kin.ik_differential(T_des, init_q6=q_init)
        if q_check is None:
            print("  [警告] 目标位姿可能不可达，控制器仍将尝试趋近。")

    try:
        run(env, kin, T_des)
    except KeyboardInterrupt:
        print("\n用户中断，退出。")
    finally:
        env.close()


if __name__ == "__main__":
    main()
