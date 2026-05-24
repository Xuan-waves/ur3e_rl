"""
关节空间阻抗控制示例

控制律：
    τ = M(q) · (q̈d + Kd(q̇d − q̇) + Kp(qd − q)) + C(q,q̇)q̇ + g(q)

其中：
    M(q)  : 关节空间质量矩阵
    C(q,q̇): 科氏力 + 向心力项
    g(q)  : 重力项
    Kp    : 关节刚度矩阵（弹簧增益）
    Kd    : 关节阻尼矩阵（阻尼增益）

由于 MuJoCo 执行器是位置 PD 伺服（ctrl = 期望关节角），
需将计算得到的力矩 τ 反解为等效位置命令：

    ctrl = q + (τ + kv × dq) / kp

其中 kp、kv 是 XML 中各关节的伺服增益（见下方 KP_SERVO / KV_SERVO）。

用法：
    python run_impedance_joint.py
    python run_impedance_joint.py --target 0 -1.2 1.2 -1.5 1.57 0
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

UR3eGraspEnv  = env_module.UR3eGraspEnv
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

# forcerange 限幅（Nm），防止 ctrl 计算结果产生超出物理范围的期望
FORCE_RANGE = np.array([54., 54., 28., 9., 9., 9.])


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


def run(env: UR3eGraspEnv, kin: UR3eKinematics, q_des: np.ndarray):
    """
    关节空间阻抗控制主循环。

    Parameters
    ----------
    env   : 已完成 reset() 的仿真环境
    kin   : UR3eKinematics 实例
    q_des : (6,) 目标关节角（rad）
    """
    # ── 阻抗增益（运行时可自由修改，无需改 XML）──────────────────────────────
    # Kp：弹簧刚度（值越大，对位置偏差越敏感）
    # Kd：阻尼（值越大，运动越平滑、超调越小）
    # 建议保持临界阻尼关系：Kd ≈ 2 × sqrt(Kp)
    #
    # 注意：腕部关节（wrist_1/2/3）的 forcerange = ±9 Nm，
    # 若 Kp 过大或重力项超过 9 Nm，会出现稳态误差。
    # 当前增益已考虑此限制，腕部 Kp 设为较小值。
    Kp = np.diag([200., 200., 100.,  50.,  50.,  50.])
    Kd = np.diag([ 28.,  28.,  20.,  14.,  14.,  14.])

    print("\n[关节空间阻抗控制] 已启动")
    print(f"  目标关节角 (deg): {np.round(np.degrees(q_des), 2)}")
    print(f"  Kp 对角: {np.diag(Kp)}")
    print(f"  Kd 对角: {np.diag(Kd)}")
    print("  关闭 MuJoCo 窗口或按 Ctrl+C 退出\n")

    gripper_ctrl = env.data.ctrl[6]  # 保留夹爪当前状态

    while env.viewer.is_running():
        for _ in range(10):  # 每次渲染执行 10 个物理步
            # 1. 读取当前状态
            q6  = env.data.qpos[:6].copy()
            dq6 = env.data.qvel[:6].copy()

            # 2. 计算关节空间阻抗力矩
            #    τ = M(q)(Kd(0−dq) + Kp(qd−q)) + C(q,q̇)q̇ + g(q)
            tau = kin.impedance_torque_joint_space(
                q6, dq6,
                qd=q_des,
                dqd=np.zeros(6),
                ddqd=np.zeros(6),
                Kp=Kp,
                Kd=Kd,
            )

            # 3. 力矩 → 等效 ctrl 位置命令
            ctrl_arm = torque_to_ctrl(q6, dq6, tau)
            ctrl_arm = np.clip(ctrl_arm, -6.2831, 6.2831)

            # 4. 组装完整 ctrl（手臂 + 夹爪）并执行
            ctrl = np.append(ctrl_arm, gripper_ctrl)
            env.step(ctrl)

        # 每帧打印关节误差与力矩饱和情况（调试用，可注释掉）
        q6  = env.data.qpos[:6].copy()
        dq6 = env.data.qvel[:6].copy()
        err = q_des - q6
        tau = kin.impedance_torque_joint_space(q6, dq6, qd=q_des, Kp=Kp, Kd=Kd)
        saturated = np.abs(tau) > FORCE_RANGE
        sat_str = "".join(["!" if s else "." for s in saturated])
        print(
            f"  误差(deg): {np.round(np.degrees(err), 1)}  "
            f"力矩(Nm): {np.round(tau, 1)}  "
            f"饱和[pan,lift,elbow,w1,w2,w3]: {sat_str}",
            end="\r",
        )


def main():
    parser = argparse.ArgumentParser(description="UR3e 关节空间阻抗控制")
    parser.add_argument(
        "--target", nargs=6, type=float,
        default=None,
        metavar=("q0", "q1", "q2", "q3", "q4", "q5"),
        help="目标关节角（rad），默认使用非奇异 home 位",
    )
    args = parser.parse_args()

    # 默认目标：可操作度较高、腕部重力负载小的构型
    # [0,-π/2,0,-π/2,0,0] 是腕部奇异点，避免使用
    # [0,-1.57,1.0,-1.0,1.57,0] 可操作度偏低（~0.012），腕部重力大
    # 推荐：大臂抬起、小臂展开，腕部接近水平，重力主要由大关节承担
    if args.target is not None:
        q_des = np.array(args.target)
    else:
        q_des = np.array([0.0, -1.0, 1.5, -2.0, -1.57, 0.0])

    # ── 环境初始化 ────────────────────────────────────────────────────────────
    env = UR3eGraspEnv()
    env.reset()

    # ── 运动到目标附近（用 MuJoCo 内置 PD 快速定位，再切入阻抗模式）────────
    print(f"[初始化] 驱动机械臂到目标角附近...")
    for _ in range(2000):   # 2000 步 = 4 秒仿真时间，给充裕稳定时间
        env.data.ctrl[:6] = q_des
        env.step()

    # ── 构建运动学对象（复用 env 中已加载的 pinocchio 模型）─────────────────
    kin = UR3eKinematics(env.model_roboplan, env.data_roboplan)

    # 打印初始状态
    q_init = env.data.qpos[:6].copy()
    T_init = kin.fk(q_init)
    manip  = kin.manipulability(q_init)
    print(f"[初始状态]")
    print(f"  关节角 (deg)   : {np.round(np.degrees(q_init), 2)}")
    print(f"  末端位置 (m)   : {np.round(T_init.translation, 4)}")
    print(f"  可操作度       : {manip:.4f}")
    # 打印各关节的重力力矩，方便判断是否超出 forcerange
    tau_grav = kin.gravity(q_init)
    print(f"  重力力矩 (Nm)  : {np.round(tau_grav, 2)}")
    print(f"  关节 forcerange: {FORCE_RANGE}")
    saturated_grav = np.abs(tau_grav) > FORCE_RANGE
    if np.any(saturated_grav):
        names = ["pan", "lift", "elbow", "wrist1", "wrist2", "wrist3"]
        snames = [names[i] for i in range(6) if saturated_grav[i]]
        print(f"  [警告] 以下关节的重力力矩已超出 forcerange，将有稳态误差: {snames}")
    if manip < 0.05:
        print(f"  [警告] 可操作度偏低（{manip:.4f}），建议换更好的目标构型")

    try:
        run(env, kin, q_des)
    except KeyboardInterrupt:
        print("\n用户中断，退出。")
    finally:
        env.close()


if __name__ == "__main__":
    main()
