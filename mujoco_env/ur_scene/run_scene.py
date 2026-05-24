"""
MuJoCo 场景启动入口脚本

用法：
    # 基本启动（无相机窗口）
    python run_scene.py

    # 启动并显示双相机实时画面
    python run_scene.py --cameras

    # 测试运动学模块
    python run_scene.py --kin-test

依赖库：
    pip install mujoco pinocchio pyroboplan coal opencv-python glfw
"""

import os
import sys
import argparse
import pinocchio
import numpy as np
_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

_PKG_NAME = os.path.basename(_THIS_DIR)

import importlib
env_module = importlib.import_module(f"{_PKG_NAME}.env.ur3e_grasp_env")
UR3eGraspEnv = env_module.UR3eGraspEnv


def run_kin_test():
    """单独测试运动学模块（不启动仿真窗口）。"""
    kin_module = importlib.import_module(f"{_PKG_NAME}.kinematics.ur3e_kinematics")
    build_kinematics = kin_module.build_kinematics
    urdf = os.path.join(_THIS_DIR, "robot_description", "urdf", "ur3e_ag95.urdf")
    srdf = os.path.join(_THIS_DIR, "robot_description", "srdf", "ur3e_ag95.srdf")
    kin  = build_kinematics(urdf, srdf)
    q0   = np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0])  # 仅供 FK 演示
    q_ns = np.array([0.0, -1.57, 1.0, -1.0, 1.57, 0.0])             # 非奇异构型
    dq0  = np.zeros(6)
    print("=" * 55)
    print("1. 正运动学（FK）—— home 构型")
    T = kin.fk(q0)
    print(f"   末端位置 : {np.round(T.translation, 4)}")
    print(f"   旋转矩阵 :\n{np.round(T.rotation, 4)}")
    print(f"   ※ home 位是腕部奇异点（wrist_2=0），manipulability≈0")

    print("\n2. 可操作度 —— 非奇异构型 q={q_ns}")
    T_ns = kin.fk(q_ns)
    print(f"   末端位置    : {np.round(T_ns.translation, 4)}")
    print(f"   manipulability : {kin.manipulability(q_ns):.4f}")
    print(f"   condition num  : {kin.condition_number(q_ns):.2f}")
    sv = kin.singular_values(q_ns)
    print(f"   singular values: {np.round(sv, 4)}")

    print("\n3. 动力学量 —— 非奇异构型")
    print(f"   重力补偿力矩 (Nm): {np.round(kin.gravity(q_ns), 3)}")
    print(f"   科氏力项 (零速度): {np.round(kin.coriolis_centrifugal(q_ns, dq0), 4)}")

    print("\n4. 差分逆解（IK）—— 从非奇异构型出发，末端沿 X 移动 5 cm")
    T_target = pinocchio.SE3(
        T_ns.rotation.copy(),
        T_ns.translation + np.array([0.05, 0, 0]),
    )
    q_sol = kin.ik_differential(T_target, init_q6=q_ns)
    if q_sol is not None:
        T_check = kin.fk(q_sol)
        err = np.linalg.norm(T_check.translation - T_target.translation)
        print(f"   IK 解    : {np.round(q_sol, 4)}")
        print(f"   位置误差 : {err*1000:.3f} mm")
    else:
        print("   IK 求解失败")
    print("=" * 55)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cameras",  action="store_true",
                        help="显示双相机实时画面（OpenCV 窗口）")
    parser.add_argument("--kin-test", action="store_true",
                        help="仅运行运动学模块测试，不启动仿真")
    args = parser.parse_args()

    if args.kin_test:
        run_kin_test()
        sys.exit(0)

    env = UR3eGraspEnv(
        show_cameras=args.cameras,
        cam_render_every=15,   # 每 15 个物理步渲染一次相机（约 33Hz）
    )
    env.reset()

    print("场景已启动，开始仿真循环...")
    if args.cameras:
        print("  俯视相机窗口：'Scene Camera (cam)'")
        print("  腕部相机窗口：'Hand Camera (hand_camera)'")
        print("  在相机窗口按 ESC 可关闭相机显示。")
    print("关闭 MuJoCo 查看器窗口可退出。")

    try:
        while env.viewer.is_running():
            for _ in range(10):
                env.step()
    except KeyboardInterrupt:
        print("用户中断，退出仿真。")
    finally:
        env.close()
