import mujoco
import mujoco.viewer
import time

def main():
    # 1. 加载模型
    # 确保 scene.xml 与此脚本在同一目录下
    # 如果 XML 中有相对路径引用 (如 ../ur3e_robotiq/...)，请确保目录结构正确
    model = mujoco.MjModel.from_xml_path("/home/xuan/ur3e_rl/mujoco_env/assets/scenes/scene.xml")
    data = mujoco.MjData(model)

    key_id = model.key("home")
    # print(key_id)
    data.qpos[:7] = key_id.qpos[:7]
    data.ctrl[:7] = key_id.ctrl[:7]
    # mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    # data.ctrl[:6] = data.qpos[:6]
        

    # 2. 启动查看器 (Passive mode)
    # launch_passive 会启动一个非阻塞的 GUI 界面
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("查看器已启动，可以在 GUI 的 'Camera' 菜单中切换视角")

        viewer.cam.fixedcamid = 0
        # 强制将摄像机模式设置为 FIXED（如果不设置，可能会保留上次关闭时的自由视角）
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

        
        # 3. 仿真循环
        while viewer.is_running():
            step_start = time.time()

            # 可以在这里加入你的控制逻辑
            # 例如: data.ctrl[...] = ...

            # 执行一步物理仿真
            mujoco.mj_step(model, data)

            # 同步渲染数据
            viewer.sync()

            # 控制循环速度，使其接近实时
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"发生错误: {e}")