# UR3e Cartesian Impedance Motion

This folder contains the reusable impedance-control layer.  It is intentionally
independent from ROS2 and VR code:

- `config.py` stores profiles and runtime defaults.
- `controller.py` contains the pure impedance law plus the RTDE forceMode motion
  wrapper.
- `scripts/hardware/ur3e_position_impedance_test.py` is only a CLI tuning tool.

The current hardware path should use `state_source="rtde"`.  The Jacobian state
source is kept for later, but it must not be used on the real robot until the
MuJoCo base/TCP frame matches the UR controller TCP frame.

## Minimal API

```python
from real_teleop.impedance import (
    DEFAULT_IMPEDANCE_TEST_CONFIG,
    ImpedanceRuntimeConfig,
    RtdeImpedanceMotion,
)

test_cfg = DEFAULT_IMPEDANCE_TEST_CONFIG
profile = test_cfg.profiles["passive"]
runtime = ImpedanceRuntimeConfig(
    robot_ip="192.168.5.1",
    control_hz=200.0,
    state_source="rtde",
)

motion = RtdeImpedanceMotion.connect(runtime.robot_ip, profile, runtime=runtime)
try:
    motion.move_to_home()
    motion.configure_force_mode()
    motion.set_target_from_current()

    # In teleop, replace this with the VR target pose at 100 Hz.
    motion.set_target_pose([0.13, -0.30, 0.15])

    while True:
        motion.step(execute=True)
finally:
    motion.close()
```

`set_target_pose(position, rotation_vector=None)` is the main control interface
for later VR integration.  The 200 Hz robot loop can call `step()` continuously,
while the VR node updates the target at 100 Hz.
