#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_teleop.config import TeleopConfig
from real_teleop.messages import make_ik_target, make_vr_command, parse_robot_state
from real_teleop.safety import SafetyLimiter
from scripts.rlt_gate.eval_rlt_gate import load_checkpoint as load_gate_checkpoint
from scripts.rlt_gate.live_rlt_gate_monitor import compose_input, decode_ros_image
from scripts.rlt_train.collect_rlt_interventions import OnlineRLTokenEncoder, RLTActionPacket
from scripts.rlt_train.config import CFG
from scripts.rlt_train.hil_serl_core import HILSERLConfig, HILSERLTrainer
from scripts.rollout_smolvla_no_rotvec.rollout_ur3e_smolvla_no_rotvec_rtc import (
    home_orientation,
    infer_no_rotvec_position_mode,
    read_policy_feature_dims,
    robot_state_to_no_rotvec_tensor,
)
from scripts.rollout_smolvla_no_rotvec.rollout_ur3e_smolvla_no_rotvec_sync import (
    read_policy_n_obs_steps,
    run_pre_model_startup_sequence,
)
from vr_servoj_test.rollout.rollout_ur3e_servoj_smolvla import (
    ObservationPacket,
    SmolVLASyncRunner,
    SyncBuffer,
    fmt_vec,
    image_msg_to_tensor,
    make_preview_frame,
    resolve_policy_path,
    stamp_to_sec,
)


DEFAULT_RLT_CHECKPOINT = (
    REPO_ROOT / "outputs/rlt_stage2/hil_serl_stage2_20260616_204313/checkpoints/stage2_ep000050.pt"
)
DEFAULT_READY_GATE_CHECKPOINT = REPO_ROOT / "outputs/ready_gate/ready_gate_20260617_150953/best.pt"


def _resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_stage2_trainer(checkpoint: Path, device: str) -> HILSERLTrainer:
    checkpoint = _resolve(checkpoint)
    raw = torch.load(checkpoint, map_location="cpu")
    cfg_dict = dict(raw.get("config", {}))
    valid = HILSERLConfig.__dataclass_fields__.keys()
    cfg = HILSERLConfig(**{key: value for key, value in cfg_dict.items() if key in valid})
    trainer = HILSERLTrainer(cfg, device)
    trainer.load(checkpoint)
    trainer.actor.eval()
    for param in trainer.actor.parameters():
        param.requires_grad_(False)
    return trainer


def choose_torch_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return name


class RLTNoVRRollout:
    def __init__(self, node, args: argparse.Namespace) -> None:
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image
        from std_msgs.msg import Float64MultiArray

        self.node = node
        self.args = args
        self.execute = bool(args.execute)
        self.policy_path = resolve_policy_path(args.policy_path)
        self.stage1_checkpoint = _resolve(args.stage1_checkpoint)
        self.rlt_checkpoint = _resolve(args.rlt_checkpoint)
        self.gate_checkpoint = _resolve(args.gate_checkpoint)
        self.ready_gate_checkpoint = _resolve(args.ready_gate_checkpoint)

        self.policy_state_dim, self.policy_action_dim = read_policy_feature_dims(self.policy_path)
        if self.policy_state_dim != 4 or self.policy_action_dim != 4:
            raise ValueError(f"Expected no-rotvec state/action dim 4, got {self.policy_state_dim}/{self.policy_action_dim}")
        model_n_obs_steps = read_policy_n_obs_steps(self.policy_path)
        self.n_obs_steps = max(1, int(args.n_obs_steps or model_n_obs_steps))
        if args.action_position_mode == "auto":
            inferred, _source = infer_no_rotvec_position_mode(self.policy_path)
            args.action_position_mode = inferred or "absolute"

        self.torch_device = choose_torch_device(args.device)
        args.device = self.torch_device
        self.safety = SafetyLimiter(TeleopConfig())
        self.fixed_quat, self.fixed_rotvec = home_orientation()
        self.pending_reference_stamps: queue.SimpleQueue[float] = queue.SimpleQueue()
        self.buffers = {
            "front": SyncBuffer(args.buffer_maxlen),
            "wrist": SyncBuffer(args.buffer_maxlen),
            "state": SyncBuffer(args.buffer_maxlen),
        }
        self.topic_counts = {"front": 0, "wrist": 0, "state": 0}
        self.topic_last = {"front": 0.0, "wrist": 0.0, "state": 0.0}
        self.last_dt_map: dict[str, float | None] = {}
        self.observation_history: deque[dict[str, Any]] = deque(maxlen=self.n_obs_steps)

        self.action_queue: deque[RLTActionPacket] = deque()
        self.action_lock = threading.Lock()
        self.current_action: RLTActionPacket | None = None
        self.rollout_generation = 0
        self.last_action_step_time = 0.0
        self.last_published_pos: np.ndarray | None = None
        self.last_published_gripper: float | None = None
        self.infer_busy = False
        self.last_log_time = 0.0
        self.inference_enabled = not bool(args.wait_ready_on_start)
        self.waiting_for_ready = bool(args.wait_ready_on_start)
        self.home_pulse_until = 0.0
        self.home_block_until = 0.0
        self.ready_start_allowed_after = self.now_sec() + float(args.ready_after_home_settle_s)

        self.gate_model, gate_cfg = load_gate_checkpoint(self.gate_checkpoint, torch.device(self.torch_device))
        self.gate_device = next(self.gate_model.parameters()).device
        self.gate_camera = str(gate_cfg.get("camera", "both"))
        self.gate_image_size = int(gate_cfg.get("image_size", 160))
        self.gate_pos_t = float(args.gate_positive_threshold if args.gate_positive_threshold is not None else gate_cfg.get("positive_threshold", 0.6))
        self.gate_neg_t = float(args.gate_negative_threshold if args.gate_negative_threshold is not None else gate_cfg.get("negative_threshold", 0.4))
        self.gate_prob: float | None = None
        self.gate_phase = 0
        self.gate_reentry_locked = False
        self.gate_entry_armed = False
        self.gate_pos_count = 0
        self.gate_neg_count = 0
        self.last_gate_infer = 0.0

        self.ready_model, ready_cfg = load_gate_checkpoint(self.ready_gate_checkpoint, self.gate_device)
        self.ready_device = next(self.ready_model.parameters()).device
        self.ready_camera = str(ready_cfg.get("camera", "both"))
        self.ready_image_size = int(ready_cfg.get("image_size", 128))
        self.ready_prob: float | None = None
        self.ready_phase = 0
        self.ready_pos_count = 0
        self.ready_neg_count = 0
        self.last_ready_infer = 0.0

        self.runner = SmolVLASyncRunner(
            self.policy_path,
            args.device,
            execution_horizon=args.execution_horizon,
            replan_every_step=False,
            amp=args.amp,
        )
        self.token_encoder = OnlineRLTokenEncoder(self.runner, self.stage1_checkpoint, args.device, amp=args.amp)
        self.trainer = load_stage2_trainer(self.rlt_checkpoint, args.device)

        self.preview_stop = threading.Event()
        self.preview_failed = False
        self.preview_thread: threading.Thread | None = None

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        data_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.callback_group = ReentrantCallbackGroup()
        self.ik_target_pub = node.create_publisher(Float64MultiArray, args.ik_target_topic, data_qos)
        self.command_pub = node.create_publisher(Float64MultiArray, args.vr_command_topic, data_qos)
        self.subs = [
            node.create_subscription(Image, args.front_image_topic, self._on_front, image_qos, callback_group=self.callback_group),
            node.create_subscription(Image, args.wrist_image_topic, self._on_wrist, image_qos, callback_group=self.callback_group),
            node.create_subscription(Float64MultiArray, args.robot_state_topic, self._on_state, data_qos, callback_group=self.callback_group),
        ]
        self.infer_timer = node.create_timer(1.0 / max(args.fps, 1.0), self._infer_tick, callback_group=self.callback_group)
        self.publish_timer = node.create_timer(1.0 / max(args.command_hz, 1.0), self._publish_tick, callback_group=self.callback_group)
        self.ready_timer = node.create_timer(1.0 / max(args.ready_gate_infer_hz, 1.0), self._ready_tick, callback_group=self.callback_group)
        if args.preview:
            self.preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
            self.preview_thread.start()

        mode = "EXECUTE" if self.execute else "DRY-RUN"
        node.get_logger().info(
            f"RLT no-VR rollout ready ({mode}). policy={self.policy_path}, rlt={self.rlt_checkpoint}, "
            f"gate={self.gate_checkpoint}, ready_gate={self.ready_gate_checkpoint}, "
            f"start={'wait_ready' if args.wait_ready_on_start else 'immediate'}, "
            f"action_mode={args.action_position_mode}, chunk={self.trainer.action_chunk_steps}, "
            f"fixed_rot={fmt_vec(self.fixed_rotvec)}"
        )

    def now_sec(self) -> float:
        return self.node.get_clock().now().nanoseconds * 1e-9

    def close(self) -> None:
        self.preview_stop.set()
        if self.preview_thread is not None:
            self.preview_thread.join(timeout=1.0)
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    def _msg_time_sec(self, msg: Any) -> float:
        header = getattr(msg, "header", None)
        stamp = getattr(header, "stamp", None)
        return self.now_sec() if stamp is None else stamp_to_sec(stamp)

    def _push(self, key: str, stamp: float, value: Any) -> None:
        self.buffers[key].push(stamp, value)
        self.topic_counts[key] += 1
        self.topic_last[key] = self.now_sec()

    def _on_front(self, msg) -> None:
        stamp = self._msg_time_sec(msg)
        self._push("front", stamp, msg)
        if self.args.sync_reference == "front":
            self.pending_reference_stamps.put(stamp)

    def _on_wrist(self, msg) -> None:
        stamp = self._msg_time_sec(msg)
        self._push("wrist", stamp, msg)
        if self.args.sync_reference == "wrist":
            self.pending_reference_stamps.put(stamp)

    def _on_state(self, msg) -> None:
        try:
            self._push("state", self.now_sec(), parse_robot_state(msg.data))
        except Exception as exc:
            self.node.get_logger().warn(f"Bad robot_state: {exc}")

    def _take_reference_stamp(self) -> float | None:
        if self.args.sync_reference in {"timer", "both"}:
            return self.now_sec()
        stamp = None
        while True:
            try:
                stamp = self.pending_reference_stamps.get_nowait()
            except queue.Empty:
                break
        return stamp

    def _infer_tick(self) -> None:
        if not self.inference_enabled:
            self._log_info(
                f"WAIT ready={self.ready_phase}/{self._fmt_prob(self.ready_prob)} "
                f"rl_gate={self.gate_phase}/{self._fmt_prob(self.gate_prob)}"
            )
            return
        if self.args.block_model_during_home and self.now_sec() < self.home_block_until:
            self._log_info(f"WAIT home/reset {self.home_block_until - self.now_sec():.2f}s")
            return
        with self.action_lock:
            if len(self.action_queue) > int(self.args.prefetch_actions):
                return
        if self.infer_busy:
            return
        stamp = self._take_reference_stamp()
        if stamp is None:
            self._log_wait("waiting for reference image")
            return
        packet = self._build_observation(stamp)
        if packet is None:
            return
        self.infer_busy = True
        generation = self.rollout_generation
        threading.Thread(target=self._run_inference, args=(packet, generation), daemon=True).start()

    def _run_inference(self, packet: ObservationPacket, generation: int | None = None) -> None:
        generation = self.rollout_generation if generation is None else int(generation)
        try:
            actions = self.runner.infer_sequence(packet, max(1, int(self.args.execution_horizon)))
            z_rl = self.token_encoder.encode(packet, actions)
            ref_chunk = np.stack([np.asarray(action.action, dtype=np.float32).reshape(-1) for action in actions], axis=0)
            packets = [
                RLTActionPacket(
                    ready_stamp=self.now_sec(),
                    obs_stamp=action.obs_stamp,
                    action=np.asarray(ref_chunk[idx], dtype=np.float32).reshape(-1),
                    ref_action_chunk=ref_chunk.copy(),
                    action_chunk=ref_chunk.copy(),
                    step_index=int(idx),
                    z_rl=z_rl,
                    gate_prob=float(self.gate_prob if self.gate_prob is not None else 0.0),
                    inference_s=action.inference_s,
                )
                for idx, action in enumerate(actions)
            ]
        except Exception as exc:
            self.node.get_logger().error(f"RLT rollout inference failed: {exc}")
            self.infer_busy = False
            return
        if not self.inference_enabled or generation != self.rollout_generation:
            self.infer_busy = False
            self._log_info(
                f"discard inference from old rollout generation={generation}, current={self.rollout_generation}"
            )
            return
        with self.action_lock:
            if generation != self.rollout_generation:
                self.infer_busy = False
                return
            if self.args.replace_queue_on_infer:
                self.action_queue.clear()
            self.action_queue.extend(packets)
        self.infer_busy = False
        self._log_info(
            f"infer actions={len(packets)} rl_gate={self.gate_phase}/{self._fmt_prob(self.gate_prob)} "
            f"infer={packets[0].inference_s * 1000.0:.0f}ms"
        )

    def _build_observation(self, stamp: float) -> ObservationPacket | None:
        front, dt_front = self.buffers["front"].nearest(stamp, self.args.max_dt_front_image)
        wrist, dt_wrist = self.buffers["wrist"].nearest(stamp, self.args.max_dt_wrist_image)
        state, dt_state = self.buffers["state"].nearest(stamp, self.args.max_dt_state)
        self.last_dt_map = {"front": dt_front, "wrist": dt_wrist, "state": dt_state}
        if front is None or wrist is None or state is None:
            self._log_wait(f"sync miss {self._format_status()}")
            return None
        self._update_rl_gate(front, wrist)
        try:
            item = {
                "observation.images.cam_front": image_msg_to_tensor(front),
                "observation.images.cam_wrist": image_msg_to_tensor(wrist),
                "observation.state": robot_state_to_no_rotvec_tensor(
                    state,
                    rl_mark=0.0,
                    gripper_max=self.args.gripper_max,
                    state_dim=self.policy_state_dim,
                ),
            }
            self.observation_history.append(item)
            history = list(self.observation_history)
            if len(history) < self.n_obs_steps:
                history = [history[0]] * (self.n_obs_steps - len(history)) + history
            batch = {
                "observation.images.cam_front": torch.stack([x["observation.images.cam_front"] for x in history], dim=0).unsqueeze(0),
                "observation.images.cam_wrist": torch.stack([x["observation.images.cam_wrist"] for x in history], dim=0).unsqueeze(0),
                "observation.state": torch.stack([x["observation.state"] for x in history], dim=0).unsqueeze(0),
                "task": self.args.task,
            }
        except Exception as exc:
            self.node.get_logger().warn(f"Failed to build observation: {exc}")
            return None
        return ObservationPacket(stamp=stamp, batch=batch, dt_map=self.last_dt_map.copy())

    def _publish_tick(self) -> None:
        from std_msgs.msg import Float64MultiArray

        stamp = self.now_sec()
        if stamp < self.home_pulse_until:
            if self.execute:
                msg = Float64MultiArray()
                msg.data = make_vr_command(
                    {
                        "enable": False,
                        "gripper": float(np.clip(self.args.home_gripper_value, 0.0, self.args.gripper_max)),
                        "home": True,
                        "reset_impedance": bool(self.args.reset_impedance_during_home),
                    }
                )
                self.command_pub.publish(msg)
            return
        if not self.inference_enabled:
            return
        state, state_age = self.buffers["state"].latest(now=stamp)
        if state is None or state_age is None or state_age > self.args.max_dt_state:
            self._log_wait(f"holding: no fresh robot state age={state_age}")
            return
        current_pos = np.asarray(state["tcp_pos"], dtype=float).reshape(3)
        current_gripper = float(np.clip(state.get("gripper", 0.0), 0.0, self.args.gripper_max))
        state_vec = np.asarray([current_pos[0], current_pos[1], current_pos[2], current_gripper], dtype=np.float32)

        with self.action_lock:
            if self.action_queue and stamp - self.last_action_step_time >= 1.0 / max(self.args.action_step_hz, 1e-6):
                self.current_action = self.action_queue.popleft()
                self.last_action_step_time = stamp
            packet = self.current_action
        if packet is None:
            self._log_info("RUN waiting for first action packet")
            return
        if stamp - packet.ready_stamp > self.args.max_action_age_s:
            self._log_wait(f"drop stale action age={stamp - packet.ready_stamp:.3f}s")
            with self.action_lock:
                self.current_action = None
            return

        ref_chunk = self.trainer._numpy_chunk(packet.ref_action_chunk)
        step_idx = min(max(int(packet.step_index), 0), ref_chunk.shape[0] - 1)
        source = "vla"
        if self.gate_phase == 1 and not self.gate_reentry_locked:
            action_chunk = self.trainer.act_chunk(packet.z_rl, state_vec, ref_chunk)
            action = self._clip_action(action_chunk[step_idx])
            source = "rlt"
        else:
            action = self._clip_action(ref_chunk[step_idx])
        target_pos, gripper = self._smooth_final_action(action)

        if self.execute:
            pose_msg = Float64MultiArray()
            pose_msg.data = make_ik_target(target_pos, self.fixed_quat)
            self.ik_target_pub.publish(pose_msg)
            command_msg = Float64MultiArray()
            command_msg.data = make_vr_command({"enable": False, "gripper": gripper, "home": False})
            self.command_pub.publish(command_msg)

        self._log_info(
            f"{'pub' if self.execute else 'dry'} src={source} rl_gate={self.gate_phase}/{self._fmt_prob(self.gate_prob)} "
            f"ready={self.ready_phase}/{self._fmt_prob(self.ready_prob)} p={fmt_vec(target_pos)} grip={gripper:.3f} "
            f"q={len(self.action_queue)}"
        )

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        out = np.asarray(action, dtype=np.float32).reshape(4).copy()
        if self.args.action_position_mode == "relative":
            raise RuntimeError("RLT no-VR rollout currently expects absolute no-rotvec actions.")
        out[:3] = self.safety.clamp_impedance_workspace(out[:3])
        if np.isfinite(float(self.args.min_action_z)):
            out[2] = max(float(out[2]), float(self.args.min_action_z))
        out[3] = float(np.clip(out[3], 0.0, self.args.gripper_max))
        return out

    def _smooth_final_action(self, action: np.ndarray) -> tuple[np.ndarray, float]:
        pos = np.asarray(action[:3], dtype=float).reshape(3)
        if self.last_published_pos is not None:
            alpha = float(np.clip(self.args.action_pose_filter_alpha, 0.0, 1.0))
            pos = alpha * pos + (1.0 - alpha) * self.last_published_pos
            delta = pos - self.last_published_pos
            norm = float(np.linalg.norm(delta))
            max_step = float(max(self.args.max_action_pos_step, 1e-6))
            if norm > max_step:
                pos = self.last_published_pos + delta / norm * max_step
        self.last_published_pos = pos.copy()
        gripper = float(action[3])
        if self.last_published_gripper is not None:
            alpha_g = float(np.clip(self.args.action_gripper_filter_alpha, 0.0, 1.0))
            gripper = alpha_g * gripper + (1.0 - alpha_g) * self.last_published_gripper
        gripper = float(np.clip(gripper, 0.0, self.args.gripper_max))
        self.last_published_gripper = gripper
        return pos.astype(np.float32), gripper

    def _update_rl_gate(self, front_msg: Any, wrist_msg: Any) -> None:
        now = time.monotonic()
        if now - self.last_gate_infer < 1.0 / max(self.args.gate_infer_hz, 1e-6):
            return
        self.last_gate_infer = now
        try:
            front_bgr = decode_ros_image(front_msg)
            wrist_bgr = decode_ros_image(wrist_msg)
            x = compose_input(front_bgr, wrist_bgr, self.gate_camera, self.gate_image_size).to(self.gate_device)
            with torch.no_grad():
                prob = float(torch.sigmoid(self.gate_model(x).view(-1))[0].detach().cpu())
        except Exception as exc:
            self.node.get_logger().warn(f"RL gate inference failed: {exc}")
            return
        self.gate_prob = prob
        if prob >= self.gate_pos_t:
            self.gate_pos_count += 1
            self.gate_neg_count = 0
        elif prob <= self.gate_neg_t:
            self.gate_neg_count += 1
            self.gate_pos_count = 0
            if self.gate_neg_count >= self.args.gate_hold_frames:
                self.gate_entry_armed = True
        else:
            self.gate_pos_count = 0
            self.gate_neg_count = 0
        if self.gate_phase == 0 and not self.gate_reentry_locked and self.gate_entry_armed and self.gate_pos_count >= self.args.gate_hold_frames:
            self.gate_phase = 1
            self.gate_entry_armed = False
            self.node.get_logger().info(f"RL gate entered: prob={prob:.3f}")
        elif self.gate_phase == 1 and self.gate_neg_count >= self.args.gate_hold_frames:
            self.gate_phase = 0
            self.gate_reentry_locked = True
            self._on_gate_exit(prob)

    def _reset_rl_gate(self) -> None:
        self.gate_prob = None
        self.gate_phase = 0
        self.gate_reentry_locked = False
        self.gate_entry_armed = False
        self.gate_pos_count = 0
        self.gate_neg_count = 0
        self.last_gate_infer = 0.0

    def _on_gate_exit(self, prob: float) -> None:
        self._drain_actions()
        self.inference_enabled = False
        self.waiting_for_ready = True
        now = self.now_sec()
        self.home_pulse_until = now + float(self.args.home_pulse_s)
        self.home_block_until = self.home_pulse_until if self.args.block_model_during_home else now
        self.ready_start_allowed_after = self.home_pulse_until + float(self.args.ready_after_home_settle_s)
        self.node.get_logger().info(f"RL gate exited: prob={prob:.3f}; return_home then wait ready_gate.")

    def _ready_tick(self) -> None:
        if not self.waiting_for_ready:
            return
        now = self.now_sec()
        if time.monotonic() - self.last_ready_infer < 1.0 / max(self.args.ready_gate_infer_hz, 1e-6):
            return
        self.last_ready_infer = time.monotonic()
        front, front_age = self.buffers["front"].latest(now=now)
        wrist, wrist_age = self.buffers["wrist"].latest(now=now)
        if front is None or wrist is None:
            return
        if front_age is not None and front_age > self.args.max_dt_front_image:
            return
        if wrist_age is not None and wrist_age > self.args.max_dt_wrist_image:
            return
        try:
            x = compose_input(decode_ros_image(front), decode_ros_image(wrist), self.ready_camera, self.ready_image_size).to(self.ready_device)
            with torch.no_grad():
                prob = float(torch.sigmoid(self.ready_model(x).view(-1))[0].detach().cpu())
        except Exception as exc:
            self.node.get_logger().warn(f"ready_gate inference failed: {exc}")
            return
        self.ready_prob = prob
        if prob >= self.args.ready_gate_positive_threshold:
            self.ready_pos_count += 1
            self.ready_neg_count = 0
        elif prob <= self.args.ready_gate_negative_threshold:
            self.ready_neg_count += 1
            self.ready_pos_count = 0
        else:
            self.ready_pos_count = 0
            self.ready_neg_count = 0
        old = self.ready_phase
        if self.ready_phase == 0 and self.ready_pos_count >= self.args.ready_gate_hold_frames:
            self.ready_phase = 1
        elif self.ready_phase == 1 and self.ready_neg_count >= self.args.ready_gate_hold_frames:
            self.ready_phase = 0
        if self.ready_phase != old:
            self.node.get_logger().info(f"ready_gate={self.ready_phase} prob={prob:.3f}")
        if (
            self.args.auto_start_on_ready
            and self.ready_phase == 1
            and now >= self.ready_start_allowed_after
            and not self.inference_enabled
        ):
            self._start_rollout("ready_gate")

    def _start_rollout(self, reason: str) -> None:
        self._drain_actions()
        self._reset_rl_gate()
        self.last_published_pos = None
        self.last_published_gripper = None
        self.last_action_step_time = 0.0
        self.waiting_for_ready = False
        self.inference_enabled = True
        if self.execute and bool(self.args.reset_impedance_on_trial_start):
            from std_msgs.msg import Float64MultiArray

            msg = Float64MultiArray()
            msg.data = make_vr_command({"enable": False, "home": False, "reset_impedance": True})
            self.command_pub.publish(msg)
        self.node.get_logger().info(f"Rollout started by {reason}.")

    def _drain_actions(self) -> None:
        self.rollout_generation += 1
        with self.action_lock:
            self.action_queue.clear()
            self.current_action = None
        self.observation_history.clear()
        while True:
            try:
                self.pending_reference_stamps.get_nowait()
            except queue.Empty:
                break

    def _preview_loop(self) -> None:
        try:
            cv2.namedWindow("UR3e RLT No-VR Rollout", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("UR3e RLT No-VR Rollout", 1280, 520)
        except Exception as exc:
            self.preview_failed = True
            self.node.get_logger().warn(f"OpenCV preview disabled: {exc}")
            return
        period = 1.0 / max(self.args.preview_hz, 1.0)
        while not self.preview_stop.is_set() and not self.preview_failed:
            try:
                now = self.now_sec()
                front, front_age = self.buffers["front"].latest(now=now)
                wrist, wrist_age = self.buffers["wrist"].latest(now=now)
                frame = make_preview_frame(front, wrist, front_age, wrist_age)
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 96), (0, 0, 0), -1)
                cv2.putText(
                    frame,
                    f"A={'RUN' if self.inference_enabled else 'WAIT'} rl={self.gate_phase}/{self._fmt_prob(self.gate_prob)} "
                    f"ready={self.ready_phase}/{self._fmt_prob(self.ready_prob)} wait_ready={int(self.waiting_for_ready)}",
                    (14, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.68,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(frame, "No VR subscriptions. q/Esc closes preview only.", (14, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (220, 220, 220), 2)
                cv2.imshow("UR3e RLT No-VR Rollout", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    self.preview_stop.set()
                    return
            except Exception as exc:
                self.preview_failed = True
                self.node.get_logger().warn(f"OpenCV preview disabled: {exc}")
                return
            time.sleep(period)
        try:
            cv2.destroyWindow("UR3e RLT No-VR Rollout")
        except Exception:
            pass

    def _format_status(self) -> str:
        dt = " ".join(f"{k}={'none' if v is None else f'{v * 1000.0:.0f}ms'}" for k, v in self.last_dt_map.items())
        now = self.now_sec()
        topics = " ".join(
            f"{k}#{self.topic_counts[k]}@{'none' if self.topic_counts[k] <= 0 else f'{(now - self.topic_last[k]) * 1000.0:.0f}ms'}"
            for k in ("front", "wrist", "state")
        )
        return f"sync[{dt}] topics[{topics}]"

    def _log_wait(self, text: str) -> None:
        if self.now_sec() - self.last_log_time > 1.0 / max(self.args.log_hz, 1e-6):
            self.last_log_time = self.now_sec()
            self.node.get_logger().warn(text)

    def _log_info(self, text: str) -> None:
        if self.now_sec() - self.last_log_time > 1.0 / max(self.args.log_hz, 1e-6):
            self.last_log_time = self.now_sec()
            self.node.get_logger().info(text)

    @staticmethod
    def _fmt_prob(value: float | None) -> str:
        return "none" if value is None else f"{value:.3f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent no-VR RLT rollout with ready-gate closed loop.")
    parser.add_argument("--policy-path", type=Path, default=CFG.policy_path)
    parser.add_argument("--stage1-checkpoint", type=Path, default=CFG.stage1_checkpoint)
    parser.add_argument("--gate-checkpoint", type=Path, default=CFG.gate_checkpoint)
    parser.add_argument("--rlt-checkpoint", type=Path, default=DEFAULT_RLT_CHECKPOINT)
    parser.add_argument("--ready-gate-checkpoint", type=Path, default=DEFAULT_READY_GATE_CHECKPOINT)
    parser.add_argument("--task", default=CFG.task)
    parser.add_argument("--front-image-topic", default=CFG.front_image_topic)
    parser.add_argument("--wrist-image-topic", default=CFG.wrist_image_topic)
    parser.add_argument("--robot-state-topic", default=CFG.robot_state_topic)
    parser.add_argument("--ik-target-topic", default=CFG.ik_target_topic)
    parser.add_argument("--vr-command-topic", default=CFG.vr_command_topic)
    parser.add_argument("--fps", type=float, default=CFG.fps)
    parser.add_argument("--command-hz", type=float, default=CFG.command_hz)
    parser.add_argument("--action-step-hz", type=float, default=CFG.action_step_hz)
    parser.add_argument("--n-obs-steps", type=int, default=CFG.n_obs_steps)
    parser.add_argument("--execution-horizon", type=int, default=CFG.execution_horizon)
    parser.add_argument("--sync-reference", choices=("front", "wrist", "timer", "both"), default=CFG.sync_reference)
    parser.add_argument("--buffer-maxlen", type=int, default=CFG.buffer_maxlen)
    parser.add_argument("--max-dt-front-image", type=float, default=CFG.max_dt_front_image)
    parser.add_argument("--max-dt-wrist-image", type=float, default=CFG.max_dt_wrist_image)
    parser.add_argument("--max-dt-state", type=float, default=CFG.max_dt_state)
    parser.add_argument("--device", default=CFG.device)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=CFG.amp)
    parser.add_argument("--hf-home", type=Path, default=CFG.hf_home)
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=CFG.offline)
    parser.add_argument("--action-position-mode", choices=("auto", "relative", "absolute"), default=CFG.action_position_mode)
    parser.add_argument("--gripper-max", type=float, default=CFG.gripper_max)
    parser.add_argument("--min-action-z", type=float, default=CFG.min_action_z)
    parser.add_argument("--action-pose-filter-alpha", type=float, default=CFG.action_pose_filter_alpha)
    parser.add_argument("--action-gripper-filter-alpha", type=float, default=CFG.action_gripper_filter_alpha)
    parser.add_argument("--max-action-pos-step", type=float, default=CFG.max_action_pos_step)
    parser.add_argument("--max-action-age-s", type=float, default=CFG.max_action_age_s)
    parser.add_argument("--prefetch-actions", type=int, default=CFG.prefetch_actions)
    parser.add_argument("--replace-queue-on-infer", action=argparse.BooleanOptionalAction, default=CFG.replace_queue_on_infer)
    parser.add_argument("--gate-positive-threshold", type=float, default=CFG.gate_positive_threshold)
    parser.add_argument("--gate-negative-threshold", type=float, default=CFG.gate_negative_threshold)
    parser.add_argument("--gate-hold-frames", type=int, default=CFG.gate_hold_frames)
    parser.add_argument("--gate-infer-hz", type=float, default=CFG.gate_infer_hz)
    parser.add_argument("--ready-gate-positive-threshold", type=float, default=0.6)
    parser.add_argument("--ready-gate-negative-threshold", type=float, default=0.4)
    parser.add_argument("--ready-gate-hold-frames", type=int, default=3)
    parser.add_argument("--ready-gate-infer-hz", type=float, default=15.0)
    parser.add_argument("--ready-after-home-settle-s", type=float, default=0.8)
    parser.add_argument("--auto-start-on-ready", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wait-ready-on-start", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--home-pulse-s", type=float, default=CFG.home_pulse_s)
    parser.add_argument("--home-gripper-value", type=float, default=CFG.home_gripper_value)
    parser.add_argument("--block-model-during-home", action=argparse.BooleanOptionalAction, default=CFG.block_model_during_home)
    parser.add_argument("--reset-impedance-on-trial-start", action=argparse.BooleanOptionalAction, default=CFG.reset_impedance_on_trial_start)
    parser.add_argument("--reset-impedance-during-home", action=argparse.BooleanOptionalAction, default=CFG.reset_impedance_during_home)
    parser.add_argument("--return-home-on-start", action=argparse.BooleanOptionalAction, default=CFG.return_home_on_start)
    parser.add_argument("--start-home-delay-s", type=float, default=CFG.start_home_delay_s)
    parser.add_argument("--start-home-pulse-s", type=float, default=CFG.start_home_pulse_s)
    parser.add_argument("--start-home-settle-s", type=float, default=CFG.start_home_settle_s)
    parser.add_argument("--start-open-gripper-s", type=float, default=CFG.start_open_gripper_s)
    parser.add_argument("--start-open-gripper-value", type=float, default=CFG.start_open_gripper_value)
    parser.add_argument("--preview", action=argparse.BooleanOptionalAction, default=CFG.preview)
    parser.add_argument("--preview-hz", type=float, default=CFG.preview_hz)
    parser.add_argument("--log-hz", type=float, default=CFG.log_hz)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    import rclpy
    from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor

    args = parse_args()
    args.policy_path = resolve_policy_path(args.policy_path)
    hf_home = _resolve(args.hf_home)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    rclpy.init()
    node = rclpy.create_node("ur3e_rlt_no_vr_rollout")
    rollout: RLTNoVRRollout | None = None
    try:
        run_pre_model_startup_sequence(node, args)
        rollout = RLTNoVRRollout(node, args)
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        try:
            executor.spin()
        finally:
            executor.shutdown()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rollout is not None:
            rollout.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
