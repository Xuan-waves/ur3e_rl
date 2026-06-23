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
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_teleop.config import TeleopConfig
from real_teleop.messages import make_ik_target, make_vr_command, parse_robot_state, parse_vr_command
from real_teleop.safety import SafetyLimiter
from scripts.rlt_gate.eval_rlt_gate import load_checkpoint as load_gate_checkpoint
from scripts.rlt_gate.live_rlt_gate_monitor import compose_input, decode_ros_image, draw_preview
from scripts.rlt_token.train_rlt_stage1 import make_stage1_model
from scripts.rlt_train.config import CFG
from scripts.rollout_smolvla_no_rotvec.rollout_ur3e_smolvla_no_rotvec_rtc import (
    home_orientation,
    infer_no_rotvec_position_mode,
    radial_deadzone,
    read_policy_feature_dims,
    robot_state_to_no_rotvec_tensor,
)
from scripts.rollout_smolvla_no_rotvec.rollout_ur3e_smolvla_no_rotvec_sync import read_policy_n_obs_steps
from vr_servoj_test.rollout.rollout_ur3e_servoj_smolvla import (
    ActionPacket,
    ObservationPacket,
    SmolVLASyncRunner,
    SyncBuffer,
    fmt_vec,
    image_msg_to_tensor,
    make_preview_frame,
    resolve_policy_path,
    stamp_to_sec,
)


@dataclass(slots=True)
class RLTActionPacket:
    ready_stamp: float
    obs_stamp: float
    action: np.ndarray
    ref_action_chunk: np.ndarray
    action_chunk: np.ndarray
    step_index: int
    z_rl: np.ndarray
    gate_prob: float
    inference_s: float


class ButtonEdges:
    def __init__(self) -> None:
        self.prev = {
            "a": False,
            "b": False,
            "x": False,
            "y": False,
            "left_trigger": False,
            "left_grip": False,
        }

    def rising(self, name: str, value: bool) -> bool:
        was = bool(self.prev.get(name, False))
        self.prev[name] = bool(value)
        return bool(value) and not was


class OnlineRLTokenEncoder:
    def __init__(self, runner: SmolVLASyncRunner, stage1_path: Path, device: str, *, amp: bool) -> None:
        from lerobot.utils.constants import (
            ACTION,
            OBS_LANGUAGE_ATTENTION_MASK,
            OBS_LANGUAGE_TOKENS,
        )
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

        self.runner = runner
        self.device = runner.device
        self.amp = bool(amp and self.device == "cuda")
        self.action_key = ACTION
        self.obs_language_tokens = OBS_LANGUAGE_TOKENS
        self.obs_language_attention_mask = OBS_LANGUAGE_ATTENTION_MASK
        self.make_att_2d_masks = make_att_2d_masks

        ckpt = torch.load(stage1_path, map_location=self.device)
        model_cfg = ckpt["model_config"]
        self.model_cfg = model_cfg
        self.architecture = str(model_cfg.get("architecture", "pooled"))
        self.chunk_size = int(model_cfg.get("chunk_size", runner.ckpt_action_horizon))
        self.model = make_stage1_model(model_cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @staticmethod
    def masked_mean(embs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(dtype=embs.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (embs * weights).sum(dim=1) / denom

    def _action_chunk(self, actions: list[ActionPacket]) -> np.ndarray:
        arr = np.stack([np.asarray(action.action, dtype=np.float32).reshape(-1) for action in actions], axis=0)
        if arr.shape[0] < self.chunk_size:
            pad = np.repeat(arr[-1:], self.chunk_size - arr.shape[0], axis=0)
            arr = np.concatenate([arr, pad], axis=0)
        return arr[: self.chunk_size]

    @torch.inference_mode()
    def encode(self, packet: ObservationPacket, actions: list[ActionPacket]) -> np.ndarray:
        if not actions:
            raise ValueError("Cannot encode RL token without VLA actions.")
        autocast = torch.autocast(device_type="cuda") if self.amp else torch.no_grad()
        with autocast:
            batch = dict(packet.batch)
            batch[self.action_key] = torch.from_numpy(self._action_chunk(actions)).unsqueeze(0)
            processed = self.runner.preprocessor(batch)
            input_keys = {
                *self.runner.policy.config.input_features.keys(),
                self.obs_language_tokens,
                self.obs_language_attention_mask,
                self.action_key,
            }
            processed = {key: value for key, value in processed.items() if key in input_keys}
            processed = self.runner.policy._prepare_batch(processed)

            images, img_masks = self.runner.policy.prepare_images(processed)
            state = self.runner.policy.prepare_state(processed)
            action_chunk = self.runner.policy.prepare_action(processed)
            lang_tokens = processed[self.obs_language_tokens]
            lang_masks = processed[self.obs_language_attention_mask]

            prefix_embs, prefix_pad_masks, prefix_att_masks = self.runner.policy.model.embed_prefix(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state=state,
            )
            timestep = torch.zeros(action_chunk.shape[0], dtype=torch.float32, device=action_chunk.device)
            suffix_embs, suffix_pad_masks, suffix_att_masks = self.runner.policy.model.embed_suffix(action_chunk, timestep)
            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
            att_2d_masks = self.make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            (prefix_out, suffix_out), _ = self.runner.policy.model.vlm_with_expert.forward(
                attention_mask=att_2d_masks,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                fill_kv_cache=False,
            )
            if self.architecture == "sequence":
                z_rl = self.model.encode(
                    prefix_out.float(),
                    suffix_out.float(),
                    prefix_pad_masks.to(dtype=torch.bool),
                    suffix_pad_masks.to(dtype=torch.bool),
                )
            else:
                vlm = self.masked_mean(prefix_out.float(), prefix_pad_masks)
                expert = self.masked_mean(suffix_out.float(), suffix_pad_masks)
                z_rl = self.model.encode(vlm, expert)
        return z_rl.detach().cpu().numpy().astype(np.float32).reshape(-1)


class RLTInterventionCollector:
    def __init__(self, node, args: argparse.Namespace) -> None:
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image
        from std_msgs.msg import Float64MultiArray

        self.node = node
        self.args = args
        self.execute = bool(args.execute)
        self.policy_path = resolve_policy_path(args.policy_path)
        self.stage1_checkpoint = args.stage1_checkpoint.expanduser().resolve()
        self.gate_checkpoint = args.gate_checkpoint.expanduser().resolve()
        self.policy_state_dim, self.policy_action_dim = read_policy_feature_dims(self.policy_path)
        model_n_obs_steps = read_policy_n_obs_steps(self.policy_path)
        self.n_obs_steps = max(1, int(args.n_obs_steps or model_n_obs_steps))
        if args.action_position_mode == "auto":
            inferred, _source = infer_no_rotvec_position_mode(self.policy_path)
            args.action_position_mode = inferred or "absolute"

        self.teleop_cfg = TeleopConfig()
        self.safety = SafetyLimiter(self.teleop_cfg)
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
        self.last_action_step_time = 0.0
        self.last_published_pos: np.ndarray | None = None
        self.last_published_gripper: float | None = None
        self.infer_busy = False
        self.last_log_time = 0.0

        self.vr_command: dict[str, Any] | None = None
        self.vr_recv_mono = 0.0
        self.buttons = ButtonEdges()
        self.inference_enabled = False
        self.home_pulse_until = 0.0
        self.home_block_until = 0.0

        self.vr_override_active = False
        self.vr_anchor_ctrl_pos: np.ndarray | None = None
        self.vr_anchor_tcp_pos: np.ndarray | None = None
        self.vr_anchor_gripper = 0.0
        self.vr_anchor_trigger = 0.0
        self.model_resume_block_until = 0.0

        self.gate_model, gate_cfg = load_gate_checkpoint(
            self.gate_checkpoint,
            torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"),
        )
        self.gate_device = next(self.gate_model.parameters()).device
        self.gate_camera = str(gate_cfg.get("camera", "both"))
        self.gate_image_size = int(gate_cfg.get("image_size", 160))
        self.gate_pos_t = float(
            args.gate_positive_threshold
            if args.gate_positive_threshold is not None
            else gate_cfg.get("positive_threshold", 0.6)
        )
        self.gate_neg_t = float(
            args.gate_negative_threshold
            if args.gate_negative_threshold is not None
            else gate_cfg.get("negative_threshold", 0.4)
        )
        self.gate_prob: float | None = None
        self.gate_phase = 0
        self.prev_gate_phase = 0
        self.gate_reentry_locked = False
        self.gate_sample_closed = False
        self.gate_entry_armed = False
        self.gate_pos_count = 0
        self.gate_neg_count = 0
        self.last_gate_infer = 0.0

        self.runner = SmolVLASyncRunner(
            self.policy_path,
            args.device,
            execution_horizon=args.execution_horizon,
            replan_every_step=False,
            amp=args.amp,
        )
        self.token_encoder = OnlineRLTokenEncoder(self.runner, self.stage1_checkpoint, args.device, amp=args.amp)

        self.current_samples: list[dict[str, Any]] = []
        self.pending_reward: float | None = None
        self.save_pending = False
        self.ignore_b_until = 0.0
        self.saved_episodes = 0
        self.episode_lock = threading.RLock()
        self.output_dir = self._make_output_dir(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._write_session_config()

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
            node.create_subscription(
                Float64MultiArray,
                args.robot_state_topic,
                self._on_state,
                data_qos,
                callback_group=self.callback_group,
            ),
            node.create_subscription(
                Float64MultiArray,
                args.vr_raw_topic,
                self._on_raw_vr,
                data_qos,
                callback_group=self.callback_group,
            ),
        ]
        self.button_timer = node.create_timer(1.0 / 60.0, self._button_tick, callback_group=self.callback_group)
        self.infer_timer = node.create_timer(1.0 / max(args.fps, 1.0), self._infer_tick, callback_group=self.callback_group)
        self.publish_timer = node.create_timer(1.0 / max(args.command_hz, 1.0), self._publish_tick, callback_group=self.callback_group)
        if args.preview:
            self.preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
            self.preview_thread.start()

        mode = "EXECUTE" if self.execute else "DRY-RUN"
        node.get_logger().info(
            f"RLT intervention collector ready ({mode}). A=toggle inference, Y=reward1, X=reward0, "
            f"B=save, left_upper_trigger=discard, left_lower_trigger=return_home. "
            f"policy={self.policy_path}, gate={self.gate_checkpoint}, "
            f"stage1={self.stage1_checkpoint}, n_obs_steps={self.n_obs_steps}, "
            f"output={self.output_dir}"
        )

    def _make_output_dir(self, output_root: Path) -> Path:
        root = output_root.expanduser()
        if not root.is_absolute():
            root = (REPO_ROOT / root).resolve()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return root / f"rlt_interventions_{stamp}"

    def _write_session_config(self) -> None:
        cfg = {}
        for key, value in vars(self.args).items():
            cfg[key] = str(value) if isinstance(value, Path) else value
        cfg["policy_state_dim"] = self.policy_state_dim
        cfg["policy_action_dim"] = self.policy_action_dim
        cfg["fixed_rotvec"] = self.fixed_rotvec.tolist()
        (self.output_dir / "session_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    def close(self) -> None:
        self.preview_stop.set()
        if self.preview_thread is not None:
            self.preview_thread.join(timeout=1.0)

    def now_sec(self) -> float:
        return self.node.get_clock().now().nanoseconds * 1e-9

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

    def _on_raw_vr(self, msg) -> None:
        try:
            self.vr_command = parse_vr_command(msg.data)
            self.vr_recv_mono = time.monotonic()
        except Exception as exc:
            self.node.get_logger().warn(f"Bad raw VR command: {exc}")

    def _button_tick(self) -> None:
        if self.vr_command is None or time.monotonic() - self.vr_recv_mono > self.args.vr_override_stale_s:
            return
        if self.buttons.rising("a", bool(self.vr_command.get("home", False))):
            if self.inference_enabled:
                self.inference_enabled = False
                self._reset_policy_tracking(clear_current=True, reset_policy=False)
                self._reset_gate_state()
                self.node.get_logger().info(
                    f"A pressed: inference stopped. samples={len(self.current_samples)} reward={self.pending_reward}"
                )
            else:
                if self.current_samples:
                    self.node.get_logger().warn(
                        "A ignored: unsaved samples exist. Press B twice to save or left upper trigger to discard."
                    )
                else:
                    self.current_samples = []
                    self.pending_reward = None
                    self.save_pending = False
                    self._reset_policy_tracking(clear_current=True, reset_policy=True)
                    self._reset_gate_state()
                    self._send_impedance_reset("trial_start")
                    self.inference_enabled = True
                    self.node.get_logger().info("A pressed: inference started.")
        if self.buttons.rising("left_grip", bool(self.vr_command.get("left_grip", False))):
            self._manual_return_home()
        if self.buttons.rising("y", bool(self.vr_command.get("rl_toggle", False))):
            self.pending_reward = 1.0
            self.node.get_logger().info("Reward selected: 1. Press B to confirm/save or left upper trigger to discard.")
        if self.buttons.rising("x", bool(self.vr_command.get("record_start", False))):
            self.pending_reward = 0.0
            self.node.get_logger().info("Reward selected: 0. Press B to confirm/save or left upper trigger to discard.")
        if self.buttons.rising("b", bool(self.vr_command.get("record_stop", False))):
            if self.now_sec() < self.ignore_b_until:
                return
            self._confirm_save()
        left_pressed = bool(self.vr_command.get("cancel_record", False)) or float(self.vr_command.get("left_trigger", 0.0)) > 0.95
        if self.buttons.rising("left_trigger", left_pressed):
            removed = self._discard_episode_state()
            self.node.get_logger().warn(f"Current RLT intervention episode discarded. removed_samples={removed}")

    def _discard_episode_state(self) -> int:
        with self.episode_lock:
            removed = len(self.current_samples)
            self.current_samples = []
            self.pending_reward = None
            self.save_pending = False
            self.ignore_b_until = self.now_sec() + float(self.args.save_button_cooldown_s)
            self.inference_enabled = False
            self._reset_policy_tracking(clear_current=True, reset_policy=False)
            self._reset_gate_state()
            return removed

    def _manual_return_home(self) -> None:
        self.inference_enabled = False
        self._reset_policy_tracking(clear_current=True, reset_policy=True)
        self._reset_gate_state()
        now = self.now_sec()
        self.home_pulse_until = now + float(self.args.manual_home_pulse_s)
        self.home_block_until = self.home_pulse_until if self.args.block_model_during_home else now
        self.node.get_logger().warn(
            "Manual return_home requested by left lower trigger (SDK left_grip). "
            "Inference stopped and action queue cleared."
        )

    def _send_impedance_reset(self, reason: str) -> None:
        if not self.execute or not bool(self.args.reset_impedance_on_trial_start):
            return
        from std_msgs.msg import Float64MultiArray

        msg = Float64MultiArray()
        gripper = float(np.clip(self.args.home_gripper_value, 0.0, self.args.gripper_max))
        msg.data = make_vr_command({"enable": False, "gripper": gripper, "home": False, "reset_impedance": True})
        self.command_pub.publish(msg)
        self.node.get_logger().info(f"Impedance reset requested: {reason}")

    def _confirm_save(self) -> None:
        if not self.current_samples:
            self.node.get_logger().warn("B pressed, but no intervention samples are buffered.")
            return
        if not self.save_pending:
            self.save_pending = True
            self.inference_enabled = False
            self._reset_policy_tracking(clear_current=True, reset_policy=False)
            reward_text = "none" if self.pending_reward is None else f"{self.pending_reward:.0f}"
            self.node.get_logger().info(
                f"Save pending: samples={len(self.current_samples)} reward={reward_text}. "
                "Press B again to save, or left upper trigger to discard."
            )
            return
        ep = self.saved_episodes
        ep_dir = self.output_dir / f"episode_{ep:06d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        keys = [
            "timestamp",
            "state",
            "vla_action",
            "human_action",
            "executed_action",
            "delta_action",
            "z_rl",
            "gate_prob",
            "gate_phase",
            "intervened",
            "source",
        ]
        source_map = {"model": 0, "vr_override": 1}
        for key in keys:
            values = []
            for sample in self.current_samples:
                value = sample[key]
                if key == "source":
                    value = source_map.get(str(value), -1)
                values.append(value)
            arrays[key] = np.asarray(values, dtype=np.float32)
        np.savez_compressed(ep_dir / "episode.npz", **arrays)
        meta = {
            "episode": ep,
            "samples": len(self.current_samples),
            "reward": None if self.pending_reward is None else float(self.pending_reward),
            "policy_path": str(self.policy_path),
            "stage1_checkpoint": str(self.stage1_checkpoint),
            "gate_checkpoint": str(self.gate_checkpoint),
            "source_map": source_map,
        }
        (ep_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        with (self.output_dir / "episodes.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(meta, separators=(",", ":")) + "\n")
        self.saved_episodes += 1
        self.node.get_logger().info(
            f"Saved RLT intervention episode {ep:06d}: samples={len(self.current_samples)} reward={self.pending_reward}"
        )
        self.current_samples = []
        self.pending_reward = None
        self.save_pending = False

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
            return
        if self.args.block_model_during_home and self.now_sec() < self.home_block_until:
            return
        with self.action_lock:
            if len(self.action_queue) > int(self.args.prefetch_actions):
                return
        if self.now_sec() < self.model_resume_block_until or self.infer_busy:
            return
        stamp = self._take_reference_stamp()
        if stamp is None:
            self._log_wait("waiting for reference image")
            return
        packet = self._build_observation(stamp)
        if packet is None:
            return
        count = max(1, int(self.args.execution_horizon))
        self.infer_busy = True
        threading.Thread(target=self._run_inference, args=(packet, count), daemon=True).start()

    def _run_inference(self, packet: ObservationPacket, count: int) -> None:
        try:
            actions = self.runner.infer_sequence(packet, count)
            z_rl = self.token_encoder.encode(packet, actions)
        except Exception as exc:
            self.node.get_logger().error(f"RLT inference/token failed: {exc}")
            self.infer_busy = False
            return
        if not self.inference_enabled or self.vr_override_active or self.now_sec() < self.model_resume_block_until:
            self.infer_busy = False
            return
        gate_prob = float(self.gate_prob if self.gate_prob is not None else 0.0)
        ref_chunk = np.stack([np.asarray(action.action, dtype=np.float32).reshape(-1) for action in actions], axis=0)
        action_chunk = ref_chunk.copy()
        packets = [
            RLTActionPacket(
                ready_stamp=action.ready_stamp,
                obs_stamp=action.obs_stamp,
                action=np.asarray(action_chunk[idx], dtype=np.float32).reshape(-1),
                ref_action_chunk=ref_chunk.copy(),
                action_chunk=action_chunk.copy(),
                step_index=int(idx),
                z_rl=z_rl,
                gate_prob=gate_prob,
                inference_s=action.inference_s,
            )
            for idx, action in enumerate(actions)
        ]
        with self.action_lock:
            if self.args.replace_queue_on_infer:
                self.action_queue.clear()
            self.action_queue.extend(packets)
            ready_stamp = self.now_sec()
            for action in self.action_queue:
                action.ready_stamp = ready_stamp
        self.infer_busy = False
        self._log_info(
            f"infer actions={len(actions)} gate={gate_prob:.3f}/{self.gate_phase} "
            f"infer={actions[0].inference_s * 1000.0:.0f}ms samples={len(self.current_samples)}"
        )

    def _build_observation(self, stamp: float) -> ObservationPacket | None:
        front, dt_front = self.buffers["front"].nearest(stamp, self.args.max_dt_front_image)
        wrist, dt_wrist = self.buffers["wrist"].nearest(stamp, self.args.max_dt_wrist_image)
        state, dt_state = self.buffers["state"].nearest(stamp, self.args.max_dt_state)
        self.last_dt_map = {"front": dt_front, "wrist": dt_wrist, "state": dt_state}
        if front is None or wrist is None or state is None:
            self._log_wait(f"sync miss {self._format_status()}")
            return None
        self._update_gate(front, wrist)
        try:
            frame_batch = {
                "observation.images.cam_front": image_msg_to_tensor(front),
                "observation.images.cam_wrist": image_msg_to_tensor(wrist),
                "observation.state": robot_state_to_no_rotvec_tensor(
                    state,
                    rl_mark=0.0,
                    gripper_max=self.args.gripper_max,
                    state_dim=self.policy_state_dim,
                ),
            }
            self.observation_history.append(frame_batch)
            history = list(self.observation_history)
            if len(history) < self.n_obs_steps:
                history = [history[0]] * (self.n_obs_steps - len(history)) + history
            batch = {
                "observation.images.cam_front": torch.stack(
                    [item["observation.images.cam_front"] for item in history], dim=0
                ).unsqueeze(0),
                "observation.images.cam_wrist": torch.stack(
                    [item["observation.images.cam_wrist"] for item in history], dim=0
                ).unsqueeze(0),
                "observation.state": torch.stack([item["observation.state"] for item in history], dim=0).unsqueeze(0),
                "task": self.args.task,
            }
        except Exception as exc:
            self.node.get_logger().warn(f"Failed to build observation: {exc}")
            return None
        return ObservationPacket(stamp=stamp, batch=batch, dt_map=self.last_dt_map.copy())

    def _update_gate(self, front_msg: Any, wrist_msg: Any) -> None:
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
            self.gate_prob = prob
            if prob >= self.gate_pos_t:
                self.gate_pos_count += 1
                self.gate_neg_count = 0
            elif prob <= self.gate_neg_t:
                self.gate_neg_count += 1
                self.gate_pos_count = 0
                if self.gate_neg_count >= self.args.gate_hold_frames:
                    self.gate_entry_armed = True
                if self.gate_phase == 1:
                    self.gate_sample_closed = True
            else:
                self.gate_pos_count = 0
                self.gate_neg_count = 0
            self.prev_gate_phase = self.gate_phase
            if (
                self.gate_phase == 0
                and not self.gate_reentry_locked
                and self.gate_entry_armed
                and self.gate_pos_count >= self.args.gate_hold_frames
            ):
                self.gate_phase = 1
                self.gate_sample_closed = False
                self.gate_entry_armed = False
                self.node.get_logger().info(f"RLT gate entered: prob={prob:.3f}")
            elif self.gate_phase == 1 and self.gate_neg_count >= self.args.gate_hold_frames:
                self.gate_phase = 0
                self.gate_reentry_locked = True
                self.node.get_logger().info(f"RLT gate exited: prob={prob:.3f}")
                self._on_gate_exit()
        except Exception as exc:
            self.node.get_logger().warn(f"Gate inference failed: {exc}")

    def _reset_gate_state(self) -> None:
        self.gate_prob = None
        self.gate_phase = 0
        self.prev_gate_phase = 0
        self.gate_reentry_locked = False
        self.gate_sample_closed = False
        self.gate_entry_armed = False
        self.gate_pos_count = 0
        self.gate_neg_count = 0
        self.last_gate_infer = 0.0

    def _should_record_gate_sample(self, packet: RLTActionPacket | None) -> bool:
        if packet is None:
            return False
        if not self.inference_enabled:
            return False
        if self.gate_phase != 1 or self.gate_reentry_locked or self.gate_sample_closed:
            return False
        prob = self.gate_prob if self.gate_prob is not None else float(packet.gate_prob)
        return float(prob) > self.gate_neg_t

    def _on_gate_exit(self) -> None:
        self._drain_action_queue()
        if self.args.home_after_gate_exit:
            now = self.now_sec()
            self.home_pulse_until = now + float(self.args.home_pulse_s)
            self.home_block_until = self.home_pulse_until if self.args.block_model_during_home else now
            self.node.get_logger().info("Gate exit: sending return-home pulse. Press A to stop inference before next trial.")

    def _publish_tick(self) -> None:
        from std_msgs.msg import Float64MultiArray

        stamp = self.now_sec()
        state, state_age = self.buffers["state"].latest(now=stamp)
        if state is None or state_age is None or state_age > self.args.max_dt_state:
            self._log_wait(f"holding: no fresh robot state age={state_age}")
            return
        current_pos = np.asarray(state["tcp_pos"], dtype=float).reshape(3)
        current_gripper = float(np.clip(state.get("gripper", 0.0), 0.0, self.args.gripper_max))

        if stamp < self.home_pulse_until:
            if self.execute:
                msg = Float64MultiArray()
                home_gripper = float(np.clip(self.args.home_gripper_value, 0.0, self.args.gripper_max))
                msg.data = make_vr_command(
                    {
                        "enable": False,
                        "gripper": home_gripper,
                        "home": True,
                        "reset_impedance": bool(self.args.reset_impedance_during_home),
                    }
                )
                self.command_pub.publish(msg)
            return

        if not self.inference_enabled:
            return

        override = self._vr_override_target(current_pos=current_pos, current_gripper=current_gripper)
        source = "model"
        packet: RLTActionPacket | None = None
        vla_action_raw: np.ndarray | None = None
        if override is not None:
            target_pos, gripper = override
            source = "vr_override"
            packet = self.current_action
            vla_action_raw = None if packet is None else packet.action
        else:
            with self.action_lock:
                if self.action_queue and stamp - self.last_action_step_time >= 1.0 / max(self.args.action_step_hz, 1e-6):
                    self.current_action = self.action_queue.popleft()
                    self.last_action_step_time = stamp
                packet = self.current_action
            if packet is None:
                return
            if stamp - packet.ready_stamp > self.args.max_action_age_s:
                self._log_wait(f"drop stale action age={stamp - packet.ready_stamp:.3f}s")
                with self.action_lock:
                    self.current_action = None
                return
            vla_action_raw = packet.action
            target_pos, gripper = self._decode_and_smooth_action(packet.action, current_pos=current_pos)

        if self.execute:
            pose_msg = Float64MultiArray()
            pose_msg.data = make_ik_target(target_pos, self.fixed_quat)
            self.ik_target_pub.publish(pose_msg)
            command_msg = Float64MultiArray()
            command_msg.data = make_vr_command({"enable": False, "gripper": gripper, "home": False})
            self.command_pub.publish(command_msg)

        if self._should_record_gate_sample(packet):
            executed = np.asarray([target_pos[0], target_pos[1], target_pos[2], gripper], dtype=np.float32)
            vla_action = self._raw_to_action_vec(vla_action_raw, current_pos=current_pos) if vla_action_raw is not None else executed
            human_action = executed if source == "vr_override" else vla_action
            self.current_samples.append(
                {
                    "timestamp": stamp,
                    "state": np.asarray(
                        [current_pos[0], current_pos[1], current_pos[2], current_gripper],
                        dtype=np.float32,
                    ),
                    "vla_action": vla_action.astype(np.float32),
                    "human_action": human_action.astype(np.float32),
                    "executed_action": executed.astype(np.float32),
                    "delta_action": (human_action - vla_action).astype(np.float32),
                    "z_rl": packet.z_rl.astype(np.float32),
                    "gate_prob": float(self.gate_prob if self.gate_prob is not None else packet.gate_prob),
                    "gate_phase": float(self.gate_phase),
                    "intervened": 1.0 if source == "vr_override" else 0.0,
                    "source": source,
                }
            )

        rlt_sample = self._should_record_gate_sample(packet)
        self._log_info(
            f"{'pub' if self.execute else 'dry'} src={source} rlt={int(rlt_sample)} infer={int(self.inference_enabled)} "
            f"pend={int(self.save_pending)} r={self.pending_reward} n={len(self.current_samples)} "
            f"p={fmt_vec(target_pos)} grip={gripper:.3f}"
        )

    def _raw_to_action_vec(self, raw_action: np.ndarray, *, current_pos: np.ndarray) -> np.ndarray:
        pos, gripper = self._decode_action_no_filter(raw_action, current_pos=current_pos)
        return np.asarray([pos[0], pos[1], pos[2], gripper], dtype=np.float32)

    def _decode_action_no_filter(self, action: np.ndarray, *, current_pos: np.ndarray) -> tuple[np.ndarray, float]:
        raw_pos = np.asarray(action[:3], dtype=float).reshape(3)
        if self.args.action_position_mode == "relative":
            pos = np.asarray(current_pos, dtype=float).reshape(3) + raw_pos
        else:
            pos = raw_pos
        pos = self.safety.clamp_impedance_workspace(pos)
        if np.isfinite(float(self.args.min_action_z)):
            pos[2] = max(float(pos[2]), float(self.args.min_action_z))
        return pos.astype(np.float32), float(np.clip(action[3], 0.0, self.args.gripper_max))

    def _decode_and_smooth_action(self, action: np.ndarray, *, current_pos: np.ndarray) -> tuple[np.ndarray, float]:
        pos, gripper = self._decode_action_no_filter(action, current_pos=current_pos)
        pos = self._smooth_position(pos)
        if self.last_published_gripper is not None:
            alpha_g = float(np.clip(self.args.action_gripper_filter_alpha, 0.0, 1.0))
            gripper = alpha_g * gripper + (1.0 - alpha_g) * self.last_published_gripper
        gripper = float(np.clip(gripper, 0.0, self.args.gripper_max))
        self.last_published_gripper = gripper
        return pos.astype(np.float32), gripper

    def _vr_override_target(self, *, current_pos: np.ndarray, current_gripper: float) -> tuple[np.ndarray, float] | None:
        if (self.gate_phase != 1 and not bool(self.args.vr_override_anytime)) or self.vr_command is None:
            self._release_vr_override(current_pos)
            return None
        if time.monotonic() - self.vr_recv_mono > self.args.vr_override_stale_s:
            self._release_vr_override(current_pos)
            return None
        pose = self.vr_command.get("pose")
        enabled = bool(self.vr_command.get("enable", False))
        if not enabled or pose is None:
            self._release_vr_override(current_pos)
            return None
        ctrl_pose = np.asarray(pose, dtype=float).reshape(7)
        ctrl_pos = self._vr_control_pos(ctrl_pose[:3], ctrl_pose[3:])
        raw_gripper = float(np.clip(self.vr_command.get("gripper", 0.0), 0.0, self.args.gripper_max))
        if not self.vr_override_active:
            self.vr_override_active = True
            self.vr_anchor_ctrl_pos = ctrl_pos.copy()
            self.vr_anchor_tcp_pos = current_pos.copy()
            self.vr_anchor_gripper = current_gripper
            self.vr_anchor_trigger = raw_gripper
            self.last_published_pos = current_pos.copy()
            self.last_published_gripper = current_gripper
            self._drain_action_queue(clear_current=False)
            self.node.get_logger().info("VR override engaged inside gate=1.")
        sign = np.asarray(self.teleop_cfg.vr_control_position_sign, dtype=float)
        if sign.shape != (3,):
            sign = np.ones(3, dtype=float)
        dpos = (ctrl_pos - self.vr_anchor_ctrl_pos) * sign * float(self.teleop_cfg.scale)
        dpos = radial_deadzone(dpos, float(self.teleop_cfg.dead_zone_pos))
        pos = self.vr_anchor_tcp_pos + dpos
        pos = self.safety.clamp_impedance_workspace(pos)
        if np.isfinite(float(self.args.min_action_z)):
            pos[2] = max(float(pos[2]), float(self.args.min_action_z))
        pos = self._smooth_position(pos)
        gripper = self.vr_anchor_gripper + (raw_gripper - self.vr_anchor_trigger) * float(self.args.vr_override_gripper_gain)
        gripper = float(np.clip(gripper, 0.0, self.args.gripper_max))
        return pos.astype(np.float32), gripper

    def _release_vr_override(self, current_pos: np.ndarray) -> None:
        if not self.vr_override_active:
            return
        self.vr_override_active = False
        self.vr_anchor_ctrl_pos = None
        self.vr_anchor_tcp_pos = None
        self.last_published_pos = current_pos.copy()
        self._drain_action_queue(clear_current=False)
        self.model_resume_block_until = self.now_sec() + float(self.args.vr_override_resume_delay_s)
        self.node.get_logger().info("VR override released; model queue drained.")

    def _drain_action_queue(self, *, clear_current: bool = True) -> None:
        with self.action_lock:
            self.action_queue.clear()
            if clear_current:
                self.current_action = None

    def _reset_policy_tracking(self, *, clear_current: bool, reset_policy: bool) -> None:
        self._drain_action_queue(clear_current=clear_current)
        self.last_action_step_time = 0.0
        self.last_published_pos = None
        self.last_published_gripper = None
        self.vr_override_active = False
        self.vr_anchor_ctrl_pos = None
        self.vr_anchor_tcp_pos = None
        self.model_resume_block_until = 0.0
        if reset_policy:
            try:
                self.runner.policy.reset()
            except Exception as exc:
                self.node.get_logger().warn(f"Policy reset failed: {exc}")

    def _vr_control_pos(self, ctrl_pos: np.ndarray, ctrl_quat: np.ndarray) -> np.ndarray:
        pos = np.asarray(ctrl_pos, dtype=float).reshape(3)
        offset = np.asarray(self.teleop_cfg.vr_controller_pivot_offset_m, dtype=float)
        if offset.shape != (3,) or float(np.linalg.norm(offset)) < 1e-9:
            return pos.copy()
        quat = np.asarray(ctrl_quat, dtype=float).reshape(4)
        norm = float(np.linalg.norm(quat))
        if norm < 1e-9:
            return pos.copy()
        return pos + R.from_quat(quat / norm).apply(offset)

    def _smooth_position(self, pos: np.ndarray) -> np.ndarray:
        pos = np.asarray(pos, dtype=float).reshape(3)
        if self.last_published_pos is not None:
            alpha = float(np.clip(self.args.action_pose_filter_alpha, 0.0, 1.0))
            pos = alpha * pos + (1.0 - alpha) * self.last_published_pos
            delta = pos - self.last_published_pos
            norm = float(np.linalg.norm(delta))
            max_step = float(max(self.args.max_action_pos_step, 1e-6))
            if norm > max_step:
                pos = self.last_published_pos + delta / norm * max_step
        self.last_published_pos = pos.copy()
        return pos

    def _preview_loop(self) -> None:
        try:
            import cv2

            cv2.namedWindow("UR3e RLT Intervention Collector", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("UR3e RLT Intervention Collector", 1280, 520)
            self.node.get_logger().info("OpenCV RLT intervention preview window started.")
        except Exception as exc:
            self.preview_failed = True
            self.node.get_logger().warn(f"OpenCV preview disabled: {exc}")
            return
        period = 1.0 / max(self.args.preview_hz, 1.0)
        while not self.preview_stop.is_set() and not self.preview_failed:
            try:
                import cv2

                now = self.now_sec()
                front, front_age = self.buffers["front"].latest(now=now)
                wrist, wrist_age = self.buffers["wrist"].latest(now=now)
                frame = make_preview_frame(front, wrist, front_age, wrist_age)
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 96), (0, 0, 0), -1)
                rlt_sample = self.gate_phase == 1 and self.gate_entry_armed and not self.gate_sample_closed and not self.gate_reentry_locked
                text = (
                    f"A={'RUN' if self.inference_enabled else 'WAIT'} rlt={int(rlt_sample)} "
                    f"pend={int(self.save_pending)} r={self.pending_reward} "
                    f"n={len(self.current_samples)} saved={self.saved_episodes}"
                )
                cv2.putText(frame, text, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(
                    frame,
                    "Y=reward1 X=reward0 B=stage/save left_upper=discard left_lower=home right_lower=VR override",
                    (14, 66),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (220, 220, 220),
                    2,
                )
                cv2.imshow("UR3e RLT Intervention Collector", frame)
                cv2.waitKey(1)
            except Exception as exc:
                self.preview_failed = True
                self.node.get_logger().warn(f"OpenCV preview disabled: {exc}")
                return
            time.sleep(period)
        try:
            import cv2

            cv2.destroyWindow("UR3e RLT Intervention Collector")
        except Exception:
            pass

    def _format_status(self) -> str:
        dt = " ".join(
            f"{key}={'none' if value is None else f'{value * 1000.0:.0f}ms'}"
            for key, value in self.last_dt_map.items()
        )
        now = self.now_sec()
        topics = " ".join(
            f"{key}#{self.topic_counts[key]}@"
            f"{'none' if self.topic_counts[key] <= 0 else f'{(now - self.topic_last[key]) * 1000.0:.0f}ms'}"
            for key in ("front", "wrist", "state")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect RLT Stage2 human intervention data around RL_gate=1.")
    parser.add_argument("--policy-path", type=Path, default=CFG.policy_path)
    parser.add_argument("--stage1-checkpoint", type=Path, default=CFG.stage1_checkpoint)
    parser.add_argument("--gate-checkpoint", type=Path, default=CFG.gate_checkpoint)
    parser.add_argument("--output-dir", type=Path, default=CFG.output_dir)
    parser.add_argument("--task", default=CFG.task)
    parser.add_argument("--front-image-topic", default=CFG.front_image_topic)
    parser.add_argument("--wrist-image-topic", default=CFG.wrist_image_topic)
    parser.add_argument("--robot-state-topic", default=CFG.robot_state_topic)
    parser.add_argument("--ik-target-topic", default=CFG.ik_target_topic)
    parser.add_argument("--vr-command-topic", default=CFG.vr_command_topic)
    parser.add_argument("--vr-raw-topic", default=CFG.vr_raw_topic)
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
    parser.add_argument("--vr-override-stale-s", type=float, default=CFG.vr_override_stale_s)
    parser.add_argument("--vr-override-resume-delay-s", type=float, default=CFG.vr_override_resume_delay_s)
    parser.add_argument("--vr-override-gripper-gain", type=float, default=CFG.vr_override_gripper_gain)
    parser.add_argument("--vr-override-anytime", action=argparse.BooleanOptionalAction, default=CFG.vr_override_anytime)
    parser.add_argument("--home-pulse-s", type=float, default=CFG.home_pulse_s)
    parser.add_argument("--manual-home-pulse-s", type=float, default=CFG.manual_home_pulse_s)
    parser.add_argument("--home-gripper-value", type=float, default=CFG.home_gripper_value)
    parser.add_argument("--home-after-gate-exit", action=argparse.BooleanOptionalAction, default=CFG.home_after_gate_exit)
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
    parser.add_argument("--save-button-cooldown-s", type=float, default=CFG.save_button_cooldown_s)
    parser.add_argument("--stage2-output-dir", type=Path, default=CFG.stage2_output_dir)
    parser.add_argument("--rlt-enable-actor", action=argparse.BooleanOptionalAction, default=CFG.rlt_enable_actor)
    parser.add_argument("--rlt-warmup-steps", type=int, default=CFG.rlt_warmup_steps)
    parser.add_argument("--rlt-min-actor-updates", type=int, default=CFG.rlt_min_actor_updates)
    parser.add_argument("--rlt-startup-updates", type=int, default=CFG.rlt_startup_updates)
    parser.add_argument("--rlt-startup-log-interval", type=int, default=CFG.rlt_startup_log_interval)
    parser.add_argument("--rlt-startup-empty-cache-interval", type=int, default=CFG.rlt_startup_empty_cache_interval)
    parser.add_argument("--rlt-replay-capacity", type=int, default=CFG.rlt_replay_capacity)
    parser.add_argument("--rlt-batch-size", type=int, default=CFG.rlt_batch_size)
    parser.add_argument("--rlt-replay-demo-ratio", type=float, default=CFG.rlt_replay_demo_ratio)
    parser.add_argument("--rlt-updates-per-step", type=int, default=CFG.rlt_updates_per_step)
    parser.add_argument("--rlt-policy-delay", type=int, default=CFG.rlt_policy_delay)
    parser.add_argument("--rlt-actor-lr", type=float, default=CFG.rlt_actor_lr)
    parser.add_argument("--rlt-critic-lr", type=float, default=CFG.rlt_critic_lr)
    parser.add_argument("--rlt-gamma", type=float, default=CFG.rlt_gamma)
    parser.add_argument("--rlt-tau", type=float, default=CFG.rlt_tau)
    parser.add_argument("--rlt-bc-weight", type=float, default=CFG.rlt_bc_weight)
    parser.add_argument("--rlt-target-noise-xyz", type=float, default=CFG.rlt_target_noise_xyz)
    parser.add_argument("--rlt-target-noise-clip-xyz", type=float, default=CFG.rlt_target_noise_clip_xyz)
    parser.add_argument("--rlt-actor-hidden-dim", type=int, default=CFG.rlt_actor_hidden_dim)
    parser.add_argument("--rlt-critic-hidden-dim", type=int, default=CFG.rlt_critic_hidden_dim)
    parser.add_argument("--rlt-fusion-mode", choices=("direct", "projected"), default=CFG.rlt_fusion_mode)
    parser.add_argument("--rlt-fusion-dim", type=int, default=CFG.rlt_fusion_dim)
    parser.add_argument("--rlt-train-action-dim", type=int, choices=(3,), default=CFG.rlt_train_action_dim)
    parser.add_argument("--rlt-action-chunk-steps", type=int, default=CFG.rlt_action_chunk_steps)
    parser.add_argument("--rlt-action-delta-scale-xyz", type=float, default=CFG.rlt_action_delta_scale_xyz)
    parser.add_argument("--rlt-checkpoint", type=Path, default=CFG.rlt_checkpoint)
    parser.add_argument("--rlt-buffer-dir", type=Path, default=CFG.rlt_buffer_dir)
    parser.add_argument(
        "--no-rlt-buffer-dir",
        dest="rlt_buffer_dir",
        action="store_const",
        const=None,
        help="Start with empty Stage2 replay buffers instead of loading a warm buffer directory.",
    )
    parser.add_argument("--rlt-save-every-episodes", type=int, default=CFG.rlt_save_every_episodes)
    parser.add_argument("--rlt-snapshot-buffers", action=argparse.BooleanOptionalAction, default=CFG.rlt_snapshot_buffers)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    import rclpy
    from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor

    args = parse_args()
    args.policy_path = resolve_policy_path(args.policy_path)
    hf_home = args.hf_home.expanduser()
    if not hf_home.is_absolute():
        hf_home = (REPO_ROOT / hf_home).resolve()
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    rclpy.init()
    node = rclpy.create_node("ur3e_rlt_intervention_collector")
    collector: RLTInterventionCollector | None = None
    try:
        collector = RLTInterventionCollector(node, args)
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        try:
            executor.spin()
        finally:
            executor.shutdown()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if collector is not None:
            collector.close()
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
