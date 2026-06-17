#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_teleop.messages import make_vr_command
from scripts.rlt_gate.eval_rlt_gate import load_checkpoint as load_gate_checkpoint
from scripts.rlt_gate.live_rlt_gate_monitor import compose_input, decode_ros_image
from scripts.rlt_train.collect_rlt_interventions import parse_args as parse_stage2_args
from scripts.rlt_train.train_hil_serl_stage2 import HILSERLStage2Runner
from scripts.rollout_smolvla_no_rotvec.rollout_ur3e_smolvla_no_rotvec_sync import run_pre_model_startup_sequence
from vr_servoj_test.rollout.rollout_ur3e_servoj_smolvla import make_preview_frame, resolve_policy_path


DEFAULT_RLT_CHECKPOINT = (
    REPO_ROOT
    / "outputs/rlt_stage2/hil_serl_stage2_20260616_204313/checkpoints/stage2_ep000050.pt"
)
DEFAULT_READY_GATE_CHECKPOINT = REPO_ROOT / "outputs/ready_gate/ready_gate_20260617_150953/best.pt"


def _argv_has(names: tuple[str, ...], argv: list[str]) -> bool:
    return any(arg == name or arg.startswith(name + "=") for arg in argv for name in names)


class RLTReadyGateRollout(HILSERLStage2Runner):
    """Closed-loop rollout: VLA+RLT inside RL_gate, home after exit, wait for ready_gate before next trial."""

    def __init__(self, node, args: argparse.Namespace) -> None:
        self.ready_gate_checkpoint = args.ready_gate_checkpoint.expanduser()
        if not self.ready_gate_checkpoint.is_absolute():
            self.ready_gate_checkpoint = (REPO_ROOT / self.ready_gate_checkpoint).resolve()
        self.ready_gate_pos_t = float(args.ready_gate_positive_threshold)
        self.ready_gate_neg_t = float(args.ready_gate_negative_threshold)
        self.ready_gate_hold_frames = int(args.ready_gate_hold_frames)
        self.ready_gate_infer_hz = float(args.ready_gate_infer_hz)
        self.ready_after_home_settle_s = float(args.ready_after_home_settle_s)
        self.auto_start_on_ready = bool(args.auto_start_on_ready)
        self.wait_ready_on_start = bool(args.wait_ready_on_start)
        self.ready_gate_prob: float | None = None
        self.ready_gate_phase = 0
        self.ready_gate_pos_count = 0
        self.ready_gate_neg_count = 0
        self.last_ready_gate_infer = 0.0
        self.waiting_for_ready = self.wait_ready_on_start
        self.ready_start_allowed_after = 0.0
        self.stage2_loaded = args.rlt_checkpoint is not None
        self.last_rollout_wait_log = 0.0

        super().__init__(node, args)
        self.ready_gate_model, ready_cfg = load_gate_checkpoint(self.ready_gate_checkpoint, self.gate_device)
        self.ready_gate_device = next(self.ready_gate_model.parameters()).device
        self.ready_gate_camera = str(ready_cfg.get("camera", "both"))
        self.ready_gate_image_size = int(ready_cfg.get("image_size", 128))

        self.ready_start_allowed_after = self.now_sec() + self.ready_after_home_settle_s
        self.ready_timer = node.create_timer(
            1.0 / max(self.ready_gate_infer_hz, 1e-6),
            self._ready_gate_tick,
            callback_group=self.callback_group,
        )

        if self.waiting_for_ready:
            self.inference_enabled = False
            self.node.get_logger().info("Waiting for ready_gate before first rollout.")
        self.node.get_logger().info(
            "RLT ready-gate rollout enabled. "
            f"rlt_checkpoint={args.rlt_checkpoint}, ready_gate={self.ready_gate_checkpoint}, "
            f"ready_camera={self.ready_gate_camera}, auto_start_on_ready={self.auto_start_on_ready}"
        )

    def _actor_ready(self) -> bool:
        return bool(self.args.rlt_enable_actor) and bool(getattr(self, "stage2_loaded", False))

    def _record_transition_seed(self, **_: Any) -> None:
        return

    def _discard_episode_state(self) -> int:
        self.episode_transitions = []
        self.current_samples = []
        self.pending_seed = None
        self.pending_reward = None
        self.save_pending = False
        return 0

    def _confirm_save(self) -> None:
        self.node.get_logger().info("B ignored in rollout mode: no Stage2 data is saved.")

    def _on_gate_exit(self) -> None:
        self._drain_action_queue()
        self.inference_enabled = False
        self.waiting_for_ready = True
        now = self.now_sec()
        if self.args.home_after_gate_exit:
            self.home_pulse_until = now + float(self.args.home_pulse_s)
            self.home_block_until = self.home_pulse_until if self.args.block_model_during_home else now
            self.ready_start_allowed_after = self.home_pulse_until + self.ready_after_home_settle_s
            self.node.get_logger().info(
                "RLT gate exited: return_home pulse sent; waiting for ready_gate before next rollout."
            )
        else:
            self.ready_start_allowed_after = now + self.ready_after_home_settle_s
            self.node.get_logger().info("RLT gate exited: waiting for ready_gate before next rollout.")

    def _manual_return_home(self) -> None:
        super()._manual_return_home()
        self.waiting_for_ready = True
        self.ready_start_allowed_after = self.home_pulse_until + self.ready_after_home_settle_s

    def _button_tick(self) -> None:
        if self.vr_command is None or time.monotonic() - self.vr_recv_mono > self.args.vr_override_stale_s:
            return
        if self.buttons.rising("a", bool(self.vr_command.get("home", False))):
            if self.inference_enabled:
                self.inference_enabled = False
                self.waiting_for_ready = True
                self._reset_policy_tracking(clear_current=True, reset_policy=False)
                self._reset_gate_state()
                self.node.get_logger().info("A pressed: inference stopped; waiting for ready_gate.")
            else:
                self._start_next_rollout("A")
        if self.buttons.rising("left_grip", bool(self.vr_command.get("left_grip", False))):
            self._manual_return_home()
        left_pressed = bool(self.vr_command.get("cancel_record", False)) or float(self.vr_command.get("left_trigger", 0.0)) > 0.95
        if self.buttons.rising("left_trigger", left_pressed):
            self.inference_enabled = False
            self.waiting_for_ready = True
            self._reset_policy_tracking(clear_current=True, reset_policy=False)
            self._reset_gate_state()
            self.node.get_logger().warn("Left upper trigger: inference stopped; waiting for ready_gate.")

    def _start_next_rollout(self, reason: str) -> None:
        self._discard_episode_state()
        self._reset_policy_tracking(clear_current=True, reset_policy=True)
        self._reset_gate_state()
        self._send_impedance_reset(f"rlt_ready_gate_start:{reason}")
        self.last_published_pos = None
        self.last_published_gripper = None
        self.last_action_step_time = 0.0
        self.inference_enabled = True
        self.waiting_for_ready = False
        self.node.get_logger().info(f"Rollout started by {reason}.")

    def _infer_tick(self) -> None:
        now = self.now_sec()
        if not self.inference_enabled:
            if now - self.last_rollout_wait_log > 1.0 / max(self.args.log_hz, 1e-6):
                self.last_rollout_wait_log = now
                ready_prob = "none" if self.ready_gate_prob is None else f"{self.ready_gate_prob:.3f}"
                self.node.get_logger().info(
                    f"WAIT: inference disabled, waiting_for_ready={int(self.waiting_for_ready)}, "
                    f"ready={self.ready_gate_phase}/{ready_prob}. Press A to force start."
                )
            return
        if self.args.block_model_during_home and now < self.home_block_until:
            if now - self.last_rollout_wait_log > 1.0 / max(self.args.log_hz, 1e-6):
                self.last_rollout_wait_log = now
                self.node.get_logger().info(f"WAIT: home/reset pulse active for {self.home_block_until - now:.2f}s")
            return
        return super()._infer_tick()

    def _publish_tick(self) -> None:
        now = self.now_sec()
        if self.execute is False and now - self.last_rollout_wait_log > 1.0 / max(self.args.log_hz, 1e-6):
            self.last_rollout_wait_log = now
            self.node.get_logger().warn("DRY-RUN: --execute is not set, targets are not published to robot.")
        if self.inference_enabled and now >= self.home_pulse_until:
            with self.action_lock:
                no_packet = self.current_action is None and not self.action_queue
            if no_packet and not self.infer_busy and now - self.last_rollout_wait_log > 1.0 / max(self.args.log_hz, 1e-6):
                self.last_rollout_wait_log = now
                self.node.get_logger().info("RUN: waiting for first inferred action packet.")
        return super()._publish_tick()

    def _ready_gate_tick(self) -> None:
        now = self.now_sec()
        if time.monotonic() - self.last_ready_gate_infer < 1.0 / max(self.ready_gate_infer_hz, 1e-6):
            return
        self.last_ready_gate_infer = time.monotonic()
        front_msg, front_age = self.buffers["front"].latest(now=now)
        wrist_msg, wrist_age = self.buffers["wrist"].latest(now=now)
        if front_msg is None or wrist_msg is None:
            return
        if front_age is not None and front_age > self.args.max_dt_front_image:
            return
        if wrist_age is not None and wrist_age > self.args.max_dt_wrist_image:
            return
        try:
            front_bgr = decode_ros_image(front_msg)
            wrist_bgr = decode_ros_image(wrist_msg)
            x = compose_input(front_bgr, wrist_bgr, self.ready_gate_camera, self.ready_gate_image_size).to(
                self.ready_gate_device
            )
            with torch.no_grad():
                prob = float(torch.sigmoid(self.ready_gate_model(x).view(-1))[0].detach().cpu())
        except Exception as exc:
            self.node.get_logger().warn(f"ready_gate inference failed: {exc}")
            return

        self.ready_gate_prob = prob
        old_phase = self.ready_gate_phase
        if prob >= self.ready_gate_pos_t:
            self.ready_gate_pos_count += 1
            self.ready_gate_neg_count = 0
        elif prob <= self.ready_gate_neg_t:
            self.ready_gate_neg_count += 1
            self.ready_gate_pos_count = 0
        else:
            self.ready_gate_pos_count = 0
            self.ready_gate_neg_count = 0

        if self.ready_gate_phase == 0 and self.ready_gate_pos_count >= self.ready_gate_hold_frames:
            self.ready_gate_phase = 1
        elif self.ready_gate_phase == 1 and self.ready_gate_neg_count >= self.ready_gate_hold_frames:
            self.ready_gate_phase = 0

        if self.ready_gate_phase != old_phase:
            self.node.get_logger().info(f"ready_gate={self.ready_gate_phase} prob={prob:.3f}")

        if (
            self.auto_start_on_ready
            and self.waiting_for_ready
            and not self.inference_enabled
            and self.ready_gate_phase == 1
            and now >= self.ready_start_allowed_after
        ):
            self._start_next_rollout("ready_gate")

    def _preview_loop(self) -> None:
        try:
            cv2.namedWindow("UR3e RLT Ready-Gate Rollout", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("UR3e RLT Ready-Gate Rollout", 1280, 520)
            self.node.get_logger().info("OpenCV RLT ready-gate rollout preview window started.")
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
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 100), (0, 0, 0), -1)
                rlt_active = self.gate_phase == 1 and not self.gate_reentry_locked
                ready_prob = "none" if self.ready_gate_prob is None else f"{self.ready_gate_prob:.3f}"
                text = (
                    f"A={'RUN' if self.inference_enabled else 'WAIT'} "
                    f"rlt={int(rlt_active)} rl_gate={0.0 if self.gate_prob is None else self.gate_prob:.3f} "
                    f"ready={self.ready_gate_phase}/{ready_prob} wait_ready={int(self.waiting_for_ready)}"
                )
                cv2.putText(frame, text, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 255), 2)
                cv2.putText(
                    frame,
                    "A=start/stop  left_lower=home  left_upper=stop/wait  right_lower=VR override",
                    (14, 66),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.56,
                    (220, 220, 220),
                    2,
                )
                cv2.imshow("UR3e RLT Ready-Gate Rollout", frame)
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
            cv2.destroyWindow("UR3e RLT Ready-Gate Rollout")
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    ready_parser = argparse.ArgumentParser(add_help=False)
    ready_parser.add_argument("--ready-gate-checkpoint", type=Path, default=DEFAULT_READY_GATE_CHECKPOINT)
    ready_parser.add_argument("--ready-gate-positive-threshold", type=float, default=0.6)
    ready_parser.add_argument("--ready-gate-negative-threshold", type=float, default=0.4)
    ready_parser.add_argument("--ready-gate-hold-frames", type=int, default=3)
    ready_parser.add_argument("--ready-gate-infer-hz", type=float, default=15.0)
    ready_parser.add_argument("--ready-after-home-settle-s", type=float, default=0.8)
    ready_parser.add_argument("--auto-start-on-ready", action=argparse.BooleanOptionalAction, default=True)
    ready_parser.add_argument("--wait-ready-on-start", action=argparse.BooleanOptionalAction, default=True)
    ready_args, remaining = ready_parser.parse_known_args()

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *remaining]
        args = parse_stage2_args()
    finally:
        sys.argv = old_argv

    if args.rlt_checkpoint is None:
        args.rlt_checkpoint = DEFAULT_RLT_CHECKPOINT
    if not _argv_has(("--stage2-output-dir",), remaining):
        args.stage2_output_dir = REPO_ROOT / "outputs/rlt_rollout"
    if not _argv_has(("--rlt-buffer-dir", "--no-rlt-buffer-dir"), remaining):
        args.rlt_buffer_dir = None
    if not _argv_has(("--rlt-startup-updates",), remaining):
        args.rlt_startup_updates = 0
    if not _argv_has(("--rlt-warmup-steps",), remaining):
        args.rlt_warmup_steps = 0
    if not _argv_has(("--rlt-min-actor-updates",), remaining):
        args.rlt_min_actor_updates = 0
    if not _argv_has(("--rlt-snapshot-buffers", "--no-rlt-snapshot-buffers"), remaining):
        args.rlt_snapshot_buffers = False
    if not _argv_has(("--rlt-save-every-episodes",), remaining):
        args.rlt_save_every_episodes = 0
    args.rlt_enable_actor = True

    for key, value in vars(ready_args).items():
        setattr(args, key, value)
    return args


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
    node = rclpy.create_node("ur3e_rlt_ready_gate_rollout")
    rollout: RLTReadyGateRollout | None = None
    try:
        run_pre_model_startup_sequence(node, args)
        rollout = RLTReadyGateRollout(node, args)
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
