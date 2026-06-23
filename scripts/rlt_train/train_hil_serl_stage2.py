#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_teleop.messages import make_ik_target, make_vr_command
from scripts.rlt_train.collect_rlt_interventions import RLTActionPacket, RLTInterventionCollector, parse_args
from scripts.rlt_train.config import CFG
from scripts.rlt_train.hil_serl_core import HILSERLConfig, HILSERLTrainer, HILTransition, ReplayBuffer
from scripts.rollout_smolvla_no_rotvec.rollout_ur3e_smolvla_no_rotvec_sync import run_pre_model_startup_sequence
from vr_servoj_test.rollout.rollout_ur3e_servoj_smolvla import fmt_vec, resolve_policy_path


class HILSERLStage2Runner(RLTInterventionCollector):
    """HIL-SERL style online Stage2 around `RL_gate=1`.

    The frozen VLA proposes a reference action. A small residual actor can refine
    it after warmup. If the human holds the VR override trigger, the executed
    action becomes the human action and the transition is also inserted into the
    intervention buffer. Updates sample from online/intervention buffers with a
    HIL-SERL-like mixture.
    """

    def __init__(self, node, args) -> None:
        args.output_dir = args.stage2_output_dir
        preview_requested = bool(args.preview)
        args.preview = False
        try:
            super().__init__(node, args)
        finally:
            args.preview = preview_requested
        if self.policy_action_dim != 4 or self.policy_state_dim != 4:
            raise ValueError(
                f"HIL-SERL Stage2 currently expects no-rotvec state/action dim 4, "
                f"got state={self.policy_state_dim}, action={self.policy_action_dim}."
            )

        self.hil_cfg = HILSERLConfig(
            z_dim=int(self.token_encoder.model.z_dim),
            state_dim=self.policy_state_dim,
            action_dim=self.policy_action_dim,
            train_action_dim=int(args.rlt_train_action_dim),
            action_chunk_steps=int(args.rlt_action_chunk_steps),
            actor_hidden_dim=int(args.rlt_actor_hidden_dim),
            critic_hidden_dim=int(args.rlt_critic_hidden_dim),
            fusion_mode=str(args.rlt_fusion_mode),
            fusion_dim=int(args.rlt_fusion_dim),
            action_delta_scale_xyz=float(args.rlt_action_delta_scale_xyz),
            actor_lr=float(args.rlt_actor_lr),
            critic_lr=float(args.rlt_critic_lr),
            gamma=float(args.rlt_gamma),
            tau=float(args.rlt_tau),
            batch_size=int(args.rlt_batch_size),
            replay_demo_ratio=float(args.rlt_replay_demo_ratio),
            bc_weight=float(args.rlt_bc_weight),
            target_noise_xyz=float(args.rlt_target_noise_xyz),
            target_noise_clip_xyz=float(args.rlt_target_noise_clip_xyz),
            train_after=int(args.rlt_warmup_steps),
            updates_per_step=int(args.rlt_updates_per_step),
            policy_delay=int(args.rlt_policy_delay),
            action_high=(1.0, 1.0, 1.0),
        )
        self.trainer = HILSERLTrainer(self.hil_cfg, args.device)
        self.trainer_lock = threading.Lock()
        if args.rlt_checkpoint is not None:
            ckpt = args.rlt_checkpoint.expanduser()
            if not ckpt.is_absolute():
                ckpt = (REPO_ROOT / ckpt).resolve()
            with self.trainer_lock:
                self.trainer.load(ckpt)
            self.node.get_logger().info(f"Loaded Stage2 checkpoint: {ckpt}")

        self.online_buffer = ReplayBuffer(int(args.rlt_replay_capacity))
        self.intervention_buffer = ReplayBuffer(int(args.rlt_replay_capacity))
        if args.rlt_buffer_dir is not None:
            buffer_dir = args.rlt_buffer_dir.expanduser()
            if not buffer_dir.is_absolute():
                buffer_dir = (REPO_ROOT / buffer_dir).resolve()
            online_path = buffer_dir / "online_buffer_latest.npz"
            intervention_path = buffer_dir / "intervention_buffer_latest.npz"
            online_size = self.online_buffer.load_npz(online_path)
            intervention_size = self.intervention_buffer.load_npz(intervention_path)
            self.inserted_transitions = online_size
            self.intervention_transitions = intervention_size
            self.node.get_logger().info(
                f"Loaded replay buffers: online={online_size} from {online_path}, "
                f"intervention={intervention_size} from {intervention_path}"
            )
        else:
            self.node.get_logger().info("Starting with empty Stage2 replay buffers (--no-rlt-buffer-dir).")
        self.pending_seed: dict[str, Any] | None = None
        self.episode_transitions: list[HILTransition] = []
        self.inserted_transitions = getattr(self, "inserted_transitions", 0)
        self.intervention_transitions = getattr(self, "intervention_transitions", 0)
        self.last_train_info: dict[str, float] | None = None
        self.last_actor_train_info: dict[str, float] | None = None
        self.training_disabled_after_error = False
        self.stage2_metadata_path = self.output_dir / "stage2_metadata.json"
        self._run_startup_updates()
        self._write_stage2_metadata()
        if preview_requested:
            self.preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
            self.preview_thread.start()
        self.node.get_logger().info(
            "HIL-SERL Stage2 online trainer enabled. "
            f"actor={args.rlt_enable_actor}, warmup={args.rlt_warmup_steps}, "
            f"min_actor_updates={args.rlt_min_actor_updates}, startup_updates={args.rlt_startup_updates}, "
            f"batch={args.rlt_batch_size}, demo_ratio={args.rlt_replay_demo_ratio:.2f}, "
            f"fusion={args.rlt_fusion_mode}/{args.rlt_fusion_dim}, "
            f"train_action_dim={args.rlt_train_action_dim}, chunk_steps={args.rlt_action_chunk_steps}, "
            f"delta_xyz={args.rlt_action_delta_scale_xyz:.3f}"
        )

    def _make_output_dir(self, output_root: Path) -> Path:
        from datetime import datetime

        root = output_root.expanduser()
        if not root.is_absolute():
            root = (REPO_ROOT / root).resolve()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return root / f"hil_serl_stage2_{stamp}"

    def _write_stage2_metadata(self) -> None:
        cfg = {
            "policy_path": str(self.policy_path),
            "stage1_checkpoint": str(self.stage1_checkpoint),
            "gate_checkpoint": str(self.gate_checkpoint),
            "hil_serl_config": asdict(self.hil_cfg),
            "startup_updates": int(self.args.rlt_startup_updates),
            "min_actor_updates": int(self.args.rlt_min_actor_updates),
            "trainer_update_step_at_start": int(self.trainer.update_step),
            "notes": (
                "online_buffer stores all gate=1 transitions; intervention_buffer stores only VR intervention "
                "transitions and is oversampled during updates, following HIL-SERL/RLPD."
            ),
        }
        self.stage2_metadata_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    def _discard_episode_state(self) -> int:
        with self.episode_lock:
            removed = super()._discard_episode_state()
            self.pending_seed = None
            self.episode_transitions = []
            return removed

    def _on_gate_exit(self) -> None:
        self._reset_policy_tracking(clear_current=True, reset_policy=False)
        if self.args.home_after_gate_exit:
            now = self.now_sec()
            self.home_pulse_until = now + float(self.args.home_pulse_s)
            self.home_block_until = self.home_pulse_until if self.args.block_model_during_home else now
        self.inference_enabled = False
        self.node.get_logger().info(
            "Gate exit: inference paused. Select reward with Y/X, press B twice to finalize, then A for next trial."
        )

    def _run_startup_updates(self) -> None:
        count = max(0, int(self.args.rlt_startup_updates))
        if count <= 0:
            return
        if len(self.online_buffer) < int(self.args.rlt_warmup_steps):
            self.node.get_logger().warn(
                f"Skipping startup Stage2 updates: online_buffer={len(self.online_buffer)} "
                f"< warmup={self.args.rlt_warmup_steps}."
            )
            return
        self.node.get_logger().info(
            f"Running startup Stage2 updates before actor control: updates={count}, "
            f"online={len(self.online_buffer)}, intervention={len(self.intervention_buffer)}"
        )
        log_interval = max(0, int(getattr(self.args, "rlt_startup_log_interval", 0)))
        empty_cache_interval = max(0, int(getattr(self.args, "rlt_startup_empty_cache_interval", 0)))
        device_is_cuda = self.trainer.device.type == "cuda" and torch.cuda.is_available()
        start_t = time.perf_counter()
        last_t = start_t
        done = 0
        for idx in range(count):
            try:
                with self.trainer_lock:
                    info = self.trainer.update(self.online_buffer, self.intervention_buffer)
            except RuntimeError as exc:
                self.training_disabled_after_error = True
                self.node.get_logger().error(f"Startup Stage2 update failed; disabling updates. error={exc}")
                break
            if info is not None:
                self.last_train_info = info
                if info.get("actor_updated", 0.0) > 0.5:
                    self.last_actor_train_info = info
            done = idx + 1
            if log_interval > 0 and (done == 1 or done % log_interval == 0 or done == count):
                now_t = time.perf_counter()
                window_s = max(now_t - last_t, 1e-9)
                total_s = max(now_t - start_t, 1e-9)
                last_t = now_t
                mem_text = ""
                if device_is_cuda:
                    allocated = torch.cuda.memory_allocated(self.trainer.device) / (1024.0**2)
                    reserved = torch.cuda.memory_reserved(self.trainer.device) / (1024.0**2)
                    mem_text = f", cuda_alloc={allocated:.0f}MiB, cuda_reserved={reserved:.0f}MiB"
                train = self.last_train_info or {}
                actor_train = self.last_actor_train_info or train
                self.node.get_logger().info(
                    f"Startup Stage2 update {done}/{count}: "
                    f"{(1.0 / (window_s / min(log_interval, done))):.1f} upd/s window, "
                    f"{done / total_s:.1f} upd/s avg, "
                    f"critic={train.get('critic_loss', 0.0):.4f}, "
                    f"actor={actor_train.get('actor_loss', 0.0):.4f}"
                    f"{mem_text}"
                )
            if empty_cache_interval > 0 and device_is_cuda and done % empty_cache_interval == 0:
                torch.cuda.empty_cache()
        train = self.last_train_info or {}
        actor_train = self.last_actor_train_info or train
        self.node.get_logger().info(
            f"Startup Stage2 updates done: updates={done}, trainer_step={self.trainer.update_step}, "
            f"critic={train.get('critic_loss', 0.0):.4f}, actor={actor_train.get('actor_loss', 0.0):.4f}, "
            f"bc={actor_train.get('bc_loss', 0.0):.4f}, q={actor_train.get('actor_q_loss', 0.0):.4f}"
        )

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
        override = self._vr_override_target(current_pos=current_pos, current_gripper=current_gripper)
        if not self.inference_enabled:
            if override is None:
                return
            target_pos, gripper = override
            if self.execute:
                pose_msg = Float64MultiArray()
                pose_msg.data = make_ik_target(target_pos, self.fixed_quat)
                self.ik_target_pub.publish(pose_msg)
                command_msg = Float64MultiArray()
                command_msg.data = make_vr_command({"enable": False, "gripper": gripper, "home": False})
                self.command_pub.publish(command_msg)
            self._log_info(
                f"{'pub' if self.execute else 'dry'} src=vr_manual rlt=0 "
                f"pend={int(self.save_pending)} r={self.pending_reward} staged={len(self.episode_transitions)} "
                f"p={fmt_vec(target_pos)} grip={gripper:.3f} infer=0"
            )
            return

        state_vec = np.asarray([current_pos[0], current_pos[1], current_pos[2], current_gripper], dtype=np.float32)
        source = "vla_warmup"
        residual_mm = 0.0
        rlt_active = False
        packet: RLTActionPacket | None
        ref_action_chunk: np.ndarray | None = None
        executed_action_chunk: np.ndarray | None = None
        if override is not None:
            target_pos, gripper = override
            source = "vr_override"
            packet = self.current_action
            ref_action_chunk = None if packet is None else self._fixed_action_chunk(packet.ref_action_chunk)
            if packet is not None:
                step_idx = min(max(int(packet.step_index), 0), ref_action_chunk.shape[0] - 1)
                ref_action = self._raw_to_action_vec(ref_action_chunk[step_idx], current_pos=current_pos)
            else:
                ref_action = None
            executed = np.asarray([target_pos[0], target_pos[1], target_pos[2], gripper], dtype=np.float32)
            if ref_action_chunk is not None and packet is not None:
                executed_action_chunk = ref_action_chunk.copy()
                executed_action_chunk[step_idx] = executed
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
            ref_action_chunk = self._fixed_action_chunk(packet.ref_action_chunk)
            step_idx = min(max(int(packet.step_index), 0), ref_action_chunk.shape[0] - 1)
            ref_action = self._raw_to_action_vec(ref_action_chunk[step_idx], current_pos=current_pos)
            rlt_active = self._actor_ready() and self._should_record_gate_sample(packet)
            if rlt_active:
                with self.trainer_lock:
                    actor_chunk = self.trainer.act_chunk(packet.z_rl, state_vec, ref_action_chunk)
                action = self._clip_final_action(actor_chunk[step_idx])
                executed_action_chunk = actor_chunk.copy()
                residual_mm = 1000.0 * float(np.linalg.norm(action[:3] - ref_action[:3]))
                source = "rlt_actor"
            else:
                action = self._clip_final_action(ref_action)
                executed_action_chunk = ref_action_chunk.copy()
                if self._actor_ready() and self.gate_phase == 1:
                    source = "vla_gate_exit"
                else:
                    source = "vla_gate0" if self._actor_ready() and self.gate_phase != 1 else "vla_warmup"
            target_pos, gripper = self._smooth_final_action(action)
            executed = np.asarray([target_pos[0], target_pos[1], target_pos[2], gripper], dtype=np.float32)
            if executed_action_chunk is not None:
                executed_action_chunk[step_idx] = executed

        if self.execute:
            pose_msg = Float64MultiArray()
            pose_msg.data = make_ik_target(target_pos, self.fixed_quat)
            self.ik_target_pub.publish(pose_msg)
            command_msg = Float64MultiArray()
            command_msg.data = make_vr_command({"enable": False, "gripper": gripper, "home": False})
            self.command_pub.publish(command_msg)

        if self._should_record_gate_sample(packet):
            if ref_action_chunk is None:
                ref_action_chunk = self._fixed_action_chunk(packet.ref_action_chunk)
            if executed_action_chunk is None:
                executed_action_chunk = ref_action_chunk.copy()
                step_idx = min(max(int(packet.step_index), 0), executed_action_chunk.shape[0] - 1)
                executed_action_chunk[step_idx] = executed
            self._record_transition_seed(
                stamp=stamp,
                z_rl=packet.z_rl.astype(np.float32),
                state_vec=state_vec,
                ref_action=ref_action_chunk.astype(np.float32),
                executed_action=executed_action_chunk.astype(np.float32),
                is_intervention=source == "vr_override",
                source=source,
                gate_prob=float(self.gate_prob if self.gate_prob is not None else packet.gate_prob),
            )

        train = self.last_train_info or {}
        actor_train = self.last_actor_train_info or train
        self._log_info(
            f"{'pub' if self.execute else 'dry'} src={source} rlt={int(rlt_active)} "
            f"pend={int(self.save_pending)} r={self.pending_reward} staged={len(self.episode_transitions)} "
            f"p={fmt_vec(target_pos)} grip={gripper:.3f} res={residual_mm:.1f}mm "
            f"buf={len(self.online_buffer)}/{len(self.intervention_buffer)} "
            f"upd={int(self.trainer.update_step)} "
            f"c={train.get('critic_loss', 0.0):.3f} act={actor_train.get('actor_loss', 0.0):.3f} "
            f"bc={actor_train.get('bc_loss', 0.0):.3f} q={actor_train.get('actor_q_loss', 0.0):.3f} "
            f"au={int(train.get('actor_updated', 0.0))}"
        )

    def _actor_ready(self) -> bool:
        return (
            bool(self.args.rlt_enable_actor)
            and len(self.online_buffer) >= int(self.args.rlt_warmup_steps)
            and int(self.trainer.update_step) >= int(self.args.rlt_min_actor_updates)
        )

    def _fixed_action_chunk(self, chunk: np.ndarray) -> np.ndarray:
        return self.trainer._numpy_chunk(chunk)

    def _clip_final_action(self, action: np.ndarray) -> np.ndarray:
        out = np.asarray(action, dtype=np.float32).reshape(4).copy()
        out[:3] = self.safety.clamp_impedance_workspace(out[:3])
        if np.isfinite(float(self.args.min_action_z)):
            out[2] = max(float(out[2]), float(self.args.min_action_z))
        out[3] = float(np.clip(out[3], 0.0, self.args.gripper_max))
        return out

    def _smooth_final_action(self, action: np.ndarray) -> tuple[np.ndarray, float]:
        action = self._clip_final_action(action)
        pos = self._smooth_position(action[:3])
        gripper = float(action[3])
        if self.last_published_gripper is not None:
            alpha_g = float(np.clip(self.args.action_gripper_filter_alpha, 0.0, 1.0))
            gripper = alpha_g * gripper + (1.0 - alpha_g) * self.last_published_gripper
        gripper = float(np.clip(gripper, 0.0, self.args.gripper_max))
        self.last_published_gripper = gripper
        return pos.astype(np.float32), gripper

    def _record_transition_seed(
        self,
        *,
        stamp: float,
        z_rl: np.ndarray,
        state_vec: np.ndarray,
        ref_action: np.ndarray,
        executed_action: np.ndarray,
        is_intervention: bool,
        source: str,
        gate_prob: float,
    ) -> None:
        with self.episode_lock:
            seed = {
                "timestamp": float(stamp),
                "z_rl": z_rl.astype(np.float32),
                "state": state_vec.astype(np.float32),
                "ref_action": ref_action.astype(np.float32),
                "action": executed_action.astype(np.float32),
                "is_intervention": float(is_intervention),
                "source": source,
                "gate_prob": float(gate_prob),
            }
            if self.pending_seed is not None:
                transition = self._make_transition(self.pending_seed, seed, reward=0.0, done=0.0)
                self._insert_transition(transition, terminal=False)
            self.pending_seed = seed

    def _make_transition(
        self,
        current: dict[str, Any],
        nxt: dict[str, Any],
        *,
        reward: float,
        done: float,
    ) -> HILTransition:
        return HILTransition(
            z_rl=np.asarray(current["z_rl"], dtype=np.float32),
            state=np.asarray(current["state"], dtype=np.float32),
            ref_action=np.asarray(current["ref_action"], dtype=np.float32),
            action=np.asarray(current["action"], dtype=np.float32),
            reward=float(reward),
            next_z_rl=np.asarray(nxt["z_rl"], dtype=np.float32),
            next_state=np.asarray(nxt["state"], dtype=np.float32),
            next_ref_action=np.asarray(nxt["ref_action"], dtype=np.float32),
            done=float(done),
            is_intervention=float(current["is_intervention"]),
        )

    def _insert_transition(self, transition: HILTransition, *, terminal: bool) -> None:
        self.episode_transitions.append(transition)
        sample = transition.to_npz_dict()
        sample["terminal"] = float(terminal)
        self.current_samples.append(sample)

    def _commit_episode_transitions(self) -> None:
        for transition in self.episode_transitions:
            self.online_buffer.insert(transition)
            self.inserted_transitions += 1
            if transition.is_intervention > 0.5:
                self.intervention_buffer.insert(transition)
                self.intervention_transitions += 1
            if self.training_disabled_after_error:
                continue
            for _ in range(max(0, int(self.args.rlt_updates_per_step))):
                try:
                    with self.trainer_lock:
                        info = self.trainer.update(self.online_buffer, self.intervention_buffer)
                except RuntimeError as exc:
                    self.training_disabled_after_error = True
                    self.node.get_logger().error(
                        f"Stage2 online update failed; disabling further updates for this run. "
                        f"Episode data remains saved/inserted. error={exc}"
                    )
                    break
                if info is not None:
                    self.last_train_info = info
                    if info.get("actor_updated", 0.0) > 0.5:
                        self.last_actor_train_info = info

    def _finalize_pending_terminal(self) -> None:
        if self.pending_seed is None:
            return
        terminal = self._make_transition(
            self.pending_seed,
            self.pending_seed,
            reward=float(self.pending_reward),
            done=1.0,
        )
        self._insert_transition(terminal, terminal=True)
        self.pending_seed = None

    def _confirm_save(self) -> None:
        with self.episode_lock:
            self._confirm_save_locked()

    def _confirm_save_locked(self) -> None:
        if not self.current_samples and self.pending_seed is None:
            self.node.get_logger().warn("B pressed, but no Stage2 transitions are buffered.")
            return
        if self.pending_reward is None:
            self.node.get_logger().warn("Select terminal reward first: Y=success(1), X=failure(0).")
            return
        if not self.save_pending:
            self.save_pending = True
            self.inference_enabled = False
            self._reset_policy_tracking(clear_current=True, reset_policy=False)
            self._finalize_pending_terminal()
            self.node.get_logger().info(
                f"Save pending: transitions={len(self.current_samples)} final_reward={self.pending_reward:.0f}. "
                "Press B again to finalize, or left trigger to discard."
            )
            return

        self._finalize_pending_terminal()
        self._commit_episode_transitions()

        ep = self.saved_episodes
        final_reward = float(self.pending_reward)
        transition_count = len(self.current_samples)
        ep_dir = self.output_dir / f"episode_{ep:06d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        keys = [
            "z_rl",
            "state",
            "ref_action",
            "action",
            "reward",
            "next_z_rl",
            "next_state",
            "next_ref_action",
            "done",
            "is_intervention",
            "terminal",
        ]
        arrays = {key: np.asarray([sample[key] for sample in self.current_samples], dtype=np.float32) for key in keys}
        np.savez_compressed(ep_dir / "episode_transitions.npz", **arrays)
        meta = {
            "episode": ep,
            "transitions": transition_count,
            "reward": final_reward,
            "online_buffer_size": len(self.online_buffer),
            "intervention_buffer_size": len(self.intervention_buffer),
            "inserted_transitions": self.inserted_transitions,
            "intervention_transitions": self.intervention_transitions,
            "policy_path": str(self.policy_path),
            "stage1_checkpoint": str(self.stage1_checkpoint),
            "gate_checkpoint": str(self.gate_checkpoint),
        }
        (ep_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        with (self.output_dir / "episodes.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(meta, separators=(",", ":")) + "\n")

        self.saved_episodes += 1
        if self.args.rlt_save_every_episodes > 0 and self.saved_episodes % int(self.args.rlt_save_every_episodes) == 0:
            self._save_stage2_checkpoint()
        train = self.last_train_info or {}
        actor_train = self.last_actor_train_info or train
        self.node.get_logger().info(
            f"Finalized Stage2 episode {ep:06d}: transitions={transition_count} "
            f"reward={final_reward:.0f} online={len(self.online_buffer)} intvn={len(self.intervention_buffer)} "
            f"upd={int(self.trainer.update_step)} c={train.get('critic_loss', 0.0):.3f} "
            f"act={actor_train.get('actor_loss', 0.0):.3f} bc={actor_train.get('bc_loss', 0.0):.3f} "
            f"q={actor_train.get('actor_q_loss', 0.0):.3f} au={int(train.get('actor_updated', 0.0))}"
        )
        self.current_samples = []
        self.episode_transitions = []
        self.pending_reward = None
        self.save_pending = False
        self.ignore_b_until = self.now_sec() + float(self.args.save_button_cooldown_s)

    def _save_stage2_checkpoint(self) -> None:
        ckpt_path = self.output_dir / "checkpoints" / f"stage2_ep{self.saved_episodes:06d}.pt"
        with self.trainer_lock:
            self.trainer.save(
                ckpt_path,
                metadata={
                    "saved_episodes": self.saved_episodes,
                    "online_buffer_size": len(self.online_buffer),
                    "intervention_buffer_size": len(self.intervention_buffer),
                },
            )
            self.trainer.save(self.output_dir / "checkpoints" / "last.pt")
        if self.args.rlt_snapshot_buffers:
            self.online_buffer.save_npz(self.output_dir / "buffers" / "online_buffer_latest.npz")
            self.intervention_buffer.save_npz(self.output_dir / "buffers" / "intervention_buffer_latest.npz")


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
    node = rclpy.create_node("ur3e_hil_serl_stage2")
    runner: HILSERLStage2Runner | None = None
    try:
        run_pre_model_startup_sequence(node, args)
        runner = HILSERLStage2Runner(node, args)
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        try:
            executor.spin()
        finally:
            executor.shutdown()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if runner is not None:
            runner._save_stage2_checkpoint()
            runner.close()
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
