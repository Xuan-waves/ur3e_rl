#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rlt_train.rollout_rlt_no_vr import (  # noqa: E402
    RLTNoVRRollout,
    _resolve,
    parse_args as parse_base_args,
)
from scripts.rollout_smolvla_no_rotvec.rollout_ur3e_smolvla_no_rotvec_sync import (  # noqa: E402
    run_pre_model_startup_sequence,
)
from vr_servoj_test.rollout.rollout_ur3e_servoj_smolvla import resolve_policy_path  # noqa: E402


class RLTNoVRRTCRollout(RLTNoVRRollout):
    """No-VR RLT rollout with a background chunk producer.

    The base no-VR rollout is deliberately synchronous at the inference boundary:
    every timer tick may launch a chunk inference.  This RTC variant keeps the ROS
    publish path light by storing the freshest aligned observation and letting a
    single worker thread refresh the action queue when it is almost empty.
    """

    def __init__(self, node, args: argparse.Namespace) -> None:
        self.latest_packet_lock = threading.Lock()
        self.latest_packet = None
        self.latest_packet_generation: int | None = None
        self.worker_stop = threading.Event()
        self.worker_thread: threading.Thread | None = None
        self.rtc_last_packet_stamp: float | None = None
        self.rtc_infer_count = max(1, int(args.rtc_infer_count))
        self.rtc_queue_refill_threshold = max(0, int(args.rtc_queue_refill_threshold))
        self.rtc_idle_sleep_s = max(0.001, float(args.rtc_idle_sleep_s))
        self.rtc_replace_queue_on_infer = bool(args.rtc_replace_queue_on_infer)
        super().__init__(node, args)
        self.rtc_infer_count = max(self.rtc_infer_count, int(self.trainer.action_chunk_steps))
        self.worker_thread = threading.Thread(target=self._rtc_worker_loop, daemon=True)
        self.worker_thread.start()
        self.node.get_logger().info(
            f"RLT no-VR RTC worker enabled. infer_count={self.rtc_infer_count}, "
            f"refill_threshold={self.rtc_queue_refill_threshold}, "
            f"replace_queue={self.rtc_replace_queue_on_infer}"
        )

    def close(self) -> None:
        self.worker_stop.set()
        if self.worker_thread is not None:
            self.worker_thread.join(timeout=2.0)
        super().close()

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
        stamp = self._take_reference_stamp()
        if stamp is None:
            self._log_wait("waiting for reference image")
            return
        packet = self._build_observation(stamp)
        if packet is None:
            return
        with self.latest_packet_lock:
            self.latest_packet = packet
            self.latest_packet_generation = self.rollout_generation

    def _drain_actions(self) -> None:
        super()._drain_actions()
        with self.latest_packet_lock:
            self.latest_packet = None
            self.latest_packet_generation = None
        self.rtc_last_packet_stamp = None

    def _rtc_worker_loop(self) -> None:
        while not self.worker_stop.is_set():
            if not self.inference_enabled:
                time.sleep(self.rtc_idle_sleep_s)
                continue
            if self.args.block_model_during_home and self.now_sec() < self.home_block_until:
                time.sleep(self.rtc_idle_sleep_s)
                continue
            with self.action_lock:
                queued = len(self.action_queue) + (1 if self.current_action is not None else 0)
            if queued > self.rtc_queue_refill_threshold:
                time.sleep(self.rtc_idle_sleep_s)
                continue
            with self.latest_packet_lock:
                packet = self.latest_packet
                generation = self.latest_packet_generation
            if packet is None or generation is None or generation != self.rollout_generation:
                time.sleep(self.rtc_idle_sleep_s)
                continue
            if self.infer_busy:
                time.sleep(self.rtc_idle_sleep_s)
                continue
            self.infer_busy = True
            old_horizon = self.args.execution_horizon
            old_replace = self.args.replace_queue_on_infer
            self.args.execution_horizon = self.rtc_infer_count
            self.args.replace_queue_on_infer = self.rtc_replace_queue_on_infer
            try:
                self._run_inference(packet, generation)
                if generation == self.rollout_generation:
                    self.rtc_last_packet_stamp = float(packet.stamp)
            finally:
                self.args.execution_horizon = old_horizon
                self.args.replace_queue_on_infer = old_replace
                # _run_inference normally clears infer_busy.  Keep this guard for
                # exceptions that occur after it restores the flag internally.
                self.infer_busy = False
            time.sleep(self.rtc_idle_sleep_s)


def parse_args() -> argparse.Namespace:
    rtc_parser = argparse.ArgumentParser(add_help=False)
    rtc_parser.add_argument(
        "--rtc-infer-count",
        type=int,
        default=10,
        help="Number of VLA actions inferred per RTC worker refresh. Clamped to at least the RLT actor chunk size.",
    )
    rtc_parser.add_argument(
        "--rtc-queue-refill-threshold",
        type=int,
        default=3,
        help="Refresh the action queue when queued actions are at or below this count.",
    )
    rtc_parser.add_argument(
        "--rtc-idle-sleep-s",
        type=float,
        default=0.005,
        help="Worker sleep interval while waiting for fresh observations or queue space.",
    )
    rtc_parser.add_argument(
        "--rtc-replace-queue-on-infer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replace pending queued actions whenever the RTC worker finishes a fresh chunk.",
    )
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print("RTC-specific options:")
        rtc_parser.print_help()
        print("\nBase rollout options:")
    rtc_args, remaining = rtc_parser.parse_known_args()

    old_argv = sys.argv[:]
    sys.argv = [sys.argv[0], *remaining]
    try:
        args = parse_base_args()
    finally:
        sys.argv = old_argv
    args.rtc_infer_count = rtc_args.rtc_infer_count
    args.rtc_queue_refill_threshold = rtc_args.rtc_queue_refill_threshold
    args.rtc_idle_sleep_s = rtc_args.rtc_idle_sleep_s
    args.rtc_replace_queue_on_infer = rtc_args.rtc_replace_queue_on_infer
    return args


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
    node = rclpy.create_node("ur3e_rlt_no_vr_rtc_rollout")
    rollout: RLTNoVRRTCRollout | None = None
    try:
        run_pre_model_startup_sequence(node, args)
        rollout = RLTNoVRRTCRollout(node, args)
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
