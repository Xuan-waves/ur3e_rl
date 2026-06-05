from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


KEEP = np.array([0, 1, 2, 6, 7], dtype=int)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a copy of a LeRobot eepose dataset with fixed rotvec fields removed."
    )
    parser.add_argument("dataset", type=Path, help="Source LeRobot dataset directory.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output dataset directory. Default: <dataset>_no_rotvec",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace output if it already exists.")
    args = parser.parse_args()

    src = args.dataset.expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    dst = args.output.expanduser().resolve() if args.output else src.with_name(src.name + "_no_rotvec")
    if dst.exists():
        if not args.overwrite:
            raise FileExistsError(f"{dst} already exists. Use --overwrite to replace it.")
        shutil.rmtree(dst)

    shutil.copytree(src, dst)
    _rewrite_data(dst)
    _rewrite_info(dst)
    _rewrite_stats_json(dst)
    _rewrite_episode_stats(dst)
    _rewrite_sync_reports(dst)
    print(f"Wrote no-rotvec dataset: {dst}")
    return 0


def _rewrite_data(root: Path) -> None:
    for path in sorted((root / "data").glob("chunk-*/*.parquet")):
        df = pd.read_parquet(path)
        for key in ("observation.state", "action"):
            df[key] = df[key].map(_strip_vec)
        df.to_parquet(path, index=False)


def _rewrite_info(root: Path) -> None:
    path = root / "meta" / "info.json"
    info = json.loads(path.read_text(encoding="utf-8"))
    features = info.get("features", {})
    if "observation.state" in features:
        features["observation.state"]["shape"] = [5]
        features["observation.state"]["names"] = {"motors": ["tcp_x", "tcp_y", "tcp_z", "gripper", "RL_mark"]}
    if "action" in features:
        names = features["action"].get("names", {}).get("motors", [])
        pos_names = list(names[:3]) if len(names) >= 3 else ["target_tcp_x", "target_tcp_y", "target_tcp_z"]
        gripper_name = names[6] if len(names) >= 7 else "target_gripper"
        rl_name = names[7] if len(names) >= 8 else "RL_mark"
        features["action"]["shape"] = [5]
        features["action"]["names"] = {"motors": pos_names + [gripper_name, rl_name]}
    info["features"] = features
    info["fixed_orientation_rotvec_removed"] = True
    path.write_text(json.dumps(info, indent=4), encoding="utf-8")


def _rewrite_stats_json(root: Path) -> None:
    path = root / "meta" / "stats.json"
    stats = json.loads(path.read_text(encoding="utf-8"))
    for key in ("observation.state", "action"):
        if key not in stats:
            continue
        for stat_key, value in list(stats[key].items()):
            stats[key][stat_key] = _strip_stat_value(value)
    path.write_text(json.dumps(stats, indent=4), encoding="utf-8")


def _rewrite_episode_stats(root: Path) -> None:
    for path in sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet")):
        df = pd.read_parquet(path)
        for prefix in ("stats/observation.state", "stats/action"):
            for col in [c for c in df.columns if c.startswith(prefix + "/")]:
                df[col] = df[col].map(_strip_stat_value)
        df.to_parquet(path, index=False)


def _rewrite_sync_reports(root: Path) -> None:
    for path in sorted((root / "meta").glob("sync_report_episode_*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        report["fixed_orientation_rotvec_removed"] = True
        action_repr = report.get("action_representation", {})
        action_repr["orientation"] = "removed_fixed_orientation"
        action_repr["removed_rotvec_indices"] = [3, 4, 5]
        report["action_representation"] = action_repr
        representation = report.get("representation", {})
        representation["state_mode"] = "eepose_xyz_gripper"
        representation["action_mode"] = "eepose_xyz_gripper"
        representation["action_orientation_source"] = "removed_fixed_orientation"
        representation["removed_rotvec_indices"] = [3, 4, 5]
        report["representation"] = representation
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _strip_vec(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (8,):
        raise ValueError(f"Expected vector shape (8,), got {arr.shape}")
    return arr[KEEP].astype(np.float32)


def _strip_stat_value(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.shape == (8,):
        return arr[KEEP].tolist()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
