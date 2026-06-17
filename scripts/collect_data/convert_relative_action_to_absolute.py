from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ACTION_STAT_KEYS = ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a copy of a LeRobot dataset where action[0:3] is converted "
            "from TCP delta to absolute target TCP position."
        )
    )
    parser.add_argument("dataset", type=Path, help="Source LeRobot dataset directory.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output dataset directory. Default: <dataset>_absolute_action",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace output if it already exists.")
    args = parser.parse_args()

    src = args.dataset.expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    dst = args.output.expanduser().resolve() if args.output else src.with_name(src.name + "_absolute_action")
    if dst.exists():
        if not args.overwrite:
            raise FileExistsError(f"{dst} already exists. Use --overwrite to replace it.")
        shutil.rmtree(dst)

    _assert_relative_dataset(src)
    shutil.copytree(src, dst)
    action_by_episode = _rewrite_data(dst)
    global_action = np.concatenate(list(action_by_episode.values()), axis=0)
    _rewrite_info(dst)
    _rewrite_stats_json(dst, global_action)
    _rewrite_episode_stats(dst, action_by_episode)
    _rewrite_sync_reports(dst)
    print(f"Wrote absolute-action dataset: {dst}")
    print(f"Converted frames: {global_action.shape[0]}")
    print(f"action_abs min={_fmt(global_action.min(axis=0))} max={_fmt(global_action.max(axis=0))}")
    return 0


def _assert_relative_dataset(root: Path) -> None:
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    names = (
        info.get("features", {})
        .get("action", {})
        .get("names", {})
        .get("motors", [])
    )
    if len(names) < 4:
        raise ValueError(f"Could not read action names from {info_path}")
    if not all(str(name).startswith("delta_tcp_") for name in names[:3]):
        raise ValueError(f"Dataset does not look relative-action based on action names: {names}")


def _rewrite_data(root: Path) -> dict[int, np.ndarray]:
    action_by_episode: dict[int, list[np.ndarray]] = {}
    for path in sorted((root / "data").glob("chunk-*/*.parquet")):
        df = pd.read_parquet(path)
        if "observation.state" not in df or "action" not in df:
            raise KeyError(f"{path} missing observation.state or action")

        new_actions: list[np.ndarray] = []
        for state_value, action_value, episode_value in zip(
            df["observation.state"], df["action"], df["episode_index"], strict=True
        ):
            state = np.asarray(state_value, dtype=np.float32).reshape(-1)
            action = np.asarray(action_value, dtype=np.float32).reshape(-1).copy()
            if state.size < 3 or action.size < 4:
                raise ValueError(f"Bad state/action shape in {path}: state={state.shape}, action={action.shape}")
            action[:3] = state[:3] + action[:3]
            new_actions.append(action.astype(np.float32))
            action_by_episode.setdefault(int(episode_value), []).append(action.astype(np.float32))

        df["action"] = new_actions
        df.to_parquet(path, index=False)

    return {episode: np.stack(actions, axis=0) for episode, actions in action_by_episode.items()}


def _rewrite_info(root: Path) -> None:
    path = root / "meta" / "info.json"
    info = json.loads(path.read_text(encoding="utf-8"))
    features = info.get("features", {})
    if "action" in features:
        features["action"]["names"] = {
            "motors": ["target_tcp_x", "target_tcp_y", "target_tcp_z", "target_gripper"]
        }
    info["features"] = features
    info["action_position_mode"] = "absolute"
    path.write_text(json.dumps(info, indent=2), encoding="utf-8")


def _rewrite_stats_json(root: Path, action: np.ndarray) -> None:
    path = root / "meta" / "stats.json"
    stats = json.loads(path.read_text(encoding="utf-8"))
    stats["action"] = _stats_dict(action)
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def _rewrite_episode_stats(root: Path, action_by_episode: dict[int, np.ndarray]) -> None:
    for path in sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet")):
        df = pd.read_parquet(path)
        if "episode_index" not in df:
            continue
        for row_idx, episode in enumerate(df["episode_index"].tolist()):
            episode_action = action_by_episode.get(int(episode))
            if episode_action is None:
                continue
            stats = _stats_dict(episode_action)
            for stat_key in ACTION_STAT_KEYS:
                column = f"stats/action/{stat_key}"
                if column in df.columns:
                    df.at[row_idx, column] = stats[stat_key]
            count_column = "stats/action/count"
            if count_column in df.columns:
                df.at[row_idx, count_column] = stats["count"]
        df.to_parquet(path, index=False)


def _rewrite_sync_reports(root: Path) -> None:
    for path in sorted((root / "meta").glob("sync_report_episode_*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        action_repr = report.get("action_representation", {})
        action_repr["position"] = "absolute"
        action_repr.pop("position_relative_to", None)
        action_repr["converted_from"] = "relative"
        action_repr["conversion"] = "action_abs[0:3] = observation.state[0:3] + action_rel[0:3]"
        report["action_representation"] = action_repr

        representation = report.get("representation", {})
        representation["ee_action_position_mode"] = "absolute"
        representation["action_position_converted_from"] = "relative"
        report["representation"] = representation
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _stats_dict(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(values.shape[0])],
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def _fmt(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(v): .5f}" for v in values) + "]"


if __name__ == "__main__":
    raise SystemExit(main())
