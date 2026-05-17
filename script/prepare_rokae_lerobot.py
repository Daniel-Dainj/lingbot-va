#!/usr/bin/env python3
# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Convert ROKAE ER7Pro + Robotiq 2F-85 HDF5 episodes to a LeRobot dataset."""

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from scipy.spatial.transform import Rotation as R


USED_ACTION_CHANNEL_IDS = list(range(0, 7)) + list(range(14, 21)) + [28]
ACTION_FEATURE_NAMES = [
    "left_eef_x",
    "left_eef_y",
    "left_eef_z",
    "left_eef_qx",
    "left_eef_qy",
    "left_eef_qz",
    "left_eef_qw",
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
    "left_joint_4",
    "left_joint_5",
    "left_joint_6",
    "left_joint_7",
    "left_gripper",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True, help="Directory containing episode_*.h5 files")
    parser.add_argument("--output-root", type=Path, required=True, help="Output LeRobot dataset root")
    parser.add_argument(
        "--task-text",
        type=str,
        default="pick up the workpiece",
        help="Language instruction written to tasks and action_config",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="rokae_er7pro_robotiq_2f85",
        help="Robot type stored in LeRobot metadata",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap used for smoke tests",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete the existing output dataset root before recreating it",
    )
    return parser.parse_args()


def pose6d_to_pose7d(pose6d: np.ndarray) -> np.ndarray:
    xyz = pose6d[:, :3].astype(np.float32)
    euler_xyz = pose6d[:, 3:6]
    quat_xyzw = R.from_euler("xyz", euler_xyz).as_quat().astype(np.float32)
    return np.concatenate([xyz, quat_xyzw], axis=1)


def build_state_vector(h5_file: h5py.File) -> np.ndarray:
    eef_pose7 = pose6d_to_pose7d(h5_file["observation/end_effector_pose_6dof"][:])
    joint_position = h5_file["observation/joint_position"][:].astype(np.float32)
    gripper_position = h5_file["observation/gripper_position"][:].astype(np.float32)
    return np.concatenate([eef_pose7, joint_position, gripper_position], axis=1)


def build_next_observation_action(state: np.ndarray) -> np.ndarray:
    return np.concatenate([state[1:], state[-1:]], axis=0).astype(np.float32)


def get_dataset_features(frame_shape):
    return {
        "action": {
            "dtype": "float32",
            "shape": (len(ACTION_FEATURE_NAMES),),
            "names": ACTION_FEATURE_NAMES,
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(ACTION_FEATURE_NAMES),),
            "names": ACTION_FEATURE_NAMES,
        },
        "observation.images.exterior_left": {
            "dtype": "video",
            "shape": frame_shape,
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": frame_shape,
            "names": ["height", "width", "channels"],
        },
    }


def rewrite_episodes_jsonl(episodes_path: Path, task_text: str):
    rewritten = []
    with episodes_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            item["tasks"] = [task_text]
            item["action_config"] = [
                {
                    "start_frame": 0,
                    "end_frame": item["length"],
                    "action_text": task_text,
                }
            ]
            rewritten.append(item)

    with episodes_path.open("w", encoding="utf-8") as f:
        for item in rewritten:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_lingbot_stats(output_root: Path, action_values: np.ndarray, task_text: str):
    q01_raw = np.quantile(action_values, 0.01, axis=0).astype(np.float32)
    q99_raw = np.quantile(action_values, 0.99, axis=0).astype(np.float32)

    q01 = np.zeros(30, dtype=np.float32)
    q99 = np.ones(30, dtype=np.float32)
    for src_idx, dst_idx in enumerate(USED_ACTION_CHANNEL_IDS):
        q01[dst_idx] = q01_raw[src_idx]
        q99[dst_idx] = q99_raw[src_idx]

    stats = {
        "task_text": task_text,
        "action_feature_names": ACTION_FEATURE_NAMES,
        "used_action_channel_ids": USED_ACTION_CHANNEL_IDS,
        "q01": q01.tolist(),
        "q99": q99.tolist(),
    }

    stats_path = output_root / "meta" / "lingbot_action_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def write_source_manifest(output_root: Path, source_items: list[dict]):
    manifest_path = output_root / "meta" / "rokae_source_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(source_items, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    raw_root = args.raw_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not raw_root.exists():
        raise FileNotFoundError(f"Raw root does not exist: {raw_root}")

    if output_root.exists():
        if not args.force:
            raise FileExistsError(
                f"Output dataset already exists: {output_root}. "
                "Use --force to remove it before conversion."
            )
        shutil.rmtree(output_root)

    episode_files = sorted(raw_root.glob("episode_*.h5"))
    if args.max_episodes is not None:
        episode_files = episode_files[: args.max_episodes]

    if not episode_files:
        raise FileNotFoundError(f"No episode_*.h5 files found under {raw_root}")

    with h5py.File(episode_files[0], "r") as probe_file:
        frame_shape = tuple(probe_file["observation/exterior_image_1_left"][0].shape)
        fps = int(probe_file.attrs["fps"])

    dataset = LeRobotDataset.create(
        repo_id=output_root.name,
        root=output_root,
        fps=fps,
        robot_type=args.robot_type,
        features=get_dataset_features(frame_shape),
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=0,
        batch_encoding_size=1,
    )

    all_actions = []
    source_manifest = []
    for dataset_episode_index, episode_file in enumerate(episode_files):
        with h5py.File(episode_file, "r") as h5_file:
            state = build_state_vector(h5_file)
            action = build_next_observation_action(state)
            left_rgb = h5_file["observation/exterior_image_1_left"][:]
            wrist_rgb = h5_file["observation/wrist_image"][:]

            if not (len(state) == len(action) == len(left_rgb) == len(wrist_rgb)):
                raise ValueError(f"Inconsistent frame count in {episode_file}")

            for frame_idx in range(len(state)):
                frame = {
                    "observation.state": state[frame_idx],
                    "action": action[frame_idx],
                    "observation.images.exterior_left": left_rgb[frame_idx],
                    "observation.images.wrist": wrist_rgb[frame_idx],
                }
                dataset.add_frame(frame, task=args.task_text, timestamp=frame_idx / fps)

            dataset.save_episode()
            all_actions.append(action)
            source_manifest.append(
                {
                    "episode_index": dataset_episode_index,
                    "episode_file": episode_file.name,
                    "num_frames": int(len(state)),
                    "fps": fps,
                    "raw_instruction": str(h5_file.attrs.get("instruction", "")),
                }
            )

    rewrite_episodes_jsonl(output_root / "meta" / "episodes.jsonl", args.task_text)
    write_lingbot_stats(output_root, np.concatenate(all_actions, axis=0), args.task_text)
    write_source_manifest(output_root, source_manifest)

    print(f"Converted {len(episode_files)} episodes to {output_root}")
    print(f"Saved LingBot normalization stats to {output_root / 'meta' / 'lingbot_action_stats.json'}")


if __name__ == "__main__":
    main()
