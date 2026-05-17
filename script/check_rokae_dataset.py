#!/usr/bin/env python3
# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Smoke-test the converted ROKAE dataset with LingBot-VA's dataset loader."""

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path

from wan_va.configs import VA_CONFIGS
from wan_va.dataset.lerobot_latent_dataset import LatentLeRobotDataset


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=str, required=True, help="Converted LeRobot dataset root")
    parser.add_argument(
        "--norm-stat-path",
        type=str,
        default=None,
        help="Optional q01/q99 stats json. Defaults to dataset_root/meta/lingbot_action_stats.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    hf_home = Path(os.getenv("HF_HOME", Path.cwd() / ".hf")).resolve()
    datasets_cache = Path(os.getenv("HF_DATASETS_CACHE", hf_home / "datasets")).resolve()
    hub_cache = Path(os.getenv("HUGGINGFACE_HUB_CACHE", hf_home / "hub")).resolve()
    hf_home.mkdir(parents=True, exist_ok=True)
    datasets_cache.mkdir(parents=True, exist_ok=True)
    hub_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_DATASETS_CACHE", str(datasets_cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub_cache))

    config = deepcopy(VA_CONFIGS["rokae_train"])
    config.dataset_path = os.path.abspath(args.dataset_root)
    config.empty_emb_path = os.path.join(config.dataset_path, "empty_emb.pt")

    norm_stat_path = args.norm_stat_path or os.path.join(config.dataset_path, "meta", "lingbot_action_stats.json")
    with open(norm_stat_path, "r", encoding="utf-8") as f:
        config.norm_stat = json.load(f)

    dataset = LatentLeRobotDataset(repo_id=config.dataset_path, config=config)
    sample = dataset[0]

    print(f"dataset_len={len(dataset)}")
    for key, value in sample.items():
        if hasattr(value, "shape"):
            print(f"{key}: shape={tuple(value.shape)} dtype={getattr(value, 'dtype', 'n/a')}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
