#!/usr/bin/env bash
set -euo pipefail

# 将原始真机 H5 数据转换成 LeRobot 数据集。
# 这里会把 action 明确定义为“下一帧观测”：
# [下一帧末端 xyzquat(7) + 下一帧关节角(7) + 下一帧夹爪(1)]。

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
UV_CACHE_DIR="${UV_CACHE_DIR:-${ROOT_DIR}/.uv-cache}"
mkdir -p "${UV_CACHE_DIR}"
export UV_CACHE_DIR
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
HF_HOME="${HF_HOME:-${ROOT_DIR}/.hf}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_DATASETS_CACHE}" "${HUGGINGFACE_HUB_CACHE}"
export HF_HOME HF_DATASETS_CACHE HUGGINGFACE_HUB_CACHE

RAW_ROOT="${RAW_ROOT:-${ROOT_DIR}/data/rokae_demo/260514/raw_episodes}"
DATASET_ROOT="${DATASET_ROOT:-${ROOT_DIR}/data/rokae_demo/260514/lerobot/pick_up_the_workpiece}"
TASK_TEXT="${TASK_TEXT:-pick up the workpiece}"
ROBOT_TYPE="${ROBOT_TYPE:-rokae_er7pro_robotiq_2f85}"

if [[ -x "${PYTHON_BIN}" ]]; then
  RUNNER=("${PYTHON_BIN}")
else
  RUNNER=(uv run python)
fi

"${RUNNER[@]}" "${ROOT_DIR}/script/prepare_rokae_lerobot.py" \
  --raw-root "${RAW_ROOT}" \
  --output-root "${DATASET_ROOT}" \
  --task-text "${TASK_TEXT}" \
  --robot-type "${ROBOT_TYPE}"
