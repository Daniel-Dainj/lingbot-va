#!/usr/bin/env bash
set -euo pipefail

# 在正式训练前做一次轻量检查：
# 确认 latent、empty_emb、归一化统计都已就绪，
# 并让 LingBot-VA 的数据加载器实际取一个 sample。

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

DATASET_ROOT="${DATASET_ROOT:-${ROOT_DIR}/data/rokae_demo/260514/lerobot/pick_up_the_workpiece}"
STATS_PATH="${STATS_PATH:-${DATASET_ROOT}/meta/lingbot_action_stats.json}"

if [[ ! -f "${DATASET_ROOT}/empty_emb.pt" ]]; then
  echo "缺少 ${DATASET_ROOT}/empty_emb.pt，请先执行 script/02.extract_rokae_latents.sh"
  exit 1
fi

if [[ ! -f "${STATS_PATH}" ]]; then
  echo "缺少 ${STATS_PATH}，请先执行 script/01.prepare_rokae_lerobot.sh"
  exit 1
fi

if [[ -x "${PYTHON_BIN}" ]]; then
  RUNNER=("${PYTHON_BIN}")
else
  RUNNER=(uv run python)
fi

"${RUNNER[@]}" "${ROOT_DIR}/script/check_rokae_dataset.py" \
  --dataset-root "${DATASET_ROOT}" \
  --norm-stat-path "${STATS_PATH}"
