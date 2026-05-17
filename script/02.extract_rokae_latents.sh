#!/usr/bin/env bash
set -euo pipefail

# 用 LingBot-VA 基座里的 tokenizer / text_encoder / VAE
# 为每个 episode 生成：
# 1. empty_emb.pt
# 2. 每路相机对应的 Wan2.2 latent 文件

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
MODEL_ROOT="${MODEL_ROOT:-${ROOT_DIR}/checkpoints/lingbot-va/lingbot-va-base}"
DEVICE="${DEVICE:-cuda:0}"
HEIGHT="${HEIGHT:-256}"
WIDTH="${WIDTH:-320}"

if [[ -x "${PYTHON_BIN}" ]]; then
  RUNNER=("${PYTHON_BIN}")
else
  RUNNER=(uv run python)
fi

"${RUNNER[@]}" "${ROOT_DIR}/script/extract_rokae_latents.py" \
  --raw-root "${RAW_ROOT}" \
  --dataset-root "${DATASET_ROOT}" \
  --model-root "${MODEL_ROOT}" \
  --device "${DEVICE}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --skip-existing
