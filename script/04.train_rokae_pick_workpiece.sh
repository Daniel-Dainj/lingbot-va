#!/usr/bin/env bash
set -euo pipefail

# 启动从 lingbot-va-base 到 rokae 单任务数据的微调。
# 默认每 500 个 optimizer step 保存一次 checkpoint。
# NUM_STEPS 表示“训练总步数”，不是“追加训练步数”。

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
MODEL_ROOT="${MODEL_ROOT:-${ROOT_DIR}/checkpoints/lingbot-va/lingbot-va-base}"
TRAIN_ROOT="${TRAIN_ROOT:-${ROOT_DIR}/runs/rokae_pick_up_the_workpiece}"
STATS_PATH="${STATS_PATH:-${DATASET_ROOT}/meta/lingbot_action_stats.json}"

NGPU="${NGPU:-1}"
MASTER_PORT="${MASTER_PORT:-29531}"
NUM_STEPS="${NUM_STEPS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
LOAD_WORKER="${LOAD_WORKER:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"

WANDB_FLAG=(--disable-wandb)
if [[ "${ENABLE_WANDB:-0}" == "1" ]]; then
  WANDB_FLAG=(--enable-wandb)
fi

if [[ -x "${PYTHON_BIN}" ]]; then
  RUNNER=("${PYTHON_BIN}" -m torch.distributed.run)
else
  RUNNER=(uv run python -m torch.distributed.run)
fi

TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
"${RUNNER[@]}" \
  --nproc_per_node="${NGPU}" \
  --master_port "${MASTER_PORT}" \
  -m wan_va.train \
  --config-name rokae_train \
  --dataset-path "${DATASET_ROOT}" \
  --model-path "${MODEL_ROOT}" \
  --save-root "${TRAIN_ROOT}" \
  --norm-stat-path "${STATS_PATH}" \
  --batch-size "${BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRAD_ACCUM_STEPS}" \
  --load-worker "${LOAD_WORKER}" \
  --learning-rate "${LEARNING_RATE}" \
  --warmup-steps "${WARMUP_STEPS}" \
  --num-steps "${NUM_STEPS}" \
  --save-interval "${SAVE_INTERVAL}" \
  "${WANDB_FLAG[@]}"
