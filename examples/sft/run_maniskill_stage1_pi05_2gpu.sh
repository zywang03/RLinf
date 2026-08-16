#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PATH="${VENV_PATH:-${REPO_PATH}/.venv}"

CONFIG_NAME="${CONFIG_NAME:-maniskill_rlt_stage1_sft_openpi_pi05}"
GPU_IDS="${GPU_IDS:-6,7}"
PLACEMENT="${PLACEMENT:-6-7}"

PI_CKPT_PATH="${PI_CKPT_PATH:-/data/ckpt/pi05_base}"
DATASET_PATH="${DATASET_PATH:-/data/datasets/lerobot/maniskill_peginsertionside_joint}"
NORM_STATS_PATH="${NORM_STATS_PATH:-${DATASET_PATH}/norm_stats.json}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  DATASET_PATH="${SMOKE_DATASET_PATH:-/data/datasets/lerobot/maniskill_peginsertionside_joint_smoke32}"
  MAX_STEPS="${MAX_STEPS:-1}"
  SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
  GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2}"
  TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1}"
  LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
else
  MAX_STEPS="${MAX_STEPS:-2000}"
  SAVE_INTERVAL="${SAVE_INTERVAL:-250}"
  GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
  TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-10000}"
  LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-500}"
fi

MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-32}"
SHARDING_STRATEGY="${SHARDING_STRATEGY:-full_shard}"
LOGGER_BACKENDS="${LOGGER_BACKENDS:-[\"wandb\"]}"

TIMESTAMP="$(date +'%Y%m%d_%H%M%S')"
LOG_DIR="${LOG_DIR:-${REPO_PATH}/logs/stage1_pi05_2gpu_${TIMESTAMP}}"

if [[ ! -d "${VENV_PATH}" ]]; then
  echo "Missing venv: ${VENV_PATH}" >&2
  exit 1
fi

if [[ ! -f "${PI_CKPT_PATH}/model.safetensors" ]]; then
  echo "Missing PI checkpoint: ${PI_CKPT_PATH}/model.safetensors" >&2
  exit 1
fi

if [[ ! -d "${DATASET_PATH}" ]]; then
  echo "Missing dataset: ${DATASET_PATH}" >&2
  exit 1
fi

if [[ ! -f "${NORM_STATS_PATH}" ]]; then
  echo "Missing norm stats: ${NORM_STATS_PATH}" >&2
  exit 1
fi

export PYTHONPATH="${PYTHONPATH:-}"
source "${VENV_PATH}/bin/activate"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export HF_HOME="${HF_HOME:-/data/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data/.cache/huggingface/datasets}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/data/datasets/lerobot}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/data/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/data/.local/share/uv/python}"

export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:17897}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:17897}"
export ALL_PROXY="${ALL_PROXY:-http://127.0.0.1:17897}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,::1}"

export WANDB_MODE="${WANDB_MODE:-online}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export EMBODIED_PATH="${SCRIPT_DIR}"
export REPO_PATH
export PYTHONPATH="${REPO_PATH}:${REPO_PATH}/.venv/libero:${PYTHONPATH:-}"

mkdir -p "${LOG_DIR}"

CMD=(
  python "${SCRIPT_DIR}/train_vla_sft.py"
  --config-path config
  --config-name "${CONFIG_NAME}"
  "~cluster.component_placement"
  "+cluster.component_placement.actor=${PLACEMENT}"
  "runner.logger.log_path=${LOG_DIR}"
  "runner.logger.logger_backends=${LOGGER_BACKENDS}"
  "runner.max_steps=${MAX_STEPS}"
  "runner.save_interval=${SAVE_INTERVAL}"
  "actor.micro_batch_size=${MICRO_BATCH_SIZE}"
  "actor.global_batch_size=${GLOBAL_BATCH_SIZE}"
  "actor.optim.total_training_steps=${TOTAL_TRAINING_STEPS}"
  "actor.optim.lr_warmup_steps=${LR_WARMUP_STEPS}"
  "actor.fsdp_config.sharding_strategy=${SHARDING_STRATEGY}"
  "data.train_data_paths.0.dataset_path=${DATASET_PATH}"
  "actor.model.model_path=${PI_CKPT_PATH}"
  "actor.model.openpi_data.norm_stats_path=${NORM_STATS_PATH}"
)

if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

{
  echo "Repo: ${REPO_PATH}"
  echo "Config: ${CONFIG_NAME}"
  echo "GPUs: ${GPU_IDS}"
  echo "Placement: actor=${PLACEMENT}"
  echo "Dataset: ${DATASET_PATH}"
  echo "PI checkpoint: ${PI_CKPT_PATH}"
  echo "Log dir: ${LOG_DIR}"
  echo "Command:"
  printf ' %q' "${CMD[@]}"
  echo
} | tee "${LOG_DIR}/run_stage1_pi05_2gpu.log"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1, command was not executed." | tee -a "${LOG_DIR}/run_stage1_pi05_2gpu.log"
  exit 0
fi

"${CMD[@]}" 2>&1 | tee -a "${LOG_DIR}/run_stage1_pi05_2gpu.log"
