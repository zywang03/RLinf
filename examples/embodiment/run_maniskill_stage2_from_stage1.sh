#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PATH="${VENV_PATH:-${REPO_PATH}/.venv}"

# Default to residual Stage2 SAC with learned std and entropy tuning.
# Override with CONFIG_NAME=maniskill_rlt_stage2_ac_mlp for the non-residual baseline,
# or CONFIG_NAME=maniskill_rlt_stage2_residual_ac_mlp for residual fixed-std/no-entropy.
CONFIG_NAME="${CONFIG_NAME:-maniskill_rlt_stage2_residual_entropy_ac_mlp}"

# Frozen Stage1/base VLA used as the RLT feature model. Override with
# BASE_MODEL_PATH=/path/to/<stage1_ckpt>/actor (STAGE1_ACTOR_PATH also works).
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${STAGE1_ACTOR_PATH:-${REPO_PATH}/logs/pi_base_model/global_step_10000/actor}}"
STAGE1_ACTOR_PATH="${STAGE1_ACTOR_PATH:-${BASE_MODEL_PATH}}"
EXPERT_ACTOR_PATH="${EXPERT_ACTOR_PATH:-${STAGE1_ACTOR_PATH}}"
NORM_STATS_PATH="${NORM_STATS_PATH:-/data/datasets/lerobot/maniskill_peginsertionside_joint/norm_stats.json}"

# Default to the stable two-GPU split for this machine: ManiSkill RGB rendering
# stays on GPU7, while frozen OpenPI/RLT rollout inference runs on GPU6.
# Override with DUAL_GPU_MODE=single for the one-GPU known-good path.
USE_DUAL_GPU="${USE_DUAL_GPU:-0}"
DUAL_GPU_MODE="${DUAL_GPU_MODE:-}"
if [[ -z "${DUAL_GPU_MODE}" ]]; then
  if [[ "${USE_DUAL_GPU}" == "1" ]]; then
    DUAL_GPU_MODE="split_rollout"
  else
    DUAL_GPU_MODE="split_rollout"
  fi
fi

case "${DUAL_GPU_MODE}" in
  single)
    GPU_IDS="${GPU_IDS:-7}"
    ACTOR_PLACEMENT="${ACTOR_PLACEMENT:-7-7}"
    ENV_PLACEMENT="${ENV_PLACEMENT:-7-7}"
    ROLLOUT_PLACEMENT="${ROLLOUT_PLACEMENT:-7-7}"
    ;;
  split_rollout)
  GPU_IDS="${GPU_IDS:-6,7}"
  ACTOR_PLACEMENT="${ACTOR_PLACEMENT:-7-7}"
  ENV_PLACEMENT="${ENV_PLACEMENT:-7-7}"
  ROLLOUT_PLACEMENT="${ROLLOUT_PLACEMENT:-6-6}"
    ;;
  env_rollout)
    # Official ManiSkill-style multi-GPU pattern: separate processes, each
    # isolated to one CUDA device, and each process vectorizes its own envs.
    # Requires GPU6 rendering smoke to pass on this machine.
    GPU_IDS="${GPU_IDS:-6,7}"
    ACTOR_PLACEMENT="${ACTOR_PLACEMENT:-7-7}"
    ENV_PLACEMENT="${ENV_PLACEMENT:-6-7}"
    ROLLOUT_PLACEMENT="${ROLLOUT_PLACEMENT:-6-7}"
    ;;
  *)
    echo "Unknown DUAL_GPU_MODE=${DUAL_GPU_MODE}; expected single, split_rollout, or env_rollout." >&2
    exit 1
    ;;
esac

# Sapien's renderer enumerates Vulkan devices independently from CUDA. Pin it to
# the same physical GPU as the Ray placement unless the caller already provided a
# more specific value. Bus ids on this machine:
#   GPU6 -> 00000000:CA:00.0
#   GPU7 -> 00000000:DA:00.0
case "${GPU_IDS%%,*}" in
  6) DEFAULT_DRI_PRIME="pci-0000_ca_00_0" ;;
  7) DEFAULT_DRI_PRIME="pci-0000_da_00_0" ;;
  *) DEFAULT_DRI_PRIME="" ;;
esac
GLOBAL_DRI_PRIME="${GLOBAL_DRI_PRIME:-0}"
if [[ "${GLOBAL_DRI_PRIME}" == "1" ]]; then
  DRI_PRIME="${DRI_PRIME:-${DEFAULT_DRI_PRIME}}"
else
  DRI_PRIME="${DRI_PRIME:-}"
fi

# The Stage1 eval runner used osmesa for non-ManiSkill GL setup. Vulkan still
# does the ManiSkill RGB work, but keeping the process environment identical
# avoids a needless difference in graphics initialization.
MUJOCO_GL="${MUJOCO_GL:-osmesa}"
PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"

USE_EXPERT="${USE_EXPERT:-0}"

TIMESTAMP="$(date +'%Y%m%d_%H%M%S')"
LOG_DIR="${LOG_DIR:-${REPO_PATH}/logs/stage2_from_stage1_${TIMESTAMP}}"
RUN_LOG="${LOG_DIR}/run_maniskill_stage2_from_stage1.log"

if [[ "${SMOKE:-0}" == "1" ]]; then
  DEFAULT_MAX_EPOCHS=2
  DEFAULT_MAX_STEPS=2
  DEFAULT_VAL_CHECK_INTERVAL=10
  DEFAULT_SAVE_INTERVAL=-1
  DEFAULT_TRAIN_NUM_ENVS=32
  DEFAULT_EVAL_NUM_ENVS=16
else
  DEFAULT_MAX_EPOCHS=5000
  DEFAULT_MAX_STEPS=-1
  DEFAULT_VAL_CHECK_INTERVAL=10
  DEFAULT_SAVE_INTERVAL=50
  DEFAULT_TRAIN_NUM_ENVS=32
  DEFAULT_EVAL_NUM_ENVS=16
fi

MAX_EPOCHS="${MAX_EPOCHS:-${DEFAULT_MAX_EPOCHS}}"
MAX_STEPS="${MAX_STEPS:-${DEFAULT_MAX_STEPS}}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-${DEFAULT_VAL_CHECK_INTERVAL}}"
SAVE_INTERVAL="${SAVE_INTERVAL:-${DEFAULT_SAVE_INTERVAL}}"
LOGGER_BACKENDS="${LOGGER_BACKENDS:-[\"wandb\"]}"
TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-${DEFAULT_TRAIN_NUM_ENVS}}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-${DEFAULT_EVAL_NUM_ENVS}}"
# Keep Stage2 rendering consistent with Stage1: the frozen VLA was trained on
# 384x384 default-shader images, and the OpenPI model upscales/downscales to
# 224x224 internally. Lower resolutions only degrade the input distribution.
SHADER_PACK="${SHADER_PACK:-default}"
CAMERA_WIDTH="${CAMERA_WIDTH:-384}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-384}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-100}"
MAX_STEPS_PER_ROLLOUT_EPOCH="${MAX_STEPS_PER_ROLLOUT_EPOCH:-${MAX_EPISODE_STEPS}}"
RESIDUAL_SCALE="${RESIDUAL_SCALE:-}"
REFERENCE_DROPOUT_PROB="${REFERENCE_DROPOUT_PROB:-}"
WARMUP_MIN_SIZE="${WARMUP_MIN_SIZE:-}"
WARMUP_POST_COLLECT_UPDATES="${WARMUP_POST_COLLECT_UPDATES:-}"
ACTOR_WEIGHT_WARMUP_UPDATES="${ACTOR_WEIGHT_WARMUP_UPDATES:-}"

if [[ ! -d "${VENV_PATH}" ]]; then
  echo "Missing venv: ${VENV_PATH}" >&2
  exit 1
fi

if [[ ! -d "${BASE_MODEL_PATH}" ]]; then
  echo "Missing base model checkpoint directory: ${BASE_MODEL_PATH}" >&2
  exit 1
fi

if [[ ! -f "${BASE_MODEL_PATH}/model_state_dict/full_weights.pt" ]]; then
  echo "Missing base model full weights: ${BASE_MODEL_PATH}/model_state_dict/full_weights.pt" >&2
  exit 1
fi

if [[ ! -f "${NORM_STATS_PATH}" ]]; then
  echo "Missing norm stats: ${NORM_STATS_PATH}" >&2
  exit 1
fi

if [[ ! -d "${EXPERT_ACTOR_PATH}" ]]; then
  echo "Missing expert actor checkpoint directory: ${EXPERT_ACTOR_PATH}" >&2
  exit 1
fi

export PYTHONPATH="${PYTHONPATH:-}"
source "${VENV_PATH}/bin/activate"

mkdir -p "${LOG_DIR}"

ENV_ROLLOUT_PREFLIGHT="${ENV_ROLLOUT_PREFLIGHT:-1}"
if [[ "${DRY_RUN:-0}" != "1" && "${DUAL_GPU_MODE}" == "env_rollout" && "${ENV_ROLLOUT_PREFLIGHT}" == "1" ]]; then
  PREFLIGHT_LOG="${LOG_DIR}/env_rollout_gpu6_preflight.log"
  echo "Preflighting GPU6 ManiSkill RGB rendering for env_rollout mode..." | tee -a "${PREFLIGHT_LOG}"
  if ! timeout 75 bash -lc "cd '${REPO_PATH}' && CUDA_VISIBLE_DEVICES=6 DRI_PRIME=pci-0000_ca_00_0 MUJOCO_GL='${MUJOCO_GL}' PYOPENGL_PLATFORM='${PYOPENGL_PLATFORM}' python examples/embodiment/check_maniskill_gpu_render.py --env-id PickCube-v1 --control-mode pd_ee_delta_pos --num-envs 1 --width 64 --height 64 --shader-pack minimal" >>"${PREFLIGHT_LOG}" 2>&1; then
    echo "GPU6 ManiSkill RGB preflight failed; falling back to split_rollout mode." | tee -a "${PREFLIGHT_LOG}"
    DUAL_GPU_MODE="split_rollout"
    GPU_IDS="6,7"
    ACTOR_PLACEMENT="7-7"
    ENV_PLACEMENT="7-7"
    ROLLOUT_PLACEMENT="6-6"
  else
    echo "GPU6 ManiSkill RGB preflight passed; using env_rollout mode." | tee -a "${PREFLIGHT_LOG}"
  fi
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export HF_HOME="${HF_HOME:-/data/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data/.cache/huggingface/datasets}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/data/datasets/lerobot}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/data/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/data/.local/share/uv/python}"

export WANDB_MODE="${WANDB_MODE:-online}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MUJOCO_GL
export PYOPENGL_PLATFORM
if [[ -n "${DRI_PRIME}" ]]; then
  export DRI_PRIME
fi
export ROBOTWIN_PATH="${ROBOTWIN_PATH:-/path/to/RoboTwin}"
export EMBODIED_PATH="${SCRIPT_DIR}"
export REPO_PATH
export PYTHONPATH="${REPO_PATH}:${ROBOTWIN_PATH}:${PYTHONPATH:-}"

CMD=(
  python "${SCRIPT_DIR}/train_embodied_agent.py"
  --config-path config
  --config-name "${CONFIG_NAME}"
  "runner.logger.log_path=${LOG_DIR}"
  "runner.logger.logger_backends=${LOGGER_BACKENDS}"
  "runner.max_epochs=${MAX_EPOCHS}"
  "runner.max_steps=${MAX_STEPS}"
  "runner.val_check_interval=${VAL_CHECK_INTERVAL}"
  "runner.save_interval=${SAVE_INTERVAL}"
  "cluster.component_placement.actor=${ACTOR_PLACEMENT}"
  "cluster.component_placement.env=${ENV_PLACEMENT}"
  "cluster.component_placement.rollout=${ROLLOUT_PLACEMENT}"
  "env.train.total_num_envs=${TRAIN_NUM_ENVS}"
  "env.eval.total_num_envs=${EVAL_NUM_ENVS}"
  "env.train.init_params.sensor_configs.shader_pack=${SHADER_PACK}"
  "env.eval.init_params.sensor_configs.shader_pack=${SHADER_PACK}"
  "env.train.init_params.sensor_configs.width=${CAMERA_WIDTH}"
  "env.train.init_params.sensor_configs.height=${CAMERA_HEIGHT}"
  "env.eval.init_params.sensor_configs.width=${CAMERA_WIDTH}"
  "env.eval.init_params.sensor_configs.height=${CAMERA_HEIGHT}"
  "env.train.max_episode_steps=${MAX_EPISODE_STEPS}"
  "env.eval.max_episode_steps=${MAX_EPISODE_STEPS}"
  "env.train.max_steps_per_rollout_epoch=${MAX_STEPS_PER_ROLLOUT_EPOCH}"
  "env.eval.max_steps_per_rollout_epoch=${MAX_STEPS_PER_ROLLOUT_EPOCH}"
  "env.train.init_params.max_episode_steps=${MAX_EPISODE_STEPS}"
  "env.eval.init_params.max_episode_steps=${MAX_EPISODE_STEPS}"
  "rollout.rlt_feature_model.model_path=${BASE_MODEL_PATH}"
  "rollout.rlt_feature_model.openpi_data.norm_stats_path=${NORM_STATS_PATH}"
)

if [[ -n "${RESIDUAL_SCALE}" ]]; then
  CMD+=(
    "actor.model.residual_scale=${RESIDUAL_SCALE}"
    "rollout.model.residual_scale=${RESIDUAL_SCALE}"
  )
fi

if [[ -n "${REFERENCE_DROPOUT_PROB}" ]]; then
  CMD+=(
    "algorithm.reference_dropout_prob=${REFERENCE_DROPOUT_PROB}"
  )
fi

if [[ -n "${WARMUP_MIN_SIZE}" ]]; then
  CMD+=(
    "algorithm.rlt_schedule.warmup_min_size=${WARMUP_MIN_SIZE}"
  )
fi

if [[ -n "${WARMUP_POST_COLLECT_UPDATES}" ]]; then
  CMD+=(
    "algorithm.rlt_schedule.warmup_post_collect_updates=${WARMUP_POST_COLLECT_UPDATES}"
  )
fi

if [[ -n "${ACTOR_WEIGHT_WARMUP_UPDATES}" ]]; then
  CMD+=(
    "algorithm.actor_weight_schedule.warmup_updates=${ACTOR_WEIGHT_WARMUP_UPDATES}"
  )
fi

if [[ "${USE_EXPERT}" == "1" ]]; then
  CMD+=(
    "env.train.rlt_policy_switch.expert_takeover.enable=True"
    "rollout.expert_model.model_path=${EXPERT_ACTOR_PATH}"
  )
else
  CMD+=(
    "env.train.rlt_policy_switch.expert_takeover.enable=False"
    "rollout.expert_model=null"
  )
fi

if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

{
  echo "Repo: ${REPO_PATH}"
  echo "Config: ${CONFIG_NAME}"
  echo "GPUs: ${GPU_IDS}"
  echo "Use dual GPU: ${USE_DUAL_GPU}"
  echo "Dual GPU mode: ${DUAL_GPU_MODE}"
  echo "DRI_PRIME: ${DRI_PRIME:-<unset>}"
  echo "Placement: actor=${ACTOR_PLACEMENT}, env=${ENV_PLACEMENT}, rollout=${ROLLOUT_PLACEMENT}"
  echo "Base model actor: ${BASE_MODEL_PATH}"
  echo "Expert actor: ${EXPERT_ACTOR_PATH}"
  echo "Use expert takeover: ${USE_EXPERT}"
  echo "Norm stats: ${NORM_STATS_PATH}"
  echo "Train/eval envs: ${TRAIN_NUM_ENVS}/${EVAL_NUM_ENVS}"
  echo "Camera: ${CAMERA_WIDTH}x${CAMERA_HEIGHT}, shader=${SHADER_PACK}"
  echo "Episode/rollout steps: max_episode=${MAX_EPISODE_STEPS}, rollout_epoch_steps=${MAX_STEPS_PER_ROLLOUT_EPOCH}"
  echo "Log dir: ${LOG_DIR}"
  echo "Command:"
  printf ' %q' "${CMD[@]}"
  echo
} | tee "${RUN_LOG}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1, command was not executed." | tee -a "${RUN_LOG}"
  exit 0
fi

"${CMD[@]}" 2>&1 | tee -a "${RUN_LOG}"
