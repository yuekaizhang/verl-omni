#!/usr/bin/env bash
# Full Qwen3-TTS GRPO training run.
#
# Same wiring as run_smoke.sh but with a longer schedule and wandb logging
# enabled. Intended for the recipe's reference training curves rather than
# CI-fast smoke.
#
# Required env (same as run_smoke.sh):
#   QWEN3_TTS_MODEL_PATH, QWEN3_ASR_BASE_URL, TRAIN_PARQUET, EVAL_PARQUET
# Optional env:
#   WANDB_PROJECT (default: qwen3-tts-verl-omni)
#   WANDB_NAME    (default: qwen3_tts_grpo_full)
#   TOTAL_STEPS   (default: 500)

set -euo pipefail

: "${QWEN3_TTS_MODEL_PATH:?set QWEN3_TTS_MODEL_PATH to the Qwen3-TTS-12Hz-0.6B-Base checkpoint}"
: "${QWEN3_ASR_BASE_URL:?set QWEN3_ASR_BASE_URL to the remote vLLM Qwen3-ASR endpoint (no co-located mode)}"
: "${TRAIN_PARQUET:?set TRAIN_PARQUET to the training parquet built by aishell_voice_clone.py}"
: "${EVAL_PARQUET:?set EVAL_PARQUET to the evaluation parquet built by aishell_voice_clone.py}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"

WANDB_PROJECT="${WANDB_PROJECT:-qwen3-tts-verl-omni}"
WANDB_NAME="${WANDB_NAME:-qwen3_tts_grpo_full}"
TOTAL_STEPS="${TOTAL_STEPS:-500}"

ROOT_DIR="$(cd "$(dirname "$0")"/../.. && pwd)"

"${ROOT_DIR}/.venv/bin/python" -m verl_omni.trainer.multi_codebook_tts_grpo.main \
  data.train_files="${TRAIN_PARQUET}" \
  data.val_files="${EVAL_PARQUET}" \
  actor_rollout_ref.model.path="${QWEN3_TTS_MODEL_PATH}" \
  reward.reward_model.base_url="${QWEN3_ASR_BASE_URL}" \
  trainer.total_training_steps="${TOTAL_STEPS}" \
  trainer.experiment_name="${WANDB_NAME}" \
  trainer.project_name="${WANDB_PROJECT}" \
  trainer.logger='[console,wandb]'
