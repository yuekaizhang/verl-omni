#!/usr/bin/env bash
# Qwen3-TTS GRPO smoke run.
#
# Prerequisites (set before running):
#   QWEN3_TTS_MODEL_PATH=Qwen/Qwen3-TTS-12Hz-0.6B-Base   # or local checkpoint dir
#   QWEN3_ASR_BASE_URL=http://<asr-host>:<port>           # separately-served vLLM Qwen3-ASR
#   TRAIN_PARQUET=/path/to/train.parquet                  # built via data_process/aishell_voice_clone.py
#   EVAL_PARQUET=/path/to/eval.parquet
#
# GPU layout: this script restricts to CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 per the
# project policy (reserve GPUs 6-7 for other workloads). The remote Qwen3-ASR
# server must be launched separately on its own GPU subset.

set -euo pipefail

: "${QWEN3_TTS_MODEL_PATH:?set QWEN3_TTS_MODEL_PATH to the Qwen3-TTS-12Hz-0.6B-Base checkpoint}"
: "${QWEN3_ASR_BASE_URL:?set QWEN3_ASR_BASE_URL to the remote vLLM Qwen3-ASR endpoint (no co-located mode)}"
: "${TRAIN_PARQUET:?set TRAIN_PARQUET to the training parquet built by aishell_voice_clone.py}"
: "${EVAL_PARQUET:?set EVAL_PARQUET to the evaluation parquet built by aishell_voice_clone.py}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
# The Ray/vllm worker raises if both ROCR_VISIBLE_DEVICES and
# CUDA_VISIBLE_DEVICES are set. Clear the AMD-style env var some
# container images leak in.
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
# Force a local-only Ray cluster so we don't accidentally join the
# host's pre-existing Ray (which may be a different Python/Ray pair).
export RAY_ADDRESS="${RAY_ADDRESS:-local}"

# Qwen3-TTS isn't shipped with stock transformers; the `qwen-tts` PyPI
# package supplies `Qwen3TTSConfig` / `Qwen3TTSForConditionalGeneration`.
# When the package isn't installed (e.g. quota-restricted .venv), set
# this to a local clone of the qwen-tts repo so verl_omni/__init__.py
# can put it on sys.path and register the classes with HF AutoModel.
export QWEN3_TTS_SOURCE_DIR="${QWEN3_TTS_SOURCE_DIR:-/lustre/fsw/portfolios/coreai/users/yuekaiz/tts/Qwen3-TTS}"

ROOT_DIR="$(cd "$(dirname "$0")"/../.. && pwd)"

# Hydra defaults write run logs / checkpoints under the launching CWD
# (``./outputs/``), which on this lustre filesystem hits the per-user
# inode quota. Redirect both Hydra's own working dir and verl's
# checkpoint dir to ``/tmp`` (host disk, plenty of space) so the smoke
# isn't blocked by quota. ``HYDRA_OUTPUT_BASE`` lets the caller override.
HYDRA_OUTPUT_BASE="${HYDRA_OUTPUT_BASE:-/tmp/qwen3_tts_smoke_outputs}"

# shellcheck disable=SC2086
"${ROOT_DIR}/.venv/bin/python" -m verl_omni.trainer.multi_codebook_tts_grpo.main \
  data.train_files="${TRAIN_PARQUET}" \
  data.val_files="${EVAL_PARQUET}" \
  actor_rollout_ref.model.path="${QWEN3_TTS_MODEL_PATH}" \
  reward.reward_model.base_url="${QWEN3_ASR_BASE_URL}" \
  trainer.total_training_steps=3 \
  trainer.experiment_name=qwen3_tts_grpo_smoke \
  trainer.default_local_dir="${HYDRA_OUTPUT_BASE}/ckpt" \
  hydra.run.dir="${HYDRA_OUTPUT_BASE}/hydra_run/\${now:%Y-%m-%d}/\${now:%H-%M-%S}" \
  hydra.sweep.dir="${HYDRA_OUTPUT_BASE}/hydra_sweep"
