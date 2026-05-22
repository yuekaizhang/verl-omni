#!/usr/bin/env bash
# Held-out evaluation for the Qwen3-TTS GRPO recipe.
#
# Usage:
#   bash eval.sh <base_ckpt> <rl_ckpt> <eval_parquet>
#
# Writes ``eval_results.json`` into the current directory with:
#   {
#     "base_cer", "rl_cer",
#     "base_median_duration_ratio", "rl_median_duration_ratio",
#     "base_mean_duration_ratio",   "rl_mean_duration_ratio"
#   }
#
# Requires QWEN3_ASR_BASE_URL pointing at the remote vLLM Qwen3-ASR endpoint.
# Rejects an eval parquet whose ``target_utt_id`` set overlaps the training
# parquet (the script's pre-run assertion in run_eval.py).

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <base_ckpt> <rl_ckpt> <eval_parquet>" >&2
  exit 2
fi

BASE_CKPT="$1"
RL_CKPT="$2"
EVAL_PARQUET="$3"
OUTPUT="${OUTPUT:-eval_results.json}"
TRAIN_PARQUET="${TRAIN_PARQUET:-}"

: "${QWEN3_ASR_BASE_URL:?set QWEN3_ASR_BASE_URL to the remote vLLM Qwen3-ASR endpoint}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"

ROOT_DIR="$(cd "$(dirname "$0")"/../.. && pwd)"

"${ROOT_DIR}/.venv/bin/python" -m verl_omni.trainer.multi_codebook_tts_grpo.run_eval \
  --base-ckpt "${BASE_CKPT}" \
  --rl-ckpt "${RL_CKPT}" \
  --eval-parquet "${EVAL_PARQUET}" \
  --asr-base-url "${QWEN3_ASR_BASE_URL}" \
  --train-parquet "${TRAIN_PARQUET}" \
  --output "${OUTPUT}"
