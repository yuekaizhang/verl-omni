# Qwen3-TTS GRPO Recipe

RL post-training of `Qwen3-TTS-12Hz-0.6B-Base` in voice-cloning mode using
`vllm-omni`'s Qwen3-TTS pipeline for rollout and a separately-served remote
`vllm` Qwen3-ASR endpoint as the reward model. Trains via upstream
`verl.trainer.main_ppo` with `algorithm.adv_estimator=grpo` and group sampling
`n>=2`. The existing diffusion trainer is not modified — this recipe is a
parallel addition.

## Pipeline at a glance

```
yuekai/aishell  --pairing-->  parquet(prompt_text, ref_audio, ref_text,
                                       speaker_id, ref_utt_id,
                                       target_utt_id, target_duration)
       |
       v
Qwen3TTSDataset --> AutoRegressiveTTSSingleTurnAgentLoop --> server.generate_tts
       |
       v
vllm-omni 2-stage  (talker emits codec_tokens + logprobs via the
verl-omni-side stage_config override; code2wav emits waveform)
       |
       v
AsrErrorRateRewardManager  --HTTP-->  remote vLLM Qwen3-ASR  --> CER --> reward
       |
       v
upstream verl GRPO loss  -->  FSDP actor update  -->  checkpoint hot-reload
```

## Layout

```
examples/qwen3_tts_grpo_trainer/
├── README.md                                 # this file
├── data_process/aishell_voice_clone.py       # build (ref_audio, ref_text, prompt_text) triples
└── run_smoke.sh                              # minimal smoke launcher

verl_omni/
├── pipelines/qwen3_tts_grpo/
│   ├── stage_configs/qwen3_tts.yaml          # verl-omni-side override (final_output:true + logprobs:1 on stage 0)
│   └── vllm_omni_rollout_adapter.py
├── agent_loop/autoregressive_tts_agent_loop.py   # AR-TTS AgentLoop / Worker / Manager
├── workers/rollout/vllm_rollout/vllm_omni_tts_async_server.py  # TTS HTTP server
├── workers/rollout/autoregressive_tts_server_client.py         # client exposing generate_tts
├── reward_loop/reward_manager/asr_error_rate.py                # AsrErrorRateRewardManager (HTTP-only)
├── utils/dataset/qwen3_tts_dataset.py                          # Qwen3TTSDataset
├── utils/reward_score/asr_error_rate.py                        # CER / WER / reward formula
├── trainer/config/qwen3_tts/*.yaml                             # Hydra config tree
└── trainer/qwen3_tts_grpo/                                     # entry point + validator
    ├── launcher.py                                              # validate_qwen3_tts_recipe_config
    └── main.py                                                  # python -m verl_omni.trainer.qwen3_tts_grpo.main
```

## Quick start (smoke)

1. **Build training parquets** from `yuekai/aishell`:

   ```bash
   .venv/bin/python examples/qwen3_tts_grpo_trainer/data_process/aishell_voice_clone.py \
       --hf-dataset yuekai/aishell \
       --output-dir /path/to/data \
       --train-pairs-per-speaker 20 \
       --eval-pairs-per-speaker 5
   ```

2. **Launch the remote Qwen3-ASR server** (separate process, separate GPU subset):

   ```bash
   CUDA_VISIBLE_DEVICES=4,5 vllm serve Qwen/Qwen3-ASR \
       --host 0.0.0.0 --port 8001
   ```

3. **Run the smoke**:

   ```bash
   QWEN3_TTS_MODEL_PATH=Qwen/Qwen3-TTS-12Hz-0.6B-Base \
   QWEN3_ASR_BASE_URL=http://localhost:8001 \
   TRAIN_PARQUET=/path/to/data/train.parquet \
   EVAL_PARQUET=/path/to/data/eval.parquet \
       bash examples/qwen3_tts_grpo_trainer/run_smoke.sh
   ```

   The launcher rejects the run before any GPU work if `default_agent_loop`,
   `agent.agent_loop_manager_class`, `algorithm.adv_estimator`, `n`, or
   `reward.reward_model.base_url` is misconfigured (or if anyone tries to
   enable a co-located ASR).

## Reward shape

```
reward = (1 - min(CER, CER_CAP))
       - empty_penalty       # truly empty / silent waveform (duration <= 0)
       - duration_penalty    # short non-empty OR duration ratio out of [0.5, 2.0]
       - repetition_penalty  # stage-0 token loops detected
reward = clip(reward, REWARD_FLOOR, REWARD_CEILING)
```

Defaults: `CER_CAP=2.0`, `empty_penalty=1.0`, `duration_penalty=0.5`,
`repetition_penalty=0.5`, `[REWARD_FLOOR, REWARD_CEILING]=[-1.0, 1.0]`. All
configurable via `RewardConfig` / Hydra overrides.

CER is the default Mandarin metric. WER is opt-in via
`reward.reward_model.metric=wer` plus `reward.reward_model.chinese_tokenization=<module.attr>`
(e.g. `jieba.lcut`).

## Held-out evaluation

```bash
QWEN3_ASR_BASE_URL=http://localhost:8001 \
TRAIN_PARQUET=/path/to/data/train.parquet \
    bash examples/qwen3_tts_grpo_trainer/eval.sh \
        /path/to/base.ckpt /path/to/rl.ckpt /path/to/data/eval.parquet
```

Writes `eval_results.json` with `{base_cer, rl_cer,
base_median_duration_ratio, rl_median_duration_ratio,
base_mean_duration_ratio, rl_mean_duration_ratio}`. Pre-run assertion
rejects overlapping `target_utt_id` between train and eval (AC-7).

## Full training run

`run_full.sh` (longer schedule, wandb logging) is the production
counterpart to `run_smoke.sh`. Same env vars; optional
`WANDB_PROJECT` / `WANDB_NAME` / `TOTAL_STEPS` overrides.

## Validation audio logging

`verl_omni.utils.validation_audio_logger.log_validation_step(...)` writes
per-step directories containing 4 generated `.wav` + 4 reference `.wav` +
(when `target_audio` is present) 4 target `.wav` files plus
`metrics.json`. Disk-write failures raise `ArtifactWriteError`; the
post-run helper `post_run_check_emitted_artifacts(...)` flags any step
that emitted zero artifacts.

## Reference docs

Full recipe write-up: [`docs/recipe/qwen3_tts_grpo.md`](../../docs/recipe/qwen3_tts_grpo.md).

## AI assistance disclosure

This recipe was drafted with AI assistance (Claude Code, Opus 4.7). The
submitting human will review and defend every line before merge.
