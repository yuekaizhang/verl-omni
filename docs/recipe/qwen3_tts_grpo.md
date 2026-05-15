# Qwen3-TTS GRPO Recipe

RL post-training of `Qwen3-TTS-12Hz-0.6B-Base` in voice-cloning mode. Rollout
uses `vllm-omni`'s Qwen3-TTS pipeline; the reward is a clipped function of
Mandarin CER computed by transcribing each synthesized waveform with a
separately-served remote `vllm` Qwen3-ASR endpoint. Training reuses upstream
`verl.trainer.main_ppo` with `algorithm.adv_estimator=grpo` and group sampling
`n>=2`. The existing diffusion trainer is not modified — this recipe is a
parallel addition.

## Pipeline

```
yuekai/aishell  ─pairing─►  parquet(prompt_text, ref_audio, ref_text,
                                    speaker_id, ref_utt_id, target_utt_id,
                                    target_duration)
        │
        ▼
Qwen3TTSDataset (custom_cls)
        │
        ▼
AutoRegressiveTTSSingleTurnAgentLoop builds the rollout request
        │
        ▼
vllm-omni 2-stage  (talker stage emits codec tokens + per-token logprobs via
the verl-omni-side stage_config override; code2wav emits waveform)
        │
        ▼
AsrErrorRateRewardManager  ──HTTP──►  remote vLLM Qwen3-ASR  ──►  CER  ──►  reward
        │
        ▼
upstream verl GRPO loss  ──►  FSDP actor update  ──►  CheckpointEngineManager.update_weights()
```

## Repository layout

```
examples/qwen3_tts_grpo_trainer/
├── README.md
├── data_process/aishell_voice_clone.py
├── run_smoke.sh
├── run_full.sh
└── eval.sh

verl_omni/
├── pipelines/qwen3_tts_grpo/
│   ├── stage_configs/qwen3_tts.yaml          # verl-omni-side override
│   └── vllm_omni_rollout_adapter.py
├── agent_loop/autoregressive_tts_agent_loop.py
├── workers/rollout/
│   ├── replica.py                            # + AudioRolloutOutput, registers vllm_omni_tts
│   ├── _tts_client_patch.py                  # monkey-patches LLMServerManager.get_client
│   ├── autoregressive_tts_server_client.py   # client exposing generate_tts
│   └── vllm_rollout/vllm_omni_tts_async_server.py
├── reward_loop/reward_manager/asr_error_rate.py
├── utils/
│   ├── dataset/qwen3_tts_dataset.py
│   └── reward_score/asr_error_rate.py
├── trainer/config/qwen3_tts/                 # AC-9 packaging layout
│   ├── qwen3_tts_trainer.yaml
│   ├── actor/qwen3_tts_actor.yaml
│   ├── ref/qwen3_tts_ref.yaml
│   ├── rollout/qwen3_tts_rollout.yaml
│   └── reward/qwen3_tts_reward.yaml
└── trainer/qwen3_tts_grpo/
    ├── launcher.py                            # validate_qwen3_tts_recipe_config
    ├── main.py                                # Hydra entry point (wraps verl.trainer.main_ppo)
    └── run_eval.py                            # eval.sh delegate, produces eval_results.json
```

## How GRPO sees the rollout

vLLM's standard AR scheduler attaches per-token `logprobs` to every sampled
codec token when `SamplingParams.logprobs=N` is set. `vllm-omni`'s
orchestrator emits a stage's `RequestOutput` to the user queue only when
the stage config sets `final_output: true`. Upstream's bundled stage config
only marks stage 1; this recipe ships a verl-omni-side override that adds
`final_output: true` + `logprobs: 1` to stage 0. The two-line diff is the
entire reason the GRPO loss can be computed without patching `vllm-omni`'s
source.

## Reward shape

```
reward = (1 - min(CER, CER_CAP))
       - empty_penalty          # duration <= 0
       - duration_penalty       # generated_duration outside [0.5x, 2.0x] target OR very short
       - repetition_penalty     # stage-0 token loops detected
reward = clip(reward, REWARD_FLOOR, REWARD_CEILING)
```

Defaults: `CER_CAP=2.0`, `empty_penalty=1.0`, `duration_penalty=0.5`,
`repetition_penalty=0.5`, `[REWARD_FLOOR, REWARD_CEILING]=[-1.0, 1.0]`.

CER is the default Mandarin metric. WER is opt-in via
`reward.reward_model.metric=wer` plus `reward.reward_model.chinese_tokenization=<module.attr>`
(e.g. `jieba.lcut`). Endpoint failures produce `success=False` /
`reward=NaN`; the AR-TTS worker excludes failed samples from the GRPO
group-mean (no silent zero-reward fallback).

## Reproducible smoke

1. Build pairs from AISHELL (same-speaker, different-utterance, disjoint by
   `target_utt_id`):

   ```bash
   .venv/bin/python examples/qwen3_tts_grpo_trainer/data_process/aishell_voice_clone.py \
       --hf-dataset yuekai/aishell \
       --output-dir /path/to/data \
       --train-pairs-per-speaker 20 \
       --eval-pairs-per-speaker 5
   ```

2. Launch the remote ASR endpoint on its own GPU subset:

   ```bash
   CUDA_VISIBLE_DEVICES=4,5 vllm serve Qwen/Qwen3-ASR --host 0.0.0.0 --port 8001
   ```

3. Run the smoke (3 optimization steps, GPUs 0-5 reserved by policy):

   ```bash
   QWEN3_TTS_MODEL_PATH=Qwen/Qwen3-TTS-12Hz-0.6B-Base \
   QWEN3_ASR_BASE_URL=http://localhost:8001 \
   TRAIN_PARQUET=/path/to/data/train.parquet \
   EVAL_PARQUET=/path/to/data/eval.parquet \
       bash examples/qwen3_tts_grpo_trainer/run_smoke.sh
   ```

   The launcher fails fast — before any GPU work — when:
   - `default_agent_loop` is not `autoregressive_tts_single_turn_agent`.
   - `agent.agent_loop_manager_class` doesn't point at the AR-TTS manager.
   - `algorithm.adv_estimator` is not `grpo`.
   - `n < 2`.
   - `reward.reward_model.base_url` is missing.
   - `reward.reward_model.co_located` is true.

4. Held-out eval against the produced parquet:

   ```bash
   QWEN3_ASR_BASE_URL=http://localhost:8001 \
   TRAIN_PARQUET=/path/to/data/train.parquet \
       bash examples/qwen3_tts_grpo_trainer/eval.sh \
           /path/to/base.ckpt /path/to/rl.ckpt /path/to/data/eval.parquet
   ```

   Writes `eval_results.json` with `{base_cer, rl_cer,
   base_median_duration_ratio, rl_median_duration_ratio,
   base_mean_duration_ratio, rl_mean_duration_ratio}`. The script refuses
   to score when the eval `target_utt_id` set overlaps the training set
   (per AC-7).

5. Full run with wandb logging:

   ```bash
   QWEN3_TTS_MODEL_PATH=... QWEN3_ASR_BASE_URL=... \
   TRAIN_PARQUET=... EVAL_PARQUET=... \
   WANDB_PROJECT=verl_omni_qwen3_tts_grpo \
   WANDB_NAME=run_001 TOTAL_STEPS=500 \
       bash examples/qwen3_tts_grpo_trainer/run_full.sh
   ```

## Hardware

This recipe defaults to `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5` (six GPUs;
the last two on the node are reserved). Typical split:

- Rollout (vllm-omni Qwen3-TTS, 2-stage): 2-3 GPUs.
- Actor + ref (FSDP): 2-3 GPUs.
- Remote Qwen3-ASR server: 1-2 GPUs from the same subset, launched as a
  separate process. **Never co-located in the trainer process** (the
  validator rejects `co_located=true` configurations).

## Validation logging

Every validation step writes a per-step directory under the run output
dir containing 4 generated `.wav` files, 4 reference `.wav` files, and
(when `target_audio` is available in the dataset) 4 target `.wav` files,
plus `metrics.json` with the scalar fields. Disk-write failures (full
disk, permission error) surface a typed error rather than silently
skipping artifacts; a post-run check flags any validation step that
emitted zero artifacts.

## AI assistance disclosure

This recipe was drafted with AI assistance (Claude Code, Opus 4.7). The
submitting human reviews and defends every line before merge.
