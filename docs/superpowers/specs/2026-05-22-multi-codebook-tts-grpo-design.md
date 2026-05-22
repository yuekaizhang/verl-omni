# Multi-Codebook TTS GRPO — Design Spec

**Date:** 2026-05-22
**Status:** Draft for review
**Driving issue:** `logs/draft2.md`
**Reference paper:** Fish Audio S2 Technical Report ([arxiv:2603.08823v2](https://arxiv.org/abs/2603.08823))

---

## 1. Problem

The current `qwen3_tts_grpo` recipe in verl-omni trains only codebook 0 of the
Qwen3-TTS codec. Residual codebooks 1..N-1 (predicted in the original model by a
depth-transformer head, `code_predictor`) receive no GRPO signal. The recipe is
also hard-coded to Qwen3-TTS at every layer (trainer dir, configs, model
registration, dataset module name, example scripts) and depends on a 269-line
top-level `qwen3_tts_autoregister.py` monkey-patcher that bridges Qwen3-TTS into
verl's PPO path by patching forward, config layout, the rollout registry, and
flash-attn utilities at runtime.

We want to:

1. Extend the GRPO loss to cover all N codebooks, using the Fish Audio S2
   weighted-sum formulation.
2. Recompute the actor's `old_log_prob` FSDP-side from rollout code sequences
   (rather than consuming vLLM-omni's logprobs as truth) for both codebook
   streams.
3. Replace the monkey-patcher by vendoring the qwen3-tts model code into the
   repo and editing it directly to expose a clean training-time forward.
4. Restructure the whole recipe under a generic `multi_codebook_tts` namespace
   so the same plumbing can host fish-speech RL later with only an adapter
   addition.
5. Export per-codebook logprobs from vllm-omni as a *diagnostic* signal (drift
   sanity check vs FSDP recompute), not as a training input.

The reward (ASR-CER with punctuation stripped) is already correct in
`verl_omni/utils/reward_score/asr_error_rate.py:68`; we only add a regression
test.

## 2. Goals / Non-goals

**Goals**

- One cohesive PR-sized change that delivers items 1-5 above and removes the
  monkey-patcher in the same change.
- No edits to upstream `verl.trainer.ppo.core_algos` — multi-codebook math is
  expressed by calling the existing per-loss-mode function returned by
  `verl.trainer.ppo.core_algos.get_policy_loss_fn(loss_mode)` twice (once per
  stream) and scalar-summing.
- Generic naming everywhere; `qwen3_tts` appears only as a leaf
  model-adapter directory and as one config file name.

**Non-goals (out of scope for v1)**

- fish-speech adapter implementation (directory reserved; tracked separately).
- Sub-talker-only / fast-AR warm-start fine-tuning mode from Fish S2.
- Streaming-rollout / partial-decode logprob alignment.
- KL-coefficient annealing per stream.
- Upstream PRs to verl or vllm-omni-verl (this lives in our fork
  `/lustre/fsw/portfolios/coreai/users/yuekaiz/tts/vllm-omni-verl`, consumed
  via `PYTHONPATH`).

## 3. Loss formulation

Following Fish Audio S2:

```
L_total = w_cb0 · L_cb0^GRPO  +  w_cb_rest · L_cb_rest^GRPO

where for each stream s ∈ {cb0, cb_rest}:
  L_s^GRPO = E_(t,k)∈s [ -A_i · ratio_(t,k) + clip ]  +  β_s · KL_s
  ratio_(t,k) = exp(log π_θ(q_t^k | ctx) - log π_θ_old(q_t^k | ctx))
  A_i = group-relative advantage from per-sample CER reward
```

Key points:

- `A_i` is computed **per sample** by standard GRPO and broadcast over both the
  cb0 time axis and the cb_rest `(time, residual_codebook)` cells.
- The cb0 stream lives on `[B, T]` cells; the cb_rest stream lives on
  `[B, T·(N-1)]` cells after flattening (N = total codebooks; for Qwen3-TTS,
  `N = config.talker_config.num_code_groups`).
- `w_cb0`, `w_cb_rest`, `β_s` (via verl's `kl_loss_coef`) are config knobs.
  Defaults: `w_cb0 = 1.0`, `w_cb_rest = 0.1`.
  Rationale: in RVQ codecs cb0 carries most of the perceptual signal.

The weighted sum is computed *after* verl's per-stream normalization, so each
stream contributes a normalized PG loss before scaling — `w_cb_rest = 0.1` is
not diluted by the larger cell count of the cb_rest stream.

## 4. Architecture

### 4.1 File layout

```
verl_omni/
  models/multi_codebook_tts/
    __init__.py                          # adapter registry by model.name
    base.py                              # MultiCodebookTTSModel ABC
    qwen3_tts/
      __init__.py                        # HF AutoConfig.register('qwen3_tts',...)
      configuration_qwen3_tts.py         # vendored from qwen-tts
      modeling_qwen3_tts.py              # vendored + edited:
                                         #   forward_training(input_ids, codec_ids)
                                         #     -> talker_logits, cb_rest_logits
      adapter.py                         # Qwen3TTSAdapter(MultiCodebookTTSModel)

  workers/actor/
    multi_codebook_dp_actor.py           # MultiCodebookDPActor(DPActor):
                                         #   compute_log_prob -> (cb0_lp, cb_rest_lp)
                                         #   update_policy -> dual compute_policy_loss + sum

  trainer/multi_codebook_tts_grpo/
    __init__.py
    main.py                              # generic hydra entry
    launcher.py                          # config validator
    run_eval.py

  trainer/config/multi_codebook_tts/
    qwen3_tts_trainer.yaml               # selects model.name=qwen3_tts
    fish_speech_trainer.yaml             # stub (commented "TODO v2")
    actor/multi_codebook_actor.yaml      # w_cb0, w_cb_rest, kl coeffs
    rollout/multi_codebook_rollout.yaml
    reward/multi_codebook_reward.yaml
    ref/multi_codebook_ref.yaml

  pipelines/multi_codebook_tts_grpo/
    __init__.py
    vllm_omni_rollout_adapter.py         # dispatches by model.name

  utils/
    attention_utils_fallback.py          # transformers flash_attn fallback
                                         # (extracted from old autoregister)
    ray_runtime_env.py                   # one place to build runtime_env dict
    dataset/multi_codebook_tts_dataset.py  # renamed from qwen3_tts_dataset.py

examples/multi_codebook_tts_grpo/
  qwen3_tts/
    run_full.sh
    run_smoke.sh
    eval.sh
    data_process/
```

### 4.2 Deletions in the same change

- `qwen3_tts_autoregister.py` (top-level)
- `verl_omni/trainer/qwen3_tts_grpo/`
- `verl_omni/trainer/config/qwen3_tts/`
- `verl_omni/pipelines/qwen3_tts_grpo/`
- `verl_omni/utils/dataset/qwen3_tts_dataset.py` (replaced by renamed copy)
- `examples/qwen3_tts_grpo_trainer/`

### 4.3 `MultiCodebookTTSModel` adapter interface

`verl_omni/models/multi_codebook_tts/base.py` defines the ABC every backbone
adapter implements:

```python
class MultiCodebookTTSModel(ABC):
    @abstractmethod
    def load_pretrained(self, path: str) -> nn.Module: ...

    @abstractmethod
    def forward_training(
        self,
        model: nn.Module,
        input_ids: torch.LongTensor,       # [B, T_total]  (prompt + codec)
        codec_ids: torch.LongTensor,       # [B, T_codec, N]  (response region only)
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor,       # [B, T_total]; 1 on generated frames
    ) -> MultiCodebookForwardOutput: ...
    # output.talker_logits   : [B, T_total, V_cb0]   (full seq; loss masks prompt)
    # output.cb_rest_logits  : [B, T_codec, N-1, V_cb_rest]   (response only)

    @property
    @abstractmethod
    def num_codebooks(self) -> int: ...

    @property
    @abstractmethod
    def cb0_vocab_size(self) -> int: ...

    @property
    @abstractmethod
    def cb_rest_vocab_size(self) -> int: ...
```

`Qwen3TTSAdapter.forward_training` drives `talker.model` + `codec_head` to
produce `talker_logits`, then calls `code_predictor.forward_finetune` on
`(talker_hidden, codec_ids)` per-frame to produce `cb_rest_logits`. The
current `_talker_training_forward` shim's clamp/pad logic for out-of-vocab
prompt tokens stays — it moves into `modeling_qwen3_tts.py` as a real
method, not a monkey-patch.

## 5. Training data flow (one step)

```
1. Rollout (vllm-omni async server, unchanged caller side):
     prompt → codes [B, T, N]
     (vllm also emits per-codebook diagnostic logprobs; not consumed by loss)

2. compute_log_prob (FSDP, NEW):
     # Let T = number of codec frames in the response.
     out = adapter.forward_training(model, input_ids, codec_ids, ...)
     # Slice talker_logits to the response region (next-token aligned):
     talker_resp = out.talker_logits[:, prompt_len-1:-1, :]  # [B, T, V_cb0]
     cb0_logprob = gather(log_softmax(talker_resp, -1),
                          codec_ids[..., 0])                 # [B, T]
     cb_rest_logprob = gather(log_softmax(out.cb_rest_logits, -1),
                              codec_ids[..., 1:])            # [B, T, N-1]
     cb_rest_logprob = cb_rest_logprob.flatten(1)            # [B, T·(N-1)]
     TensorDict ← old_log_probs_cb0, old_log_probs_cb_rest

3. compute_ref_log_prob: same dual-forward against frozen ref policy →
     ref_log_probs_cb0, ref_log_probs_cb_rest.

4. compute_advantages (GRPO, unchanged): per-sample reward → A_i [B].

5. MultiCodebookDPActor.update_policy:
     out = adapter.forward_training(model, ...)
     new_cb0_lp, new_cb_rest_lp = gather as above

     # response_mask_codec [B, T] = bool mask over the T codec frames in the response.
     loss_cb0 = verl_policy_loss_fn(
         old_log_prob = old_cb0_lp,                                 # [B, T]
         log_prob     = new_cb0_lp,                                 # [B, T]
         advantages   = A_i.unsqueeze(-1).expand(-1, T),
         response_mask= response_mask_codec,                        # [B, T] bool
         ref_log_prob = ref_cb0_lp,
         config       = actor_cfg.cb0,
     )

     mask_rest = response_mask_codec.unsqueeze(-1) \
                     .expand(-1, -1, N-1).flatten(1)               # [B, T·(N-1)]
     loss_cb_rest = verl_policy_loss_fn(
         old_log_prob = old_cb_rest_lp,                             # [B, T·(N-1)]
         log_prob     = new_cb_rest_lp,                             # [B, T·(N-1)]
         advantages   = A_i.unsqueeze(-1).expand(-1, T*(N-1)),
         response_mask= mask_rest,
         ref_log_prob = ref_cb_rest_lp,
         config       = actor_cfg.cb_rest,
     )

     total_loss = w_cb0 * loss_cb0 + w_cb_rest * loss_cb_rest

     metrics: cb0/pg_clipfrac, cb0/ppo_kl, cb_rest/pg_clipfrac,
              cb_rest/ppo_kl, vllm_drift/cb0, vllm_drift/cb_rest

6. Diagnostic: mean( |vllm_logprob - fsdp_logprob| ) per stream → wandb.
```

## 6. vllm-omni modifications

Edits live in `/lustre/fsw/portfolios/coreai/users/yuekaiz/tts/vllm-omni-verl`
and are loaded via `PYTHONPATH` injection from
`verl_omni.utils.ray_runtime_env`.

Changes:

- In the qwen3-tts vLLM model wrapper, when `SamplingParams.logprobs > 0`,
  also run `code_predictor` on the *sampled* cb0 tokens to produce log-probs
  for cb1..cbN-1 at the sampled residual tokens.
- Extend the existing per-step return payload (currently carrying cb0
  logprobs) with `extra_logprobs.cb_rest` shape `[T, N-1]`.
- The rollout adapter
  (`verl_omni/pipelines/multi_codebook_tts_grpo/vllm_omni_rollout_adapter.py`)
  reads these and stuffs into the rollout TensorDict at
  `vllm_logprob_cb0`, `vllm_logprob_cb_rest`. Loss never touches them.
- All edits are confined to the qwen3-tts model registration in vllm-omni;
  generic vllm-omni dispatcher code is untouched. fish-speech support added
  later mirrors the same hook.

The fork is consumed via `PYTHONPATH` in worker `runtime_env.env_vars`,
plumbed through `verl_omni.utils.ray_runtime_env.build_runtime_env()` (the
single source of truth, replacing the inline block currently in
`verl_omni/trainer/qwen3_tts_grpo/main.py:96-114`).

## 7. Config schema

`trainer/config/multi_codebook_tts/actor/multi_codebook_actor.yaml` (flat
schema — each stream gets a full per-stream actor sub-config rather than
relying on Hydra group composition):

```yaml
w_cb0: 1.0
w_cb_rest: 0.1

cb0:
  policy_loss:
    loss_mode: vanilla
    clip_ratio: 0.2
  use_kl_loss: true
  kl_loss_coef: 0.001
  kl_loss_type: low_var_kl
  loss_agg_mode: token-mean

cb_rest:
  policy_loss:
    loss_mode: vanilla
    clip_ratio: 0.2
  use_kl_loss: true
  kl_loss_coef: 0.001
  kl_loss_type: low_var_kl
  loss_agg_mode: token-mean
```

`trainer/config/multi_codebook_tts/qwen3_tts_trainer.yaml` sets
`model.name: qwen3_tts` and inherits the actor / rollout / reward / ref
sub-configs above.

## 8. Error handling & fail-closed

- **Launcher validator** (`launcher.py`): require `model.name` ∈ registered
  adapters; require `w_cb0 + w_cb_rest > 0`; require ref + rollout + actor
  configs all reference the same `model.name`. Fails fast before
  `ray.init`.
- **Shape assertion in `compute_log_prob`**: assert
  `talker_logits.shape[1] == codec_ids.shape[1]` and
  `cb_rest_logits.shape[2] == adapter.num_codebooks - 1`; raise with
  pointer to the adapter module on mismatch.
- **AC-8 post-run audio-artifact check**: carried over unchanged from the
  current `main.py` `finally` block.
- **Drift alarm (soft)**: if mean `|vllm_logprob - fsdp_logprob|` > 1.0 nat
  per token on either stream for an entire step, emit a `logger.warning`
  with first 3 offending sample indices. Not a hard failure — drift is
  expected to some degree on residual codebooks.

## 9. Testing

### 9.1 Unit

- `tests/multi_codebook_tts/test_loss_math.py`:
  Build a tiny Qwen3TTSConfig (`num_code_groups=4, hidden_size=64,
  num_hidden_layers=2`). Random rollout fixture, 2 samples, T=8. Assert:
  - shapes from `adapter.forward_training` are
    talker `(2, T_total, V_cb0)` and cb_rest `(2, 8, 3, V_cb_rest)`.
  - `MultiCodebookDPActor.compute_log_prob` returns two log-prob tensors
    `(2, 8)` and `(2, 24)` with finite values.
  - Regression check on the combination formula: stub the per-stream
    policy-loss function to return known scalars (e.g. 0.7 and 0.3) and
    assert `update_policy` produces `w_cb0 * 0.7 + w_cb_rest * 0.3` so
    future refactors that drop the weighted sum are caught.

- `tests/multi_codebook_tts/test_advantage_broadcast.py`:
  Per-sample advantage `[B]` correctly broadcasts to cb0 `[B, T]` and
  cb_rest `[B, T·(N-1)]` cells, with consistent masking on padding frames.

- `tests/multi_codebook_tts/test_reward_punctuation.py`:
  Regression: `compute_cer("你好。", "你好")` == 0.0 and
  `compute_wer(...)` strips punctuation identically. Confirms current
  reward behavior holds after the rename.

### 9.2 Integration smoke

- `examples/multi_codebook_tts_grpo/qwen3_tts/run_smoke.sh`:
  2 GRPO steps, 8 samples, real Qwen3-TTS-12Hz-0.6B-Base, vllm-omni
  rollout. Checks (must all pass):
  - `total_loss` finite at both steps.
  - `cb0/ppo_kl`, `cb_rest/ppo_kl`, `cb0/pg_clipfrac`,
    `cb_rest/pg_clipfrac` all logged.
  - At least one validation audio artifact written
    (AC-8 carries through).
  - `vllm_drift/cb0` and `vllm_drift/cb_rest` logged.

### 9.3 What is *not* tested in v1

- fish-speech adapter (doesn't exist yet).
- Numerical equivalence between dual-stream sum and a hand-rolled
  single-pass implementation on the full Fish-S2 loss — checked only on
  the tiny synthetic fixture.

## 10. Migration / rollout

This is one cohesive change (Option 1 from brainstorming):

1. Land all new files under `multi_codebook_tts/`.
2. Delete the listed old paths in the same change.
3. Update CI / README references.
4. Run `run_smoke.sh` on a single node, 8 GPUs (GPUs 0-5 per user
   convention) to validate end-to-end before opening the PR.

No two-trainer transition window. The current `qwen3_tts_grpo` recipe only
trains cb0 so it is not directly comparable to the new dual-stream output;
keeping it around would only multiply maintenance.

## 11. Open questions

None blocking. Two parameters are best tuned empirically after the smoke
run lands:

- Defaults for `w_cb0 / w_cb_rest`. Starting at `1.0 / 0.1` reflects RVQ
  cb0's perceptual weight; if `cb_rest/ppo_kl` saturates or stalls, raise
  `w_cb_rest`.
- `kl_loss_coef` per stream. Starting from the existing single-stream
  value `0.001` for both; cb_rest may want a slightly tighter KL since
  its policy is less informed at init.
