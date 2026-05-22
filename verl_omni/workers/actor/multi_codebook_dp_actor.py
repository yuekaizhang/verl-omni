"""Multi-codebook data-parallel actor for the `multi_codebook_tts_grpo` recipe.

`MultiCodebookDPActor` is a standalone class (not a subclass of upstream verl's
`DPActor`) that exposes the two operations the recipe trainer calls per step:

- `compute_log_prob(data, model)` -> `TensorDict` with two named log-prob fields
  `old_log_probs_cb0` (`[B, T]`) and `old_log_probs_cb_rest` (`[B, T*(N-1)]`).
  The legacy `old_log_probs` field is NOT written; any stale consumer reading
  it fails loudly (per round-0 task17 Codex audit, no shared upstream consumer
  hardcodes that field, so this is safe).
- `update_policy(data, model)` -> `TensorDict` with `actor/total_loss` and
  per-stream metrics under `actor/cb0/*` and `actor/cb_rest/*`. Internally
  calls `verl.trainer.ppo.core_algos.get_policy_loss_fn(loss_mode)` once per
  stream with per-stream tensors and combines as
  `w_cb0 * L_cb0 + w_cb_rest * L_cb_rest`. Each stream is normalized first
  (per its own `loss_agg_mode`) so AC-1.1's per-stream-normalization
  invariance holds.

The adapter (see `verl_omni.models.multi_codebook_tts.base.MultiCodebookTTSModel`)
hides codec-specific forward + log-prob extraction so this class stays
codec-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# `tensordict` and the adapter registry are imported lazily inside the
# methods that need them. Keeping the top-level imports torch-only lets
# the math helpers below (`_gather_log_probs_*`, `_build_cb_rest_mask`,
# `_prefix_metrics`) be unit-tested on a Python install that has torch
# but not tensordict / verl.


@dataclass
class _StreamLossConfig:
    """Per-stream policy-loss configuration.

    Mirrors the YAML schema at
    `trainer/config/multi_codebook_tts/actor/multi_codebook_actor.yaml`. The
    `loss_mode` resolves a `PolicyLossFn` in `verl.trainer.ppo.core_algos.POLICY_LOSS_REGISTRY`;
    `clip_ratio` is read by that function (e.g. `compute_policy_loss_vanilla`)
    from `config.policy_loss.clip_ratio` when `config` is supplied.
    `loss_agg_mode` ('token-mean' / 'token-sum' / 'sample-mean') drives
    `agg_loss`. `kl_loss_coef` + `kl_loss_type` drive an optional KL penalty
    against the reference policy.
    """

    loss_mode: str = "vanilla"
    clip_ratio: float = 0.2
    use_kl_loss: bool = True
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"
    loss_agg_mode: str = "token-mean"


@dataclass
class MultiCodebookActorConfig:
    """Top-level multi-codebook actor configuration."""

    w_cb0: float = 1.0
    w_cb_rest: float = 0.1
    cb0: _StreamLossConfig = None  # type: ignore[assignment]
    cb_rest: _StreamLossConfig = None  # type: ignore[assignment]
    model_name: str = "qwen3_tts"

    def __post_init__(self) -> None:
        if self.cb0 is None:
            self.cb0 = _StreamLossConfig()
        if self.cb_rest is None:
            self.cb_rest = _StreamLossConfig()
        if self.w_cb0 + self.w_cb_rest <= 0:
            raise ValueError(
                f"w_cb0 + w_cb_rest must be > 0; got {self.w_cb0=}, "
                f"{self.w_cb_rest=}."
            )


def _gather_log_probs_cb0(
    talker_logits: torch.Tensor,
    target_cb0: torch.Tensor,
    prompt_lens: torch.Tensor,
    T_codec: int,
) -> torch.Tensor:
    """Gather cb0 per-frame log-probs from talker_logits.

    talker_logits: `[B, T_total, V_cb0]`, full-sequence logits from
    `forward_training`. We slice each sample's response region as
    `talker_logits[b, prompt_lens[b]-1 : prompt_lens[b]-1+T_codec, :]` via
    `torch.gather` so variable prompt lengths are handled correctly (AC-5.1).

    target_cb0: `[B, T_codec]` codec-id of codebook 0 at each response frame.

    Returns: `[B, T_codec]` log-probs of the target cb0 at each response frame.
    """
    B, T_total, V_cb0 = talker_logits.shape
    device = talker_logits.device

    # frame_offsets[b, t] = prompt_lens[b] - 1 + t  (logit predicting frame t)
    frame_offsets = (
        prompt_lens.to(device).unsqueeze(1) - 1
        + torch.arange(T_codec, device=device).unsqueeze(0)
    ).clamp(0, T_total - 1)  # [B, T_codec]

    # Gather per-sample response logits, then per-frame log-prob of target.
    resp_logits = torch.gather(
        talker_logits, dim=1,
        index=frame_offsets.unsqueeze(-1).expand(-1, -1, V_cb0),
    )  # [B, T_codec, V_cb0]
    resp_log_softmax = F.log_softmax(resp_logits, dim=-1)
    # Clamp targets defensively (mask handles invalid positions at loss layer).
    safe_target = target_cb0.clamp(0, V_cb0 - 1)
    return resp_log_softmax.gather(
        dim=-1, index=safe_target.unsqueeze(-1),
    ).squeeze(-1)  # [B, T_codec]


def _gather_log_probs_cb_rest(
    cb_rest_logits: torch.Tensor,
    target_cb_rest: torch.Tensor,
) -> torch.Tensor:
    """Gather cb_rest per-position log-probs from `cb_rest_logits`.

    cb_rest_logits: `[B, T_codec, N-1, V_cb_rest]` (already in
    frame-major-outer + codebook-major-inner layout per AC-5.2).

    target_cb_rest: `[B, T_codec, N-1]` codec-ids of cb1..cb_{N-1} at each
    response frame.

    Returns: `[B, T_codec * (N-1)]` log-probs of the target codes, flattened
    in the same codebook-major-within-frame order (AC-5.2). The actor's
    response mask is similarly flattened by the caller, so positions line up.
    """
    log_softmax = F.log_softmax(cb_rest_logits, dim=-1)
    V_cb_rest = cb_rest_logits.size(-1)
    safe_target = target_cb_rest.clamp(0, V_cb_rest - 1)
    per_position = log_softmax.gather(
        dim=-1, index=safe_target.unsqueeze(-1),
    ).squeeze(-1)  # [B, T_codec, N-1]
    # AC-5.2 flatten: codebook-major-within-frame (frame outer, codebook inner).
    return per_position.flatten(1, 2)  # [B, T_codec * (N-1)]


def _build_cb_rest_mask(
    response_mask_codec: torch.Tensor, num_residual_codebooks: int,
) -> torch.Tensor:
    """Broadcast the per-frame codec mask over the residual-codebook axis and
    flatten in codebook-major-within-frame order (AC-5.2).

    response_mask_codec: `[B, T_codec]`.
    Returns: `[B, T_codec * (N-1)]` (AC-5.5: sum = `response_mask_codec.sum() * (N-1)`).
    """
    if num_residual_codebooks <= 0:
        raise ValueError(
            f"num_residual_codebooks must be > 0; got {num_residual_codebooks=}. "
            f"Codec must have at least 2 codebooks for the cb_rest stream."
        )
    return (
        response_mask_codec.unsqueeze(-1)
        .expand(-1, -1, num_residual_codebooks)
        .flatten(1, 2)
        .contiguous()
    )


def _prefix_metrics(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Rename `actor/<name>` keys from a `compute_policy_loss_vanilla`-style
    metrics dict to `actor/<prefix>/<name>`. Other keys pass through unchanged."""
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if key.startswith("actor/"):
            out[f"actor/{prefix}/{key[len('actor/'):]}"] = value
        else:
            out[f"{prefix}/{key}" if "/" not in key else key] = value
    return out


class MultiCodebookDPActor:
    """Multi-codebook data-parallel actor.

    The recipe trainer constructs one instance per actor worker, passing the
    `model.name` from config so the right `MultiCodebookTTSModel` adapter is
    resolved from the registry.
    """

    def __init__(self, config: MultiCodebookActorConfig) -> None:
        from verl_omni.models.multi_codebook_tts.base import (
            MultiCodebookTTSModel,
            get_adapter,
        )

        self.config = config
        self.adapter: MultiCodebookTTSModel = get_adapter(config.model_name)

    # ------------------------------------------------------------------
    # compute_log_prob: invoked by the worker's `compute_log_prob` /
    #                   `compute_ref_log_prob` entry points.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_log_prob(
        self,
        data,
        model: nn.Module,
    ):
        """Recompute per-stream log-probs from the rollout sequences.

        Required `data` fields:
        - `input_ids` `[B, T_total]`
        - `codec_ids` `[B, T_codec, N]` (response-region codec tokens; all
          codebooks)
        - `attention_mask` `[B, T_total]`
        - `response_mask_codec` `[B, T_codec]` (1 on generated codec frames,
          0 on padding / post-EOS)
        - `prompt_lens` `[B]`

        Writes (does NOT mutate inputs):
        - `old_log_probs_cb0` `[B, T_codec]`
        - `old_log_probs_cb_rest` `[B, T_codec * (N-1)]`

        The legacy unified `old_log_probs` field is intentionally absent.
        """
        from tensordict import TensorDict

        out = self.adapter.forward_training(
            model=model,
            input_ids=data["input_ids"],
            codec_ids=data["codec_ids"],
            attention_mask=data["attention_mask"],
            response_mask=data.get("response_mask"),
            prompt_lens=data["prompt_lens"],
        )
        codec_ids = data["codec_ids"]
        B, T_codec, N = codec_ids.shape
        prompt_lens = data["prompt_lens"]

        cb0_lp = _gather_log_probs_cb0(
            out.talker_logits,
            codec_ids[..., 0],
            prompt_lens,
            T_codec,
        )  # [B, T_codec]
        cb_rest_lp = _gather_log_probs_cb_rest(
            out.cb_rest_logits,
            codec_ids[..., 1:],
        )  # [B, T_codec*(N-1)]

        return TensorDict(
            {
                "old_log_probs_cb0": cb0_lp,
                "old_log_probs_cb_rest": cb_rest_lp,
            },
            batch_size=(B,),
        )

    # ------------------------------------------------------------------
    # update_policy: invoked by the worker's `update_actor` entry point.
    # ------------------------------------------------------------------

    def update_policy(
        self,
        data,
        model: nn.Module,
    ) -> dict[str, Any]:
        """Compute the multi-codebook PPO loss and return scalar loss +
        per-stream metrics.

        Required `data` fields:
        - All of `compute_log_prob`'s required fields, PLUS
        - `old_log_probs_cb0` / `old_log_probs_cb_rest` (written by a prior
          `compute_log_prob` call)
        - `advantages` `[B]` (per-sample GRPO group-relative advantage)
        - Optionally `ref_log_probs_cb0` / `ref_log_probs_cb_rest` if
          `use_kl_loss` is on in either stream config (currently
          `MultiCodebookActorConfig.cb0.use_kl_loss` / `.cb_rest.use_kl_loss`).

        Returns a dict containing `actor/total_loss` (Python float) and the
        per-stream policy-loss metrics renamed under `actor/cb0/*` and
        `actor/cb_rest/*`. The caller backprops `total_loss_tensor` (also in
        the dict under `_total_loss_tensor` for autograd) before logging.
        """
        # Per-stream tensors. Required.
        old_cb0_lp = data["old_log_probs_cb0"]
        old_cb_rest_lp = data["old_log_probs_cb_rest"]
        advantages = data["advantages"]
        response_mask_codec = data["response_mask_codec"]

        codec_ids = data["codec_ids"]
        B, T_codec, N = codec_ids.shape
        N_residual = N - 1

        # Forward through the current policy (gradients enabled).
        out = self.adapter.forward_training(
            model=model,
            input_ids=data["input_ids"],
            codec_ids=codec_ids,
            attention_mask=data["attention_mask"],
            response_mask=data.get("response_mask"),
            prompt_lens=data["prompt_lens"],
        )
        new_cb0_lp = _gather_log_probs_cb0(
            out.talker_logits,
            codec_ids[..., 0],
            data["prompt_lens"],
            T_codec,
        )                                                  # [B, T_codec]
        new_cb_rest_lp = _gather_log_probs_cb_rest(
            out.cb_rest_logits,
            codec_ids[..., 1:],
        )                                                  # [B, T_codec*(N-1)]

        # Masks: per-stream layouts (AC-5).
        mask_cb0 = response_mask_codec                     # [B, T_codec]
        mask_cb_rest = _build_cb_rest_mask(mask_cb0, N_residual)
        # AC-5.5 invariant: cb_rest_mask.sum() == cb0_mask.sum() * (N-1).
        # (Asserted lazily in unit tests rather than at runtime.)

        # Advantages: broadcast per-sample [B] to each stream's [B, L] cell
        # layout via expand. AC-1 explicit broadcast.
        A_cb0 = advantages.to(new_cb0_lp.dtype).unsqueeze(-1).expand(-1, T_codec)
        A_cb_rest = advantages.to(new_cb_rest_lp.dtype).unsqueeze(-1).expand(
            -1, T_codec * N_residual,
        )

        # Dual policy-loss invocation. Each stream is normalized first (via
        # the stream's `loss_agg_mode`); the weighted sum happens AFTER.
        loss_cb0_tensor, metrics_cb0 = self._compute_stream_loss(
            stream_cfg=self.config.cb0,
            old_log_prob=old_cb0_lp,
            log_prob=new_cb0_lp,
            advantages=A_cb0,
            response_mask=mask_cb0,
            ref_log_prob=data.get("ref_log_probs_cb0"),
        )
        loss_cb_rest_tensor, metrics_cb_rest = self._compute_stream_loss(
            stream_cfg=self.config.cb_rest,
            old_log_prob=old_cb_rest_lp,
            log_prob=new_cb_rest_lp,
            advantages=A_cb_rest,
            response_mask=mask_cb_rest,
            ref_log_prob=data.get("ref_log_probs_cb_rest"),
        )

        # Fish-S2 weighted sum (AC-1). Per-stream normalization already
        # happened inside `get_policy_loss_fn(...)`; `w_cb_rest=0.1` is
        # NOT diluted by the larger cell count of cb_rest.
        total_loss_tensor = (
            self.config.w_cb0 * loss_cb0_tensor
            + self.config.w_cb_rest * loss_cb_rest_tensor
        )

        # Metrics: prefix per-stream policy metrics under actor/cb0/* +
        # actor/cb_rest/* (AC-4).
        out_metrics: dict[str, Any] = {}
        out_metrics.update(_prefix_metrics(metrics_cb0, "cb0"))
        out_metrics.update(_prefix_metrics(metrics_cb_rest, "cb_rest"))
        out_metrics["actor/total_loss"] = float(total_loss_tensor.detach().cpu())
        # Tensor handle so the recipe trainer can backprop without re-computing.
        out_metrics["_total_loss_tensor"] = total_loss_tensor
        return out_metrics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_stream_loss(
        self,
        stream_cfg: _StreamLossConfig,
        old_log_prob: torch.Tensor,
        log_prob: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        ref_log_prob: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Call `get_policy_loss_fn(stream_cfg.loss_mode)` with the per-stream
        tensors. Optionally adds a KL penalty against `ref_log_prob`."""
        # Lazy import: `verl.trainer.ppo.core_algos` pulls in `verl.utils.*`
        # which can take a few seconds; defer until update_policy runs.
        from verl.trainer.ppo import core_algos as _ppo_core

        policy_loss_fn = _ppo_core.get_policy_loss_fn(stream_cfg.loss_mode)

        # Build the per-stream config object the policy-loss fn expects. The
        # vanilla policy loss reads `config.policy_loss.clip_ratio`; using
        # `SimpleNamespace` keeps us decoupled from upstream's dataclass.
        from types import SimpleNamespace
        loss_cfg = SimpleNamespace(
            policy_loss=SimpleNamespace(clip_ratio=stream_cfg.clip_ratio),
        )

        pg_loss, metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode=stream_cfg.loss_agg_mode,
            config=loss_cfg,
        )

        if stream_cfg.use_kl_loss and ref_log_prob is not None:
            # Low-variance KL estimator: kl = exp(ref - new) - (ref - new) - 1.
            # Apply over the stream's response_mask before reducing.
            diff = (ref_log_prob - log_prob).clamp(-30.0, 30.0)
            kl_per_position = diff.exp() - diff - 1.0
            kl_loss = (kl_per_position * response_mask).sum() / response_mask.sum().clamp_min(1.0)
            pg_loss = pg_loss + stream_cfg.kl_loss_coef * kl_loss
            metrics["actor/kl_loss"] = float(kl_loss.detach().cpu())

        return pg_loss, metrics


__all__ = [
    "MultiCodebookDPActor",
    "MultiCodebookActorConfig",
    "_StreamLossConfig",
]
