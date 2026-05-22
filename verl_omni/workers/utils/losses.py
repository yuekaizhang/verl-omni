# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging

from tensordict import TensorDict
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, Metric
from verl.workers.utils.padding import no_padding_2_padding

from verl_omni.trainer.diffusion.diffusion_algos import get_diffusion_loss_fn
from verl_omni.workers.config import DiffusionActorConfig

_LOG = logging.getLogger(__name__)
_MULTI_CODEBOOK_CB_REST_MISSING_LOGGED = False


def diffusion_loss(config: DiffusionActorConfig, model_output, data: TensorDict, dp_group=None):
    """Compute loss for diffusion model"""
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    metrics = {}

    loss_mode = config.diffusion_loss.get("loss_mode", "flow_grpo")
    loss_func = get_diffusion_loss_fn(loss_mode)
    loss_func.validate_inputs(loss_name=loss_mode, model_output=model_output, data=data)
    loss_result = loss_func(config=config, model_output=model_output, data=data)
    loss_value = loss_result.loss
    metrics_values = loss_result.metrics

    metrics_values = Metric.from_dict(metrics_values, aggregation=AggregationType.MEAN)

    metrics.update(metrics_values)
    if loss_result.add_loss_metric:
        metrics["actor/loss"] = Metric(value=loss_value, aggregation=AggregationType.MEAN)

    if config.use_kl_loss:
        loss_func = get_diffusion_loss_fn("kl")
        loss_func.validate_inputs(loss_name="kl", model_output=model_output, data=data)
        kl_result = loss_func(config=config, model_output=model_output, data=data)
        loss_value += kl_result.loss * config.kl_loss_coef
        metrics.update(Metric.from_dict(kl_result.metrics, aggregation=AggregationType.MEAN))
        metrics["kl_coef"] = config.kl_loss_coef
        if kl_result.add_loss_metric:
            metrics["actor/weighted_kl_loss"] = Metric(
                value=kl_result.loss * config.kl_loss_coef,
                aggregation=AggregationType.MEAN,
            )

    gradient_accumulation_steps = tu.get_non_tensor_data(data, "gradient_accumulation_steps", default=None)
    loss_value = loss_value / gradient_accumulation_steps

    sp_size = tu.get_non_tensor_data(data, "sp_size", default=None)
    if sp_size > 1:
        loss_value = loss_value * sp_size

    return loss_value, metrics


def _per_stream_policy_loss(
    *,
    stream_name: str,
    loss_mode: str,
    loss_agg_mode: str,
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    ref_log_prob=None,
    use_kl_loss: bool = False,
    kl_loss_coef: float = 0.0,
    kl_loss_type: str = "low_var_kl",
    config=None,
    rollout_is_weights=None,
    global_batch_info=None,
):
    """Run `verl.trainer.ppo.core_algos.get_policy_loss_fn(loss_mode)` once for a
    single stream and rename the metrics under `actor/<stream_name>/*`.

    Optional KL penalty against `ref_log_prob`. Aggregation uses verl's
    `agg_loss` so the global_batch_info path matches upstream ppo_loss.
    """
    policy_loss_fn = get_policy_loss_fn(loss_mode)
    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )
    # Rename per-stream metrics: actor/<x> -> actor/<stream>/<x>.
    renamed: dict = {}
    for k, v in pg_metrics.items():
        if k.startswith("actor/"):
            renamed[f"actor/{stream_name}/{k[len('actor/'):]}"] = v
        else:
            renamed[k] = v
    total = pg_loss

    if use_kl_loss and ref_log_prob is not None:
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=kl_loss_type)
        kl = agg_loss(
            loss_mat=kld,
            loss_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            **(global_batch_info or {}),
        )
        total = total + kl * kl_loss_coef
        renamed[f"actor/{stream_name}/kl_loss"] = kl

    return total, renamed


def multi_codebook_ppo_loss(config, model_output, data: TensorDict, dp_group=None):
    """Fish-S2-style multi-codebook PPO loss.

    Composes two independent per-stream `get_policy_loss_fn(loss_mode)`
    invocations (cb0 + cb_rest), each normalized first with its own
    `loss_agg_mode: token-mean`, then combined as
    `total_loss = w_cb0 * L_cb0 + w_cb_rest * L_cb_rest`. Metrics are
    emitted under `actor/cb0/*` and `actor/cb_rest/*` per AC-4.

    Drop-in replacement for upstream `verl.workers.utils.losses.ppo_loss`
    (same `(config, model_output, data, dp_group)` signature) selected
    via the `engine_workers` dispatch when `actor.multi_codebook_loss` is
    truthy in the config.

    The cb_rest stream needs `model_output["log_probs_cb_rest"]` and
    `data["old_log_probs_cb_rest"]`. When the FSDP engine has not yet
    been extended to populate `log_probs_cb_rest` (the v1 state — only
    cb0's `log_probs` is emitted by upstream's logprob extractor), the
    cb_rest contribution falls back to zero and a one-time `logger.warning`
    flags the gap. The cb0 metrics + total_loss still emit so the
    smoke run validates the wiring end-to-end. Removing the fallback
    after the FSDP engine extension lands (see follow-up M5 in the plan)
    enables the full Fish-S2 loss path.
    """
    global _MULTI_CODEBOOK_CB_REST_MISSING_LOGGED

    # 1) cb0 stream — same path as upstream ppo_loss.
    log_prob = no_padding_2_padding(model_output["log_probs"], data)

    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    if (
        data["dp_size"] > 1
        or data["batch_num_tokens"] is not None
        or data["global_batch_size"] is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    # Per-stream config knobs. Read from the new actor YAML; fall back to
    # upstream-style defaults when missing so the loss is robust under
    # config-shape drift.
    cb0_cfg = config.get("cb0", None) or {}
    cb_rest_cfg = config.get("cb_rest", None) or {}
    w_cb0 = float(config.get("w_cb0", 1.0))
    w_cb_rest = float(config.get("w_cb_rest", 0.1))

    cb0_loss_mode = cb0_cfg.get("policy_loss", {}).get("loss_mode", "vanilla")
    cb0_agg = cb0_cfg.get("loss_agg_mode", config.loss_agg_mode)
    cb_rest_loss_mode = cb_rest_cfg.get("policy_loss", {}).get("loss_mode", "vanilla")
    cb_rest_agg = cb_rest_cfg.get("loss_agg_mode", config.loss_agg_mode)

    fields = ["response_mask", "old_log_probs", "advantages"]
    if "rollout_is_weights" in data:
        fields.append("rollout_is_weights")
    if "ref_log_prob" in data:
        fields.append("ref_log_prob")
    has_cb_rest = "log_probs_cb_rest" in model_output and "old_log_probs_cb_rest" in data
    if has_cb_rest:
        fields.append("old_log_probs_cb_rest")
        if "ref_log_prob_cb_rest" in data:
            fields.append("ref_log_prob_cb_rest")
    padded = data.select(*fields).to_padded_tensor()

    response_mask = padded["response_mask"].to(bool)
    cb0_total, cb0_metrics = _per_stream_policy_loss(
        stream_name="cb0",
        loss_mode=cb0_loss_mode,
        loss_agg_mode=cb0_agg,
        old_log_prob=padded["old_log_probs"],
        log_prob=log_prob,
        advantages=padded["advantages"],
        response_mask=response_mask,
        ref_log_prob=padded.get("ref_log_prob"),
        use_kl_loss=bool(cb0_cfg.get("use_kl_loss", False)),
        kl_loss_coef=float(cb0_cfg.get("kl_loss_coef", 0.0)),
        kl_loss_type=str(cb0_cfg.get("kl_loss_type", "low_var_kl")),
        config=config,
        rollout_is_weights=padded.get("rollout_is_weights"),
        global_batch_info=config.global_batch_info,
    )

    # 2) cb_rest stream — present only when the FSDP engine has been
    # extended to populate `log_probs_cb_rest` in model_output.
    if has_cb_rest:
        log_prob_cb_rest = no_padding_2_padding(model_output["log_probs_cb_rest"], data)
        # cb_rest mask broadcast = response_mask repeated `(N-1)` times across
        # the residual codebook axis; the FSDP engine is expected to ship the
        # mask under `response_mask_cb_rest` for clarity. If absent, derive it.
        if "response_mask_cb_rest" in data:
            cb_rest_mask = data["response_mask_cb_rest"].to(bool)
        else:
            # Best-effort: assume cb_rest carries `T * (N-1)` cells per sample;
            # broadcast response_mask. The shape mismatch (if any) surfaces
            # immediately inside `compute_policy_loss_vanilla`.
            cb_rest_mask = response_mask
        cb_rest_advantages = padded["advantages"]
        if "advantages_cb_rest" in data:
            cb_rest_advantages = data["advantages_cb_rest"]

        cb_rest_total, cb_rest_metrics = _per_stream_policy_loss(
            stream_name="cb_rest",
            loss_mode=cb_rest_loss_mode,
            loss_agg_mode=cb_rest_agg,
            old_log_prob=padded["old_log_probs_cb_rest"],
            log_prob=log_prob_cb_rest,
            advantages=cb_rest_advantages,
            response_mask=cb_rest_mask,
            ref_log_prob=padded.get("ref_log_prob_cb_rest"),
            use_kl_loss=bool(cb_rest_cfg.get("use_kl_loss", False)),
            kl_loss_coef=float(cb_rest_cfg.get("kl_loss_coef", 0.0)),
            kl_loss_type=str(cb_rest_cfg.get("kl_loss_type", "low_var_kl")),
            config=config,
            rollout_is_weights=None,
            global_batch_info=config.global_batch_info,
        )
    else:
        if not _MULTI_CODEBOOK_CB_REST_MISSING_LOGGED:
            _LOG.warning(
                "[multi_codebook_ppo_loss] model_output is missing "
                "'log_probs_cb_rest' (FSDP engine has not been extended to "
                "produce per-stream log-probs). cb_rest contribution falls "
                "back to 0 for this step; only cb0 contributes to the "
                "gradient. AC-4 metric keys (`actor/cb_rest/*`) will not "
                "appear until the engine extension lands. Suppressing "
                "subsequent warnings."
            )
            _MULTI_CODEBOOK_CB_REST_MISSING_LOGGED = True
        # Zero scalar with the same dtype/device as cb0_total so the
        # `total_loss` weighted sum is well-typed.
        cb_rest_total = cb0_total.detach() * 0.0
        cb_rest_metrics = {}

    # 3) Fish-S2 weighted sum (AC-1). Per-stream normalization already
    # happened inside `get_policy_loss_fn(...)`; w_cb_rest is not diluted.
    total_loss = w_cb0 * cb0_total + w_cb_rest * cb_rest_total

    metrics: dict = {}
    metrics.update(Metric.from_dict(cb0_metrics, aggregation=AggregationType.MEAN))
    metrics["actor/cb0/pg_loss"] = Metric(value=cb0_total, aggregation=metric_aggregation)
    if cb_rest_metrics:
        metrics.update(Metric.from_dict(cb_rest_metrics, aggregation=AggregationType.MEAN))
        metrics["actor/cb_rest/pg_loss"] = Metric(value=cb_rest_total, aggregation=metric_aggregation)
    metrics["actor/total_loss"] = Metric(value=total_loss, aggregation=metric_aggregation)
    # Preserve `actor/loss` for downstream observability tooling that hard-codes
    # the legacy key.
    metrics["actor/loss"] = Metric(value=total_loss, aggregation=metric_aggregation)

    return total_loss, metrics
