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
"""TTS-specific subclass of the vllm-omni rollout server.

The diffusion server in :mod:`vllm_omni_async_server` builds
:class:`OmniDiffusionSamplingParams`, threads ``image_data`` / ``video_data``,
and reads ``final_res.images[0]`` — image/video only. The Qwen3-TTS recipe
needs a parallel sibling that:

1. Loads the verl-omni-side stage config (``final_output: true`` and
   ``logprobs: 1`` on stage 0) so per-token codec logprobs are surfaced.
2. Exposes :meth:`generate_tts` returning :class:`AudioRolloutOutput` —
   one ``CompletionAudio`` per grouped sample (``n>=2``) containing the
   codec token IDs, per-token logprobs, and waveform.
3. Rejects the diffusion-shaped :meth:`generate` with
   :class:`UnsupportedOutputTypeError` so misuse fails fast.
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from dataclasses import asdict
from typing import Any, ClassVar
from uuid import uuid4

import ray
import vllm_omni.entrypoints.cli.serve
from verl.utils.import_utils import import_external_libs
from verl.workers.rollout.utils import run_uvicorn
from vllm.entrypoints.openai.api_server import build_app
from vllm.sampling_params import SamplingParams
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.entrypoints import AsyncOmni
from vllm_omni.entrypoints.openai.api_server import omni_init_app_state
from vllm_omni.inputs.data import OmniCustomPrompt
from vllm_omni.outputs import OmniRequestOutput

from verl_omni.pipelines.qwen3_tts_grpo import STAGE_CONFIG_PATH
from verl_omni.workers.rollout.replica import (
    AudioRolloutOutput,
    CompletionAudio,
    UnsupportedOutputTypeError,
)
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import (
    vLLMOmniHttpServer,
    vLLMOmniReplica,
)

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

_DEFAULT_AUDIO_SAMPLE_RATE = 24000


class vLLMOmniTTSHttpServer(vLLMOmniHttpServer):
    """Qwen3-TTS-specific vllm-omni HTTP rollout server.

    Loads the verl-omni-side stage_config so stage 0 emits codec tokens +
    per-token logprobs (required by GRPO).
    """

    # Methods whose ``collective_rpc`` must be routed only to stage 0
    # (the AR codec talker). Stage 1 is the code2wav decoder and has no
    # FSDP-side weights to receive; if we fan ``update_weights_from_ipc``
    # to both stages they race on the same ZMQ socket and one stage
    # hangs forever waiting for buckets that never arrive. See
    # ``vllm_omni/engine/orchestrator.py:761-762`` — when ``stage_ids``
    # is not passed the orchestrator defaults to all stages.
    _STAGE0_ONLY_RPC_METHODS: ClassVar[frozenset[str]] = frozenset({
        "update_weights_from_ipc",
    })

    async def collective_rpc(  # type: ignore[override]
        self,
        method,
        timeout=None,
        args: tuple = (),
        kwargs=None,
    ):
        stage_ids = [0] if method in self._STAGE0_ONLY_RPC_METHODS else None
        await self.engine.collective_rpc(
            method=method,
            timeout=timeout,
            args=args,
            kwargs=kwargs,
            stage_ids=stage_ids,
        )

    def _init_model_config(self, model_config):  # type: ignore[override]
        # The diffusion-side base class coerces the incoming model_config
        # into a ``DiffusionModelConfig`` (which drops ``hf_config_path``,
        # ``override_config``, etc.). Qwen3-TTS is an autoregressive LM
        # behind a vLLM engine, so it needs ``HFModelConfig`` instead —
        # the AR rollout path reads ``hf_config_path`` to discover the
        # tokenizer / generation_config / attn_implementation override.
        from verl.utils.config import omega_conf_to_dataclass
        from verl.workers.config.model import HFModelConfig

        return omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)

    async def run_server(self, args: argparse.Namespace) -> None:  # type: ignore[override]
        engine_args = OmniEngineArgs.from_cli_args(args)
        engine_args = asdict(engine_args)

        engine_args["stage_configs_path"] = str(STAGE_CONFIG_PATH)
        logger.info("[vLLMOmniTTSHttpServer] stage_configs_path=%s", engine_args["stage_configs_path"])

        # The diffusion base reads ``self.config.external_lib`` but on the
        # AR-TTS path ``self.config`` is a ``RolloutConfig`` (no
        # ``external_lib`` field). The model-side external_lib lives on
        # ``self.model_config`` (an ``HFModelConfig``). Fall back to None
        # for either layout so we don't fail when the field is absent.
        external_lib = getattr(self.config, "external_lib", None) or getattr(
            self.model_config, "external_lib", None
        )
        import_external_libs(external_lib)

        engine_client = AsyncOmni(**engine_args)
        app = build_app(args)
        await omni_init_app_state(engine_client, app.state, args)

        self.engine = engine_client
        self._server_port, self._server_task = await run_uvicorn(app, args, self._server_address)

    async def generate(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        """Reject diffusion-shaped requests routed here by mistake."""

        raise UnsupportedOutputTypeError(
            "vLLMOmniTTSHttpServer is configured for Qwen3-TTS audio rollouts; "
            "use generate_tts(...) instead of generate(...). The diffusion path "
            "returns DiffusionOutput with .images[0], which is incorrect for "
            "audio outputs."
        )

    async def generate_tts(
        self,
        prompt_text: str,
        ref_audio: Any,
        ref_text: str,
        *,
        sampling_params: dict[str, Any],
        request_id: str | None = None,
        n: int | None = None,
        task_type: str = "Base",
    ) -> AudioRolloutOutput:
        """Sample ``n>=2`` speech-token sequences for one ``(prompt_text, ref_audio, ref_text)`` triple.

        Returns an :class:`AudioRolloutOutput` whose ``completions`` list has
        one entry per grouped sample, each carrying the stage-0 codec tokens,
        per-token logprobs, and the stage-1 waveform — correlated by
        ``(request_id, sample_index)``.

        Raises:
            ValueError: ``n`` is missing, less than 2, or not an integer.
        """

        if n is None or not isinstance(n, int):
            raise ValueError(
                f"generate_tts requires an integer n>=2 (got n={n!r}). "
                "Group size is mandatory for GRPO; pass it explicitly."
            )
        if n < 2:
            raise ValueError(
                f"generate_tts requires n>=2 for group-relative GRPO sampling (got n={n}). "
                "A single sample cannot form a group."
            )

        request_id = request_id or uuid4().hex

        # Build prompt for Qwen3-TTS Base mode.
        #
        # vllm-omni 0.18 switched from the OmniCustomPrompt({"extra_args":...})
        # shape to vLLM's standard decoder-only schema: a dict with a top-level
        # ``"prompt"`` (str) plus an ``"additional_information"`` field whose
        # values are *lists* (one per group sample) of the side-channel inputs.
        # See BL-20260517-qwen3-tts-prompt-and-stage-shape.
        additional_information = {
            "text":      [prompt_text],
            "ref_audio": [ref_audio],
            "ref_text":  [ref_text],
            "task_type": [task_type],
        }
        custom_prompt = {
            "prompt": prompt_text,
            "additional_information": additional_information,
        }

        # Stage 0 sampling params: standard vLLM SamplingParams; logprobs is
        # provided by the yaml override but we re-assert it here so the AR
        # scheduler always attaches per-token logprobs to EngineCoreOutput.
        #
        # Note on repetition_penalty: vLLM's ``apply_penalties``
        # (``vllm/model_executor/layers/utils.py:87``) computes
        # ``get_token_bin_counts_and_mask(prompt_tokens_tensor, vocab_size=logits.shape[1], ...)``
        # which scatter-adds the prompt token IDs into a bucket indexed
        # by *output* vocab size. For Qwen3-TTS stage 0 the logits vocab
        # is the codec vocab (~3072), but the prompt token IDs are HF
        # tokenizer IDs (vocab ~151K). Any prompt ID >= 3072 makes the
        # scatter raise ``CUDA error: device-side assert triggered``
        # (``ScatterGatherKernel.cu:163``: ``scatter gather kernel index
        # out of bounds``). Hard-forcing repetition_penalty=1.0 (no
        # penalty) bypasses ``apply_penalties`` entirely, which is the
        # right behavior for codec sampling anyway — repetition penalty
        # is text-LM-specific.
        stage0_sampling = SamplingParams(
            n=n,
            temperature=sampling_params.get("temperature", 0.9),
            top_p=sampling_params.get("top_p", 1.0),
            top_k=sampling_params.get("top_k", 50),
            max_tokens=sampling_params.get("max_tokens", 4096),
            seed=sampling_params.get("seed"),
            logprobs=int(sampling_params.get("logprobs", 1)),
            repetition_penalty=1.0,
            stop_token_ids=sampling_params.get("stop_token_ids", [2150]),
        )

        # Stage 1 (code2wav): deterministic decoder. vllm-omni's orchestrator
        # calls ``params.clone()`` on the stage-1 SamplingParams in
        # ``_prewarm_async_chunk_stages`` (see
        # vllm_omni/engine/orchestrator.py:65,673), so passing ``None``
        # raises ``AttributeError: 'NoneType' object has no attribute 'clone'``.
        # Construct a concrete SamplingParams — code2wav doesn't sample
        # autoregressively but the orchestrator still needs an object to
        # clone. See BL-20260517-qwen3-tts-prompt-and-stage-shape (the
        # stage-1 fix that the live smoke needed).
        stage1_sampling = SamplingParams(
            temperature=0.0,
            max_tokens=65536,
            detokenize=True,
        )

        sampling_params_list = [stage0_sampling, stage1_sampling]

        generator = self.engine.generate(
            prompt=custom_prompt,
            request_id=request_id,
            sampling_params_list=sampling_params_list,
        )

        # Stage-0 buffers: per (request_id, sample_index) -> (token_ids, logprobs, finish_reason)
        stage0: dict[int, dict[str, Any]] = {}
        # Stage-1 buffers: per sample_index -> waveform
        stage1: dict[int, Any] = {}
        final_stop_reason: str | None = None
        num_preempted: int | None = None
        sample_rate = _DEFAULT_AUDIO_SAMPLE_RATE

        async for omni_out in generator:
            if isinstance(omni_out, OmniRequestOutput) and omni_out.request_output is not None:
                if omni_out.stage_id == 0:
                    for completion in omni_out.outputs:
                        index = getattr(completion, "index", 0)
                        token_ids = list(getattr(completion, "token_ids", []) or [])
                        logprobs_raw = getattr(completion, "logprobs", None) or []
                        per_token_logprobs: list[float] = []
                        for step in logprobs_raw:
                            if not step:
                                continue
                            # vLLM Logprobs is a dict[token_id, Logprob]. We pick
                            # the Logprob of the sampled token (the first entry
                            # when logprobs=1 — vLLM puts the sampled token first).
                            first = next(iter(step.values()))
                            per_token_logprobs.append(float(getattr(first, "logprob", first)))
                        if not token_ids:
                            continue
                        stage0[index] = {
                            "codec_tokens": token_ids,
                            "logprobs": per_token_logprobs,
                            "finish_reason": getattr(completion, "finish_reason", None),
                        }
                elif omni_out.stage_id == 1:
                    mm = omni_out.multimodal_output or {}
                    audio = mm.get("audio")
                    if audio is None:
                        # Some configs surface the waveform under 'waveform'.
                        audio = mm.get("waveform")
                    sr = mm.get("sample_rate")
                    if sr is not None:
                        sample_rate = int(sr)
                    # When stage 1 batches audio across the n samples, treat the
                    # whole multimodal_output as a list-of-waveforms; otherwise
                    # store it under index 0 and rely on stage-0 grouping.
                    if isinstance(audio, list):
                        for i, wav in enumerate(audio):
                            stage1[i] = wav
                    else:
                        stage1[0] = audio
                # Capture finish/preemption metadata from the last stage.
                req_output = omni_out.request_output
                fr = getattr(req_output, "finish_reason", None)
                if fr:
                    final_stop_reason = "completed" if fr in ("stop", "length") else fr
                np_ = getattr(req_output, "num_preempted", None)
                if np_ is not None:
                    num_preempted = np_

        # Join stage-0 + stage-1 per sample_index.
        completions: list[CompletionAudio] = []
        for index in sorted(stage0.keys()):
            waveform = stage1.get(index, stage1.get(0))
            data = stage0[index]
            completions.append(
                CompletionAudio(
                    sample_index=index,
                    codec_tokens=data["codec_tokens"],
                    logprobs=data["logprobs"],
                    waveform=waveform,
                    finish_reason=data["finish_reason"],
                )
            )

        if not completions:
            raise RuntimeError(
                f"generate_tts request {request_id} produced no stage-0 completions. "
                "Check that the verl-omni stage_config override sets final_output:true "
                "on stage 0 and that the engine was launched with stage_configs_path."
            )

        return AudioRolloutOutput(
            completions=completions,
            sample_rate=sample_rate,
            stop_reason=final_stop_reason,
            num_preempted=num_preempted,
        )


class vLLMOmniTTSReplica(vLLMOmniReplica):
    """Replica that launches :class:`vLLMOmniTTSHttpServer`."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.server_class = ray.remote(vLLMOmniTTSHttpServer)

    def _get_server_name_prefix(self) -> str:  # type: ignore[override]
        return "vllm_omni_tts_"


# Keep a reference to vllm_omni.entrypoints.cli.serve so its CLI registration
# survives import-order changes — the diffusion server already imports it.
_ = vllm_omni.entrypoints.cli.serve
