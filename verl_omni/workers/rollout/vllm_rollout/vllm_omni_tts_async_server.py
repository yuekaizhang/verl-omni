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
import asyncio
import logging
import os
from collections import defaultdict
from dataclasses import asdict
from typing import Any, ClassVar
from uuid import uuid4

import numpy as np
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

from verl_omni.pipelines.multi_codebook_tts_grpo import STAGE_CONFIG_PATH
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

    def _estimate_prompt_len(
        self,
        additional_information: dict[str, Any],
    ) -> int:
        """Compute the placeholder ``prompt_token_ids`` length the talker expects.

        The Qwen3-TTS talker's ``preprocess`` (qwen3_tts_talker.py:511)
        replaces *all* input embeddings via ``_build_prompt_embeds``, but
        the placeholder length must equal the resulting embedding length
        — otherwise the ICL prefill region gets misaligned and the model
        ends up emitting the ref_text prefix instead of synthesizing the
        target ``text``. See BL-20260518-qwen3-tts-prompt-token-ids-placeholder.

        Mirrors vllm-omni's official offline-inference demo at
        ``vllm-omni/examples/offline_inference/text_to_speech/qwen3_tts
        /end2end.py:_estimate_prompt_len``. Tokenizer + talker_config +
        speech_tokenizer are cached on the instance so the cost is one
        load per replica, not per-request.
        """

        from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
            Qwen3TTSTalkerForConditionalGeneration,
        )

        cache = getattr(self, "_qwen3_tts_estimate_cache", None)
        if cache is None:
            from transformers import AutoTokenizer
            from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (
                Qwen3TTSConfig,
            )

            model_name = self.model_config.path
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True, padding_side="left",
            )
            tts_config = Qwen3TTSConfig.from_pretrained(model_name, trust_remote_code=True)
            speech_tok = None
            try:
                from transformers.utils import cached_file
                from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_tokenizer import (
                    Qwen3TTSTokenizer,
                )

                import torch

                st_cfg_path = cached_file(model_name, "speech_tokenizer/config.json")
                if st_cfg_path:
                    speech_tok = Qwen3TTSTokenizer.from_pretrained(
                        os.path.dirname(st_cfg_path), torch_dtype=torch.bfloat16,
                    )
            except Exception as exc:
                logger.debug("Could not load speech tokenizer: %s", exc)
            cache = (tokenizer, getattr(tts_config, "talker_config", None), speech_tok)
            self._qwen3_tts_estimate_cache = cache
        tokenizer, talker_config, speech_tok = cache

        task_type = (additional_information.get("task_type") or ["Base"])[0]

        def _estimate_ref_code_len(ref_audio):
            """Encode ref_audio via the speech tokenizer to get the exact codec frame count.

            Falls back to a duration-based estimate (12 Hz codec frame rate
            for Qwen3-TTS-12Hz-*-Base) when speech_tokenizer is unavailable
            or the encode raises.
            """
            if not isinstance(ref_audio, (str, list)):
                return None
            audio_path = ref_audio[0] if isinstance(ref_audio, list) else ref_audio
            if not isinstance(audio_path, str) or not audio_path.strip():
                return None
            try:
                import soundfile as sf

                audio, sr = sf.read(audio_path, always_2d=False)
                if audio.ndim > 1:
                    audio = audio[:, 0]
                wav_np = np.asarray(audio, dtype=np.float32)
            except Exception:
                return None
            if speech_tok is not None:
                try:
                    enc = speech_tok.encode(wav_np, sr=int(sr), return_dict=True)
                    ref_code = getattr(enc, "audio_codes", None)
                    if isinstance(ref_code, list):
                        ref_code = ref_code[0] if ref_code else None
                    if ref_code is not None and hasattr(ref_code, "shape"):
                        shape = ref_code.shape
                        if len(shape) == 2:
                            return int(shape[0])
                        if len(shape) == 3:
                            return int(shape[1])
                except Exception as exc:
                    logger.debug("speech_tok.encode failed; using duration fallback: %s", exc)
            codec_hz = getattr(talker_config, "codec_frame_rate", None) or 12
            return int(len(wav_np) / sr * codec_hz)

        return Qwen3TTSTalkerForConditionalGeneration.estimate_prompt_len_from_additional_information(
            additional_information=additional_information,
            task_type=task_type,
            tokenize_prompt=lambda t: tokenizer(t, padding=False)["input_ids"],
            codec_language_id=getattr(talker_config, "codec_language_id", None),
            spk_is_dialect=getattr(talker_config, "spk_is_dialect", None),
            estimate_ref_code_len=_estimate_ref_code_len,
        )

    async def _generate_one_tts_sample(
        self,
        *,
        sub_request_id: str,
        custom_prompt: dict[str, Any],
        sampling_params_list: list[SamplingParams],
        sample_index: int,
    ) -> dict[str, Any]:
        """Generate ONE codec/waveform pair for the given sub-request.

        Returns a dict with ``codec_tokens``, ``logprobs``, ``waveform``,
        ``sample_rate``, ``finish_reason``, ``num_preempted``. The caller
        is responsible for combining N of these into an AudioRolloutOutput.
        """
        codec_tokens: list[int] = []
        logprobs: list[float] = []
        finish_reason: str | None = None
        num_preempted: int | None = None
        sample_rate: int = _DEFAULT_AUDIO_SAMPLE_RATE
        stage1_audio: np.ndarray | None = None

        async for omni_out in self.engine.generate(
            prompt=custom_prompt,
            request_id=sub_request_id,
            sampling_params_list=sampling_params_list,
        ):
            if not isinstance(omni_out, OmniRequestOutput):
                continue
            if omni_out.request_output is None:
                continue
            if omni_out.stage_id == 0:
                # Collect codec tokens + per-token logprobs from the
                # *latest* stage-0 event for this sub-request (each event
                # carries the cumulative codec sequence, so the final
                # event has the full sequence).
                for completion in omni_out.outputs:
                    token_ids = list(getattr(completion, "token_ids", []) or [])
                    if not token_ids:
                        continue
                    codec_tokens = token_ids
                    lp_raw = getattr(completion, "logprobs", None) or []
                    per_step: list[float] = []
                    for step in lp_raw:
                        if not step:
                            continue
                        first = next(iter(step.values()))
                        per_step.append(float(getattr(first, "logprob", first)))
                    logprobs = per_step
                    fr = getattr(completion, "finish_reason", None)
                    if fr:
                        finish_reason = "completed" if fr in ("stop", "length") else fr
            elif omni_out.stage_id == 1:
                mm = omni_out.multimodal_output or {}
                audio = None
                sr_raw = None
                if isinstance(mm, dict):
                    # Explicit None-fallback chain instead of ``or`` —
                    # ``mm[key]`` can be a torch.Tensor with >1 element,
                    # and ``Tensor or x`` raises ``RuntimeError: Boolean
                    # value of Tensor with more than one value is ambiguous``.
                    for k in ("audio", "waveform", "model_outputs"):
                        v = mm.get(k)
                        if v is not None:
                            audio = v
                            break
                    for k in ("sample_rate", "sr"):
                        v = mm.get(k)
                        if v is not None:
                            sr_raw = v
                            break
                    if isinstance(sr_raw, list) and sr_raw:
                        sr_raw = sr_raw[0]
                    if hasattr(sr_raw, "item"):
                        try:
                            sr_raw = int(sr_raw.item())
                        except Exception:
                            sr_raw = None
                    if sr_raw is not None:
                        sample_rate = int(sr_raw)
                # ``codec_streaming: true`` makes stage 1 emit a *cumulative*
                # list of audio chunks across successive events. Overwriting
                # with the concat of the latest list each time yields the
                # full waveform once the final event arrives.
                if isinstance(audio, list) and audio:
                    chunks = [np.asarray(a) for a in audio if a is not None]
                    chunks = [c for c in chunks if c.size > 0]
                    if chunks:
                        stage1_audio = np.concatenate(chunks)
                elif audio is not None:
                    wav_np = np.asarray(audio)
                    if wav_np.size > 0:
                        stage1_audio = wav_np
            np_ = getattr(omni_out.request_output, "num_preempted", None)
            if np_ is not None:
                num_preempted = np_

        return {
            "sample_index": sample_index,
            "codec_tokens": codec_tokens,
            "logprobs": logprobs,
            "waveform": stage1_audio,
            "sample_rate": sample_rate,
            "finish_reason": finish_reason,
            "num_preempted": num_preempted,
        }

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
        language: str | None = None,
    ) -> AudioRolloutOutput:
        """Sample ``n>=2`` speech-token sequences for one ``(prompt_text, ref_audio, ref_text)`` triple.

        vllm-omni 0.18's orchestrator hardcodes ``parent_req=None,
        request_index=0`` at both ``orchestrator.py:527-533`` and
        ``:681-687``, so ``SamplingParams.n>1`` collapses to a single
        completion. To get GRPO-grouped completions, this method emits
        ``n`` independent requests (different seeds, same prompt) and
        gathers them via :func:`asyncio.gather`. Each sub-request's
        codec_tokens + logprobs + waveform become one ``CompletionAudio``
        in the returned :class:`AudioRolloutOutput`.

        The payload shape mirrors the upstream vllm-omni end2end demo
        (``offline_inference/text_to_speech/qwen3_tts/end2end.py``):

        * ``prompt_token_ids = [0] * estimate_prompt_len(additional_information)``
          — a placeholder of the talker's expected prefill length. Passing
          the raw ``prompt_text`` as ``"prompt"`` (the previous behaviour)
          let vLLM tokenize it independently, producing a span_len that
          didn't match the talker's ICL prefill embedding length and
          causing the model to emit the ref_text prefix instead of
          synthesizing ``text``. See
          BL-20260518-qwen3-tts-prompt-token-ids-placeholder.
        * ``additional_information``: ``task_type, ref_audio, ref_text,
          text, language, x_vector_only_mode, max_new_tokens`` (lists of
          length 1 each — the Qwen3-TTS talker side-channel schema).

        Raises:
            ValueError: ``n`` is missing, less than 2, or not an integer.
        """

        if n is None or not isinstance(n, int):
            raise ValueError(
                f"generate_tts requires an integer n>=1 (got n={n!r}). "
                "Pass it explicitly."
            )
        if n < 1:
            raise ValueError(
                f"generate_tts requires n>=1 (got n={n}). When the caller "
                "(verl's ``ray_trainer.fit``) already pre-expands the input "
                "batch by ``rollout.n`` via ``batch.repeat(repeat_times=n)``, "
                "each agent_loop call should request n=1 here so the output "
                "cardinality matches the pre-expanded batch."
            )

        request_id = request_id or uuid4().hex
        max_tokens = int(sampling_params.get("max_tokens", 4096))

        additional_information: dict[str, Any] = {
            "task_type":          [task_type],
            "ref_audio":          [ref_audio],
            "ref_text":           [ref_text],
            "text":               [prompt_text],
            "x_vector_only_mode": [False],
            "max_new_tokens":     [max_tokens],
        }
        if language is not None:
            additional_information["language"] = [language]

        # Placeholder-length prompt_token_ids — the talker replaces these
        # embeddings entirely during preprocess but the length must match.
        prompt_len = self._estimate_prompt_len(additional_information)
        custom_prompt = {
            "prompt_token_ids": [0] * int(prompt_len),
            "additional_information": additional_information,
        }

        # Stage 1 (code2wav): deterministic decoder. vllm-omni's orchestrator
        # calls ``params.clone()`` on the stage-1 SamplingParams in
        # ``_prewarm_async_chunk_stages`` (see
        # vllm_omni/engine/orchestrator.py:65,673), so passing ``None``
        # raises ``AttributeError: 'NoneType' object has no attribute 'clone'``.
        stage1_sampling = SamplingParams(
            temperature=0.0,
            max_tokens=65536,
            detokenize=True,
        )

        # Hard-force repetition_penalty=1.0 to skip ``apply_penalties``,
        # which scatter-adds prompt ids into a codec-vocab-sized bucket
        # and triggers a CUDA OOB on Qwen3-TTS (logits are codec-vocab
        # ~3072, but prompt placeholder ids are 0 which is fine, while
        # any non-codec token would explode). The repetition penalty is
        # text-LM-specific and not meaningful for codec sampling.
        base_seed = sampling_params.get("seed")
        if base_seed is None:
            # Use a stable per-request seed when none is provided.
            base_seed = int(uuid4().int % (1 << 31))

        async def _run_sample(sample_index: int) -> dict[str, Any]:
            stage0 = SamplingParams(
                n=1,
                temperature=sampling_params.get("temperature", 0.9),
                top_p=sampling_params.get("top_p", 1.0),
                top_k=sampling_params.get("top_k", 50),
                max_tokens=max_tokens,
                seed=int(base_seed) + sample_index,
                logprobs=int(sampling_params.get("logprobs", 1)),
                repetition_penalty=1.0,
                stop_token_ids=sampling_params.get("stop_token_ids", [2150]),
            )
            sub_id = f"{request_id}-s{sample_index}"
            return await self._generate_one_tts_sample(
                sub_request_id=sub_id,
                custom_prompt=custom_prompt,
                sampling_params_list=[stage0, stage1_sampling],
                sample_index=sample_index,
            )

        # Sequential, not ``asyncio.gather`` — vllm-omni's AsyncOmni.generate
        # appears to share orchestrator state across concurrent in-flight
        # requests in a way that corrupts the talker's multi-codebook
        # codec emission (the code_predictor for quantizers 1..15 races
        # the codec_head's quantizer 0 emission and code2wav sees ``input_ids
        # length 1 not divisible by num_quantizers 16``, producing empty
        # audio for ~98 % of completions). Running n sequential generate()
        # calls is ~n× slower in wall time per row but yields correct
        # audio. The per-row latency is recovered because verl's
        # AgentLoopManager already issues the per-row agent loops in
        # parallel across N=batch_size replicas, so n=4 sub-requests per
        # row × 4 rows ÷ 4 replicas = 4 in flight cluster-wide regardless.
        results: list[dict[str, Any]] = []
        for i in range(n):
            results.append(await _run_sample(i))

        completions: list[CompletionAudio] = []
        sample_rate = _DEFAULT_AUDIO_SAMPLE_RATE
        num_preempted: int | None = None
        final_stop_reason: str | None = None
        for r in results:
            if r["codec_tokens"]:
                completions.append(
                    CompletionAudio(
                        sample_index=int(r["sample_index"]),
                        codec_tokens=r["codec_tokens"],
                        logprobs=r["logprobs"],
                        waveform=r["waveform"],
                        finish_reason=r["finish_reason"],
                    )
                )
            sample_rate = int(r["sample_rate"]) or sample_rate
            if r["finish_reason"] and final_stop_reason is None:
                final_stop_reason = r["finish_reason"]
            if r["num_preempted"] is not None:
                num_preempted = (num_preempted or 0) + int(r["num_preempted"])

        if not completions:
            raise RuntimeError(
                f"generate_tts request {request_id} produced no stage-0 completions across "
                f"n={n} sub-requests. Check that the verl-omni stage_config override sets "
                "final_output:true on stage 0 and that the engine was launched with "
                "stage_configs_path."
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
