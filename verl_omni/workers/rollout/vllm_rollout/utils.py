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
import os

import torch
from verl.workers.rollout.vllm_rollout.utils import VLLM_LORA_INT_ID, VLLM_LORA_NAME, VLLM_LORA_PATH, set_death_signal
from vllm_omni.diffusion.worker.diffusion_worker import CustomPipelineWorkerExtension

from verl_omni.utils.vllm_omni import OmniTensorLoRARequest, VLLMOmniHijack
from verl_omni.workers.rollout.vllm_rollout.npu_utils import NPUColocateWorkerMixin

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class vLLMOmniColocateWorkerExtension(NPUColocateWorkerMixin, CustomPipelineWorkerExtension):
    """
    The class for vLLM-Omni's worker to inherit from, in the colocate setting.
    By defining an extension class, the code can work no matter what is
    the underlying worker class. This way, the code can be compatible
    with both vLLM V0 and V1.
    NOTE: we define this class in a separate module, and the main module
    should pass the full qualified name as `worker_extension_cls` argument.

    Feature support:
    1. LoRA
    2. NPU (Ascend) memory-pool, sleep, and wake_up — via NPUColocateWorkerMixin
    """

    def __new__(cls, **kwargs):
        set_death_signal()

        # 1. patch for Lora
        VLLMOmniHijack.hijack()

        return super().__new__(cls)

    def update_weights_from_ipc(self, peft_config: dict = None, base_sync_done=False, use_shm: bool = False):
        """Update the weights of the rollout model."""

        from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightReceiver

        # In async mode, make sure the old lora is removed before adding the new one
        if peft_config and base_sync_done:
            self.remove_lora(VLLM_LORA_INT_ID)

        assert self.device is not None
        receiver = BucketedWeightReceiver(
            zmq_handle=self._get_zmq_handle(),
            device=self.device,
            use_shm=use_shm,
        )
        receiver.receive_weights(
            on_bucket_received=lambda weights: self._update_weights(
                weights, peft_config=peft_config, base_sync_done=base_sync_done
            )
        )

    def _update_weights(self, weights: list[tuple[str, torch.Tensor]], peft_config: dict, base_sync_done: bool):
        # Diagnostic toggle (BL-20260518-fsdp-vllm-sync-suspect): standalone
        # vllm-omni with ``.hf_cache`` weights synthesizes correct Chinese
        # audio (CER=0.0000); production verl-omni with FSDP→vLLM-synced
        # weights produces all-NaN rewards. To isolate whether the FSDP
        # sync itself corrupts the codec emission, set
        # ``QWEN3_TTS_SKIP_FSDP_VLLM_SYNC=1`` to make this method a no-op
        # so the rollout permanently uses the ``.hf_cache`` weights. The
        # actor still trains, but the rollout is effectively frozen —
        # only useful as a diagnostic to bisect the failure source.
        if os.environ.get("QWEN3_TTS_SKIP_FSDP_VLLM_SYNC", "0") == "1":
            n_pushed = sum(1 for _ in weights) if isinstance(weights, list) else 0
            logger.info(
                "Qwen3-TTS FSDP→vLLM sync SKIPPED via env "
                "QWEN3_TTS_SKIP_FSDP_VLLM_SYNC=1 (would have pushed %d weights)",
                n_pushed,
            )
            return
        if peft_config and base_sync_done:
            weights = dict(weights)
            lora_request = OmniTensorLoRARequest(
                lora_name=VLLM_LORA_NAME,
                lora_int_id=VLLM_LORA_INT_ID,
                lora_path=VLLM_LORA_PATH,
                peft_config=peft_config,
                lora_tensors=weights,
            )
            self.add_lora(lora_request)
            logger.info(f"vLLM-Omni load weights, loaded_params: {len(weights)}")
        else:
            # FSDP→vLLM Qwen3-TTS weight forwarding.
            #
            # Training side is ``Qwen3TTSForConditionalGeneration`` whose
            # root has two children: ``talker.*`` and ``speaker_encoder.*``.
            # vLLM-omni stage 0 loads ``Qwen3TTSTalkerForConditionalGeneration``
            # — its root *is* the talker — and ships its OWN
            # ``hf_to_vllm_mapper`` (a transformers ``WeightsMapper``) that
            # rewrites ``talker.model.layers.`` → ``model.layers.``,
            # ``talker.codec_head.`` → ``lm_head.``, ``speaker_encoder.``
            # → ``speaker_encoder.`` etc. (see
            # ``vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py``).
            # Pre-stripping ``talker.`` here would break that mapper and
            # leave the actual model running on dummy / unloaded weights,
            # which surfaces downstream as a CUDA scatter-gather OOB
            # assert on the first ``generate_tts`` call.
            #
            # Forward weights unchanged and let
            # ``reload_weights(weights_iterator=..., is_checkpoint_format=True)``
            # invoke the model's ``load_weights`` → ``AutoWeightsLoader``
            # → ``hf_to_vllm_mapper`` pipeline.
            #
            # Speaker-encoder caveat: vllm-omni's
            # ``Qwen3TTSTalkerForConditionalGeneration.load_weights``
            # lazily builds ``self.speaker_encoder`` on the FIRST call
            # that sees ``speaker_encoder.*`` weights, then loads them.
            # On subsequent calls the AutoWeightsLoader can't navigate
            # the existing speaker_encoder submodules (raises
            # ``ValueError: There is no module or parameter named
            # 'speaker_encoder.blocks.0.conv.weight' ... available
            # parameters belonging to ... (Conv1d) are: set()``). The
            # speaker_encoder takes a *reference audio* and produces a
            # speaker embedding — it's read-only relative to the policy
            # being trained, so we can safely skip its weights on every
            # call after the first.
            # Always drop ``speaker_encoder.*`` from FSDP→vLLM syncs.
            #
            # Two reasons together force this:
            #
            # 1. With ``rollout.load_format=auto`` (set in
            #    qwen3_tts_rollout.yaml so the codec_embedding /
            #    code_predictor heads receive real values at engine
            #    init, not random ``dummy`` tensors), vllm-omni's
            #    ``Qwen3TTSTalkerForConditionalGeneration.load_weights``
            #    already lazy-builds and loads ``self.speaker_encoder``
            #    during the safetensors checkpoint load. The talker's
            #    AutoWeightsLoader can't reload weights into the
            #    already-built submodule on subsequent syncs (raises
            #    ``ValueError: There is no module or parameter named
            #    'speaker_encoder.blocks.0.conv.weight' ... available
            #    parameters belonging to ... (Conv1d) are: set()``).
            #
            # 2. The speaker_encoder takes a *reference audio* and
            #    produces a speaker embedding — it is read-only
            #    relative to the policy being trained, so we don't
            #    actually need to refresh its weights from the actor.
            #
            # When ``load_format=dummy`` was the default and the
            # initial load skipped speaker_encoder.*, we used to push
            # them on the FIRST sync (lazy-built then) and skip
            # afterwards. With ``load_format=auto`` that first push
            # also fails because the submodule is already built. The
            # simpler invariant is: always skip.
            stage0_weights = [
                (n, t) for (n, t) in weights
                if not n.startswith("speaker_encoder.")
            ]
            logger.info(
                "Qwen3-TTS FSDP→vLLM sync: forwarding %d weights "
                "(speaker_encoder always skipped — preloaded via "
                "load_format=auto)",
                len(stage0_weights),
            )

            # vllm-omni 0.18 renamed worker.load_weights -> worker.reload_weights
            # (which forwards to ``model_runner.reload_weights``). FSDP-side
            # weights arrive in *original* (unfused) Qwen3-TTS layout
            # (separate q_proj/k_proj/v_proj, gate_proj/up_proj). vLLM-omni's
            # Qwen3-TTS model fuses these into qkv_proj / gate_up_proj for
            # kernel performance. Pass ``is_checkpoint_format=True`` so the
            # model_runner runs vLLM's checkpoint loader, which knows about
            # the fusion mapping (see model load_weights in
            # vllm_omni/model_executor/models/qwen3_tts/).
            if hasattr(self, "reload_weights"):
                self.reload_weights(
                    weights_iterator=iter(stage0_weights),
                    is_checkpoint_format=True,
                )
            else:
                # Older vllm-omni still exposes ``load_weights``.
                self.load_weights(stage0_weights)

    def _get_zmq_handle(self) -> str:
        """Get ZMQ handle for communication.
        Uses Ray job id + replica_rank + local_rank to form the handle so it
        matches the sender side regardless of CUDA_VISIBLE_DEVICES differences,
        avoids collisions when multiple replicas share the same node, and is
        unique per Ray job to avoid cross-job collisions on shared hosts. The
        job id is forwarded by the vLLMHttpServer actor as VERL_RAY_JOB_ID and
        inherited by this vLLM worker subprocess.
        """
        replica_rank = os.environ.get("VERL_REPLICA_RANK", "0")
        job_id = os.environ.get("VERL_RAY_JOB_ID", "0")
        return f"ipc:///tmp/rl-colocate-zmq-{job_id}-replica-{replica_rank}-rank-{self.local_rank}.sock"
