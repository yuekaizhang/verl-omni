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
"""LLMServerClient subclass exposing ``generate_tts`` for AR-TTS rollouts.

Upstream :class:`verl.workers.rollout.llm_server.LLMServerClient` only has
``generate(...)``, which dispatches to ``server.generate.remote(...)`` on
the picked replica. The AR-TTS rollout server defines a sibling
``generate_tts`` method; this client routes the same load-balancing dance
to that method.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from verl.workers.rollout.llm_server import LLMServerClient


class AutoRegressiveTTSServerClient(LLMServerClient):
    """Client that load-balances ``generate_tts`` across replicas.

    Wire this in by setting
    ``actor_rollout_ref.rollout.llm_server_client_class`` to its FQN, or by
    instantiating it directly inside the AR-TTS agent-loop worker.
    """

    async def generate_tts(
        self,
        *,
        request_id: str,
        prompt_text: str,
        ref_audio: Any,
        ref_text: str,
        sampling_params: dict[str, Any],
        n: int,
        task_type: str = "Base",
        language: str | None = None,
    ) -> Any:
        """Acquire a server, call ``generate_tts`` on it, release the server.

        Returns the :class:`AudioRolloutOutput` produced by the server.
        """

        server_id, server = await self._acquire_server(request_id)
        try:
            return await server.generate_tts.remote(
                prompt_text=prompt_text,
                ref_audio=ref_audio,
                ref_text=ref_text,
                sampling_params=sampling_params,
                request_id=uuid4().hex,
                n=n,
                task_type=task_type,
                language=language,
            )
        finally:
            self._release_server(server_id)
