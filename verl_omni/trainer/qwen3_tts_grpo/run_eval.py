# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Held-out evaluation pipeline for the Qwen3-TTS GRPO recipe (AC-7).

Runs the AR-TTS rollout for each eval row against both the base checkpoint
and the RL checkpoint, transcribes each generated waveform through the
remote Qwen3-ASR endpoint, computes CER and duration ratios, and writes
``eval_results.json`` with the AC-7 schema.

The ``--mock-inference-json`` flag is an optional **test-only** path that
loads pre-rendered hypotheses + durations from a JSON file so the schema
and the train/eval disjointness guard can be exercised without a live
TTS engine (CI-friendly). The default code path runs the real inference.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from verl_omni.reward_loop.reward_manager.asr_error_rate import (
    AsrEndpointConfig,
    AsrErrorRateRewardManager,
)
from verl_omni.utils.reward_score.asr_error_rate import compute_cer

logger = logging.getLogger(__name__)


def _assert_disjoint(eval_path: Path, train_path: Path | None) -> None:
    if train_path is None or not str(train_path):
        return
    train_ids = set(pd.read_parquet(train_path)["target_utt_id"].astype(str))
    eval_ids = set(pd.read_parquet(eval_path)["target_utt_id"].astype(str))
    overlap = train_ids & eval_ids
    if overlap:
        raise RuntimeError(
            f"eval parquet target_utt_id overlaps train ({len(overlap)} ids). "
            "Refusing to score — pick disjoint splits per AC-7."
        )


async def _live_inference_one(
    asyncomni: Any,
    row: dict[str, Any],
) -> tuple[np.ndarray | None, int, float]:
    """Drive one Qwen3-TTS rollout via the vllm-omni async engine.

    Returns ``(waveform, sample_rate, generated_seconds)``. ``waveform`` is
    ``None`` when the engine emits no audio (e.g. stage-1 failed).
    """

    from vllm.sampling_params import SamplingParams
    from vllm_omni.inputs.data import OmniCustomPrompt

    extra_args = {
        "text": str(row["prompt_text"]),
        "ref_audio": row["ref_audio"],
        "ref_text": str(row["ref_text"]),
        "task_type": "Base",
    }
    prompt: OmniCustomPrompt = {"extra_args": extra_args}
    sampling = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=4096, logprobs=1)
    waveform: np.ndarray | None = None
    sample_rate: int = 24000

    async for omni_out in asyncomni.generate(
        prompt=prompt,
        request_id=str(row.get("target_utt_id", "eval")),
        sampling_params_list=[sampling, None],
    ):
        if getattr(omni_out, "stage_id", None) == 1:
            mm = omni_out.multimodal_output or {}
            audio = mm.get("audio") or mm.get("waveform")
            sr = mm.get("sample_rate")
            if sr is not None:
                sample_rate = int(sr)
            if audio is not None:
                waveform = np.asarray(audio[0] if isinstance(audio, list) else audio)
    duration = float(len(waveform)) / max(sample_rate, 1) if waveform is not None else 0.0
    return waveform, sample_rate, duration


async def _live_inference_split(
    rows: list[dict[str, Any]],
    *,
    ckpt_path: str,
    stage_configs_path: str,
    asyncomni_factory: Callable[..., Any] | None = None,
) -> dict[str, tuple[np.ndarray, int, float]]:
    """Spin up an AsyncOmni for ``ckpt_path`` and roll out every eval row."""

    if asyncomni_factory is None:
        from vllm_omni.entrypoints import AsyncOmni

        engine = AsyncOmni(
            model=ckpt_path,
            stage_configs_path=stage_configs_path,
            trust_remote_code=True,
            enforce_eager=True,
        )
    else:
        engine = asyncomni_factory(model=ckpt_path, stage_configs_path=stage_configs_path)

    try:
        outputs: dict[str, tuple[np.ndarray, int, float]] = {}
        for row in rows:
            waveform, sr, duration = await _live_inference_one(engine, row)
            target_id = str(row["target_utt_id"])
            outputs[target_id] = (waveform, sr, duration)
        return outputs
    finally:
        shutdown = getattr(engine, "shutdown", None) or getattr(engine, "close", None)
        if shutdown is not None:
            res = shutdown()
            if asyncio.iscoroutine(res):
                await res


async def _score_inference(
    asr: AsrErrorRateRewardManager,
    rows: list[dict[str, Any]],
    inference: dict[str, tuple[Any, int, float]],
    hypothesis_map: dict[str, str] | None = None,
) -> dict[str, float]:
    """Score per-row CER + duration_ratio.

    When ``hypothesis_map`` is provided (the test/mock path) it is used
    directly; otherwise each generated waveform is transcribed via the
    remote ASR endpoint.
    """

    cers: list[float] = []
    duration_ratios: list[float] = []
    for row in rows:
        target_id = str(row["target_utt_id"])
        waveform, sr, gen_duration = inference.get(target_id, (None, 24000, 0.0))
        if hypothesis_map is not None:
            transcript = str(hypothesis_map.get(target_id, ""))
        else:
            transcript = ""
            if waveform is not None:
                try:
                    transcript = await asr.transcribe(np.asarray(waveform), int(sr))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("ASR failure on %s: %s", target_id, exc)
                    transcript = ""
        cer = compute_cer(hypothesis=transcript, reference=str(row["prompt_text"]))
        cers.append(cer)
        target_duration = float(row.get("target_duration") or 1.0)
        if target_duration > 0:
            duration_ratios.append(gen_duration / target_duration)
    return {
        "cer": float(sum(cers) / max(len(cers), 1)),
        "median_duration_ratio": float(statistics.median(duration_ratios)) if duration_ratios else 0.0,
        "mean_duration_ratio": float(sum(duration_ratios) / max(len(duration_ratios), 1)),
    }


def _mock_inference(
    mock_split: dict[str, Any], rows: Iterable[dict[str, Any]]
) -> dict[str, tuple[np.ndarray | None, int, float]]:
    hypotheses = mock_split.get("hypotheses", {})
    durations = mock_split.get("durations", {})
    sample_rate = int(mock_split.get("sample_rate", 24000))
    out: dict[str, tuple[np.ndarray | None, int, float]] = {}
    for row in rows:
        target_id = str(row["target_utt_id"])
        if target_id in hypotheses:
            duration = float(durations.get(target_id, 0.0))
            # Render a placeholder waveform whose length matches the duration so
            # the duration ratio computation runs against an array, not a stub.
            num_samples = max(int(duration * sample_rate), 1)
            waveform = np.zeros(num_samples, dtype=np.float32)
            out[target_id] = (waveform, sample_rate, duration)
        else:
            out[target_id] = (None, sample_rate, 0.0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-ckpt", required=True)
    parser.add_argument("--rl-ckpt", required=True)
    parser.add_argument("--eval-parquet", required=True)
    parser.add_argument("--asr-base-url", required=True)
    parser.add_argument("--output", default="eval_results.json")
    parser.add_argument("--train-parquet", default=None, help="Optional, enables target_utt_id disjointness check.")
    parser.add_argument(
        "--mock-inference-json",
        default=None,
        help=(
            "TEST-ONLY: path to a JSON file of the form "
            "{'base': {'hypotheses': {utt_id: transcript}, 'durations': {utt_id: seconds}}, 'rl': {...}}. "
            "Used to exercise the AC-7 JSON schema and disjointness guard without a live TTS engine. "
            "Production runs omit this flag; the script then drives vllm-omni Qwen3-TTS for each row."
        ),
    )
    parser.add_argument(
        "--stage-configs-path",
        default=str(
            Path(__file__).resolve().parents[2]
            / "pipelines"
            / "qwen3_tts_grpo"
            / "stage_configs"
            / "qwen3_tts.yaml"
        ),
        help="Path to the verl-omni Qwen3-TTS stage config (the one that enables stage-0 emission).",
    )
    args = parser.parse_args()

    eval_path = Path(args.eval_parquet)
    _assert_disjoint(eval_path, Path(args.train_parquet) if args.train_parquet else None)

    rows = pd.read_parquet(eval_path).to_dict(orient="records")
    asr = AsrErrorRateRewardManager(config={}, endpoint=AsrEndpointConfig(base_url=args.asr_base_url))

    async def _run() -> dict[str, float]:
        if args.mock_inference_json:
            mock = json.loads(Path(args.mock_inference_json).read_text())
            base_inference = _mock_inference(mock.get("base", {}), rows)
            rl_inference = _mock_inference(mock.get("rl", {}), rows)
            base_hypotheses = mock.get("base", {}).get("hypotheses", {})
            rl_hypotheses = mock.get("rl", {}).get("hypotheses", {})
        else:
            base_inference = await _live_inference_split(
                rows,
                ckpt_path=args.base_ckpt,
                stage_configs_path=args.stage_configs_path,
            )
            rl_inference = await _live_inference_split(
                rows,
                ckpt_path=args.rl_ckpt,
                stage_configs_path=args.stage_configs_path,
            )
            base_hypotheses = None
            rl_hypotheses = None
        base_metrics = await _score_inference(asr, rows, base_inference, base_hypotheses)
        rl_metrics = await _score_inference(asr, rows, rl_inference, rl_hypotheses)
        return {
            "base_cer": base_metrics["cer"],
            "rl_cer": rl_metrics["cer"],
            "base_median_duration_ratio": base_metrics["median_duration_ratio"],
            "rl_median_duration_ratio": rl_metrics["median_duration_ratio"],
            "base_mean_duration_ratio": base_metrics["mean_duration_ratio"],
            "rl_mean_duration_ratio": rl_metrics["mean_duration_ratio"],
        }

    results = asyncio.run(_run())
    Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
