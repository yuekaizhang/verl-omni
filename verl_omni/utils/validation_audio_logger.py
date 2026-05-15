# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Validation audio artifact logger for the Qwen3-TTS GRPO recipe (AC-8).

Per AC-8 each validation step writes:

- at least 4 randomly selected ``generated.wav`` files,
- the matching ``ref_audio.wav`` files,
- and (when ``target_audio`` is present in the dataset) the matching
  ``target_audio.wav`` files,

plus a ``metrics.json`` with the scalar fields. Disk-write failures
surface as a typed :class:`ArtifactWriteError`; a post-run check flags
any validation step that emitted zero artifacts.

When wandb is enabled the same artifacts are also logged as
``wandb.Audio`` objects through the same code path; the on-disk files
remain authoritative.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import numpy as np

logger = logging.getLogger(__name__)


class ArtifactWriteError(RuntimeError):
    """Raised when validation artifact writing fails (disk full, permission, etc.)."""


def _maybe_write_wav(path: Path, waveform: Any, sample_rate: int) -> bool:
    """Write a waveform to ``path`` as 16-bit PCM WAV. Returns True on success."""

    if waveform is None:
        return False
    import soundfile as sf

    arr = np.asarray(waveform)
    if arr.dtype.kind == "O":
        try:
            arr = np.asarray(arr.item() if arr.shape == () else arr[0])
        except Exception:
            return False
    if arr.size == 0:
        return False
    if arr.dtype not in (np.float32, np.float64, np.int16, np.int32):
        arr = arr.astype(np.float32)
    try:
        sf.write(str(path), arr, int(sample_rate), format="WAV", subtype="PCM_16")
    except Exception as exc:
        raise ArtifactWriteError(f"Failed to write {path}: {exc}") from exc
    return True


def _maybe_copy_audio(src: Any, dst: Path) -> bool:
    """Copy a referenced audio path (or skip if the value is an ndarray/None)."""

    if not isinstance(src, (str, Path)):
        return False
    src_path = Path(str(src))
    if not src_path.exists():
        return False
    try:
        shutil.copyfile(src_path, dst)
    except OSError as exc:
        raise ArtifactWriteError(f"Failed to copy {src_path} -> {dst}: {exc}") from exc
    return True


def log_validation_step(
    *,
    out_dir: Path,
    step: int,
    samples: list[dict[str, Any]],
    scalar_metrics: dict[str, float],
    num_samples: int = 4,
    seed: int | None = None,
    wandb_run: Any = None,
) -> int:
    """Persist validation artifacts for one step.

    Each entry of ``samples`` must contain:

    - ``waveform``: synthesized audio (ndarray or path).
    - ``sample_rate``: int.
    - ``ref_audio``: optional reference audio (path or ndarray).
    - ``target_audio``: optional ground-truth audio (path).

    Returns the number of generated audio files actually written.
    """

    if not samples:
        raise ArtifactWriteError(
            f"Validation step {step} received zero samples — refusing to write empty artifact dir."
        )
    step_dir = out_dir / f"validation_step_{step:06d}"
    try:
        step_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ArtifactWriteError(f"Cannot create artifact dir {step_dir}: {exc}") from exc

    rng = random.Random(seed if seed is not None else step)
    picks = rng.sample(samples, k=min(num_samples, len(samples)))

    generated_written = 0
    for i, sample in enumerate(picks):
        sr = int(sample.get("sample_rate", 24000))
        if _maybe_write_wav(step_dir / f"generated_{i:02d}.wav", sample.get("waveform"), sr):
            generated_written += 1
            if wandb_run is not None:
                _maybe_log_wandb_audio(wandb_run, step_dir / f"generated_{i:02d}.wav", step=step, kind="generated")
        ref = sample.get("ref_audio")
        if isinstance(ref, (str, Path)):
            _maybe_copy_audio(ref, step_dir / f"ref_audio_{i:02d}.wav")
        else:
            _maybe_write_wav(step_dir / f"ref_audio_{i:02d}.wav", ref, sr)
        target = sample.get("target_audio")
        if isinstance(target, (str, Path)):
            _maybe_copy_audio(target, step_dir / f"target_audio_{i:02d}.wav")

    metrics_path = step_dir / "metrics.json"
    try:
        metrics_path.write_text(json.dumps(scalar_metrics, indent=2, ensure_ascii=False))
    except OSError as exc:
        raise ArtifactWriteError(f"Failed to write {metrics_path}: {exc}") from exc

    if generated_written == 0:
        raise ArtifactWriteError(
            f"Validation step {step}: no generated waveforms were writable (saw "
            f"{len(picks)} candidates). Refusing to silently skip per AC-8."
        )
    return generated_written


def post_run_check_emitted_artifacts(run_output_dir: Path) -> list[Path]:
    """Return the list of validation-step dirs that emitted at least one generated wav.

    Raises :class:`ArtifactWriteError` if any ``validation_step_*`` directory
    exists but contains zero generated audio files (AC-8 negative test).
    """

    if not run_output_dir.exists():
        return []
    bad: list[Path] = []
    good: list[Path] = []
    for step_dir in sorted(run_output_dir.glob("validation_step_*")):
        if not step_dir.is_dir():
            continue
        wavs = list(step_dir.glob("generated_*.wav"))
        if wavs:
            good.append(step_dir)
        else:
            bad.append(step_dir)
    if bad:
        raise ArtifactWriteError(
            f"Validation step(s) emitted zero audio artifacts: "
            f"{', '.join(str(p.name) for p in bad)}. AC-8 requires at least one "
            "generated waveform per validation step."
        )
    return good


def _maybe_log_wandb_audio(wandb_run: Any, path: Path, *, step: int, kind: str) -> None:
    try:
        import wandb

        wandb_run.log({f"validation/{kind}/{path.stem}": wandb.Audio(str(path))}, step=step)
    except Exception as exc:
        logger.warning("wandb audio logging failed for %s: %s", path, exc)


__all__ = [
    "ArtifactWriteError",
    "log_validation_step",
    "post_run_check_emitted_artifacts",
]
