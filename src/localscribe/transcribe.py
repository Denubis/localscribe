# pattern: Imperative Shell
"""Transcription stage: NVIDIA Parakeet-TDT-0.6b-v3 via NeMo.

Emits native word-level timestamps in a single small model — the timestamp
precision is what lets reconciliation land speaker boundaries correctly.

Long audio is transcribed in VRAM-safe windows and stitched back onto one
timeline (guardrail #1): a single pass over an 80-minute file exhausted the
reference 24 GB GPU, so we window the audio, transcribe each window, free CUDA
between windows, and shift
each window's timestamps back by its offset.
"""

from __future__ import annotations

import logging
import shutil
from importlib import import_module
from pathlib import Path
from typing import Any

from .audio import split_into_windows
from .errors import TranscriptionError
from .models import Word
from .parse import parse_words, stitch_windows
from .runtime import free_cuda

logger = logging.getLogger(__name__)

MODEL = "nvidia/parakeet-tdt-0.6b-v3"

#: Local relative-position attention with a bounded context window. Recommended
#: for audio beyond ~24 minutes.
_LOCAL_ATTENTION = ("rel_pos_local_attn", [256, 256])

#: Per-window length. The 25-minute em session fit comfortably under 24 GB; this
#: leaves generous headroom and keeps any one window well clear of OOM.
WINDOW_SECONDS = 1200


def transcribe(recording: Path) -> list[Word]:
    """Return time-stamped words for ``recording`` (mono 16 kHz)."""
    try:
        nemo_asr = import_module("nemo.collections.asr")
        open_dict = import_module("omegaconf").open_dict
    except Exception as exc:  # noqa: BLE001 - boundary: wrap import failure
        raise TranscriptionError(f"NeMo ASR is unavailable: {exc}") from exc

    try:
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL)
        model.change_attention_model(*_LOCAL_ATTENTION)
        # Reusing the TDT CUDA graph across long windows caused an illegal-memory
        # abort on the RTX 5080. Keep eager decoding in the persistent config:
        # timestamps=True can rebuild the decoder, undoing a runtime-only disable.
        with open_dict(model.cfg.decoding.greedy):
            model.cfg.decoding.greedy.use_cuda_graph_decoder = False
        model.change_decoding_strategy(model.cfg.decoding, verbose=False)
    except Exception as exc:  # noqa: BLE001 - boundary: wrap load failure
        raise TranscriptionError(f"failed to load transcription model {MODEL}: {exc}") from exc

    windows = split_into_windows(recording, WINDOW_SECONDS)
    chunk_dir = windows[0][1].parent if windows else None
    try:
        per_window = [(offset, _transcribe_one(model, chunk)) for offset, chunk in windows]
    finally:
        if chunk_dir is not None:
            shutil.rmtree(chunk_dir, ignore_errors=True)

    return stitch_windows(per_window)


def _transcribe_one(model: Any, chunk: Path) -> list[Word]:
    """Transcribe one window, returning words with chunk-relative timestamps."""
    try:
        output = model.transcribe([str(chunk)], timestamps=True)
    except Exception as exc:  # noqa: BLE001 - boundary: wrap inference failure
        raise TranscriptionError(f"transcription failed on {chunk}: {exc}") from exc
    finally:
        free_cuda()

    if not output:
        logger.warning("transcription returned no hypotheses for %s", chunk)
        return []
    return parse_words(_word_stamps(output[0]))


def _word_stamps(hypothesis: object) -> list[dict[str, object]]:
    """Pull the word-timestamp list off a NeMo Hypothesis.

    NeMo has used both ``.timestamp`` and ``.timestep`` for the dict across
    versions; accept either and return an empty list if neither is present.
    """
    timing = getattr(hypothesis, "timestamp", None)
    if timing is None:
        timing = getattr(hypothesis, "timestep", None)
    if isinstance(timing, dict):
        return timing.get("word", [])
    return []
