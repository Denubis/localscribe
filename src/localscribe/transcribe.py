# pattern: Imperative Shell
"""Transcription stage: NVIDIA Parakeet-TDT-0.6b-v3 via NeMo.

Emits native word-level timestamps in a single small model — the timestamp
precision is what lets reconciliation land speaker boundaries correctly.

Long audio is transcribed in windows and stitched back onto one timeline
(guardrail #1): a single pass over an 80-minute file exhausted the reference
24 GB GPU. Windows are cut at pauses so no word is split.

The model also skips speech now and then, silently. Measured 2026-10-06/07 on
fifteen meetings: 10-to-20-minute windows skipped quiet speakers (stretches about 8 dB
below their window's loud level, up to half a minute long), and 30-second windows
skipped different, normal-volume stretches mid-window, while choosing words a
little better in long windows. Neither length alone was clean, so we run both:
the long pass is the transcript, the short pass fills any hole in it, and any
stretch still silent while the audio is at speaking level is transcribed once
more on its own (guardrail #2: a skip must not silently lose speech).

Finally each word's end is pulled back to where the sound stops, so the pauses
between words are visible to the paragraph breaks downstream.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

from .audio import (
    LEVEL_FRAME_SECONDS,
    choose_cut_points,
    extract_span,
    find_holes,
    measure_levels,
    split_into_windows,
)
from .errors import TranscriptionError
from .models import Word
from .parse import parse_words, stitch_windows
from .runtime import free_cuda
from .timing import fill_gaps, trim_word_ends

logger = logging.getLogger(__name__)

MODEL = "nvidia/parakeet-tdt-0.6b-v3"

#: The two passes: window length (seconds) and how far back from its limit to
#: look for the quietest point to cut at. The long pass is the primary
#: transcript; the short pass witnesses its holes. Ten minutes is the longest
#: window the model's own attention handles within the reference 24 GB
#: (attention memory grows with the square of the window; 20 minutes wanted
#: about 31 GB), and it chose words as well as 20-minute windows under the
#: bounded local attention NVIDIA suggests for long audio.
LONG_PASS = (600.0, 60.0)
SHORT_PASS = (30.0, 10.0)

#: A stretch with no words is a hole when it is at least this long. It is filled
#: from the other pass, then, if still at speaking level for at least this share
#: of its frames, transcribed once more on its own.
HOLE_SECONDS = 1.0
HOLE_SPEECH_FRACTION = 0.4

#: Audio taken either side of a hole so the model hears the words at its edges whole.
HOLE_PADDING_SECONDS = 0.5

#: Shortest silence after a word that its stamped end is pulled back over.
MIN_PAUSE_SECONDS = 0.1


def transcribe(recording: Path) -> list[Word]:
    """Return time-stamped words for ``recording`` (mono 16 kHz)."""
    try:
        nemo_asr = import_module("nemo.collections.asr")
    except Exception as exc:  # noqa: BLE001 - boundary: wrap import failure
        raise TranscriptionError(f"NeMo ASR is unavailable: {exc}") from exc

    try:
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL)
    except Exception as exc:  # noqa: BLE001 - boundary: wrap load failure
        raise TranscriptionError(f"failed to load transcription model {MODEL}: {exc}") from exc

    levels = measure_levels(recording)
    primary = _transcribe_windows(model, recording, levels, LONG_PASS)
    witness = _transcribe_windows(model, recording, levels, SHORT_PASS)
    words = fill_gaps(primary, witness, min_seconds=HOLE_SECONDS)
    logger.info("short pass filled %d words into holes of the long pass", len(words) - len(primary))

    holes = find_holes(
        levels,
        [(word.start, word.end) for word in words],
        frame_seconds=LEVEL_FRAME_SECONDS,
        min_seconds=HOLE_SECONDS,
        speech_fraction=HOLE_SPEECH_FRACTION,
    )
    if holes:
        logger.info("retrying %d untranscribed stretches at speaking level", len(holes))

    def transcribe_span(start: float, end: float) -> list[Word]:
        clip = extract_span(recording, start, end)
        try:
            return _transcribe_one(model, clip)
        finally:
            clip.unlink(missing_ok=True)

    words = fill_holes(words, holes, padding=HOLE_PADDING_SECONDS, transcribe_span=transcribe_span)
    return trim_word_ends(
        words, levels, frame_seconds=LEVEL_FRAME_SECONDS, min_pause_seconds=MIN_PAUSE_SECONDS
    )


def _transcribe_windows(
    model: Any,
    recording: Path,
    levels: Sequence[float],
    window: tuple[float, float],
) -> list[Word]:
    """One pass over the whole recording in pause-cut windows of the given length."""
    target_seconds, search_seconds = window
    cuts = choose_cut_points(
        levels,
        frame_seconds=LEVEL_FRAME_SECONDS,
        target_seconds=target_seconds,
        search_seconds=search_seconds,
    )
    windows = split_into_windows(recording, cuts)
    try:
        per_window = [(offset, _transcribe_one(model, chunk)) for offset, chunk in windows]
    finally:
        shutil.rmtree(windows[0][1].parent, ignore_errors=True)
    return stitch_windows(per_window)


def fill_holes(
    words: Sequence[Word],
    holes: Sequence[tuple[float, float]],
    *,
    padding: float,
    transcribe_span: Callable[[float, float], list[Word]],
) -> list[Word]:
    """Transcribe each hole (padded) and merge the words that fall inside it.

    ``transcribe_span(start, end)`` returns words timed from ``start``. Words
    centred outside the hole belong to the padding, where the neighbours were
    already transcribed, and are dropped.
    """
    merged = list(words)
    for hole_start, hole_end in holes:
        clip_start = max(0.0, hole_start - padding)
        for word in transcribe_span(clip_start, hole_end + padding):
            start, end = word.start + clip_start, word.end + clip_start
            if hole_start <= (start + end) / 2 <= hole_end:
                merged.append(Word(text=word.text, start=start, end=end))
    merged.sort(key=lambda word: word.start)
    return merged


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
