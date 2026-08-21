# pattern: Functional Core
"""Parse raw model output into core value objects, tolerating malformed entries.

This is the ingestion boundary for guardrail #2: the ASR and diarisation stages
hand over plain Python structures (dicts, tuples), and these functions validate
each entry, dropping and logging anything malformed rather than crashing. One bad
word stamp must never cost the whole transcript.

Pure: the shell does the model I/O and extracts the plain structures; these
functions only transform and validate.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from typing import Any

from .models import SpeakerTurn, Word

logger = logging.getLogger(__name__)


def parse_words(stamps: Iterable[dict[str, Any]]) -> list[Word]:
    """Convert NeMo word-timestamp dicts to ``Word`` objects.

    Each stamp is expected to carry the token text under ``segment`` (or ``word``
    as a fallback across NeMo versions) plus numeric ``start`` and ``end`` in
    seconds. Malformed stamps are skipped.
    """
    words: list[Word] = []
    for stamp in stamps:
        text = stamp.get("segment") or stamp.get("word")
        start = _as_finite_float(stamp.get("start"))
        end = _as_finite_float(stamp.get("end"))
        if text is None or start is None or end is None:
            logger.warning("dropping malformed word stamp: %r", stamp)
            continue
        words.append(Word(text=str(text), start=start, end=end))
    return words


def parse_turns(spans: Iterable[tuple[Any, Any, Any]]) -> list[SpeakerTurn]:
    """Convert ``(start, end, speaker)`` spans from diarisation to ``SpeakerTurn``.

    Spans with non-numeric times or a missing speaker label are skipped.
    """
    turns: list[SpeakerTurn] = []
    for span in spans:
        try:
            start_raw, end_raw, speaker = span
        except (TypeError, ValueError):
            logger.warning("dropping malformed speaker span: %r", span)
            continue
        start = _as_finite_float(start_raw)
        end = _as_finite_float(end_raw)
        if speaker is None or start is None or end is None:
            logger.warning("dropping malformed speaker span: %r", span)
            continue
        turns.append(SpeakerTurn(speaker=str(speaker), start=start, end=end))
    return turns


def stitch_windows(windows: list[tuple[float, list[Word]]]) -> list[Word]:
    """Reassemble windowed transcription onto one timeline.

    Long audio is transcribed in windows to stay under VRAM (guardrail #1). Each
    window's word timestamps are relative to its own start, so we shift them by
    the window's offset into the full recording and concatenate in order.
    """
    return [
        Word(text=word.text, start=word.start + offset, end=word.end + offset)
        for offset, words in windows
        for word in words
    ]


def _as_finite_float(value: object) -> float | None:
    """Coerce a real, finite number to float; return None for anything else.

    Excludes bool (an int subclass) and NaN/inf. Returning ``float | None`` lets
    callers narrow with a plain ``is None`` check.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None
