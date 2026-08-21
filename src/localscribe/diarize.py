# pattern: Imperative Shell
"""Diarisation stage: pyannote speaker-diarization-community-1.

Loads the gated pipeline (weights already cached, token read from the HF cache),
runs it on the recording, and extracts exclusive speaker turns — the "one speaker
active at a time" view that makes word-to-speaker attribution clean. The plain
``(start, end, speaker)`` spans are handed to the tested parser; no business
logic lives here.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import Any

from .errors import DiarizationError
from .models import SpeakerTurn
from .parse import parse_turns

logger = logging.getLogger(__name__)

MODEL = "pyannote/speaker-diarization-community-1"


def diarize(
    recording: Path,
    *,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    progress: bool | None = None,
) -> list[SpeakerTurn]:
    """Return exclusive speaker turns for ``recording``.

    This is the slow stage on a long recording and it used to run silently.
    ``progress`` shows pyannote's own per-step bar; the default follows whether
    stdout is a terminal, so piping into a log stays clean.
    """
    torch = import_module("torch")
    Pipeline = import_module("pyannote.audio").Pipeline

    try:
        pipeline = Pipeline.from_pretrained(MODEL, token=True)
    except Exception as exc:  # noqa: BLE001 - boundary: wrap any load failure
        raise DiarizationError(f"failed to load diarisation model {MODEL}: {exc}") from exc
    if pipeline is None:
        raise DiarizationError(f"diarisation model {MODEL} failed to load (returned None)")

    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    show = sys.stdout.isatty() if progress is None else progress
    try:
        # None means "unconstrained" to pyannote, so the bounds pass through as-is.
        with _progress_hook(show) as hook:
            output = pipeline(
                recording, min_speakers=min_speakers, max_speakers=max_speakers, hook=hook
            )
    except Exception as exc:  # noqa: BLE001 - boundary: wrap any inference failure
        raise DiarizationError(f"diarisation failed on {recording}: {exc}") from exc

    return parse_turns(_spans(output.exclusive_speaker_diarization))


@contextmanager
def _progress_hook(show: bool) -> Iterator[Any]:
    """Yield pyannote's ProgressHook, or None when progress is off or unavailable.

    Best-effort by design (guardrail 2): a missing or changed hook API costs a
    progress bar, never a transcript, so an import failure here yields None and
    diarisation runs exactly as it did before.
    """
    if not show:
        yield None
        return
    try:
        ProgressHook = import_module(
            "pyannote.audio.pipelines.utils.hook"
        ).ProgressHook
    except ImportError:
        logger.debug("pyannote ProgressHook unavailable; diarising without progress")
        yield None
        return
    with ProgressHook() as hook:
        yield hook


def _spans(diarization: Any) -> Iterator[tuple[float, float, str]]:
    """Yield ``(start, end, speaker)`` from a pyannote diarisation result.

    pyannote 4.0 iterates as ``(segment, speaker)`` pairs; fall back to the older
    ``itertracks`` shape if that is what this object provides.
    """
    try:
        pairs = [(turn.start, turn.end, speaker) for turn, speaker in diarization]
    except (TypeError, ValueError):
        pairs = [
            (turn.start, turn.end, speaker)
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
    yield from pairs
