# pattern: Functional Core
"""Value objects shared across the functional core.

Immutable (frozen) so they have value-equality and are safe to pass around the
pure reconciliation and rendering code without aliasing surprises.

Times are seconds from the start of the recording (floats), matching what both
pyannote diarisation and NeMo ASR emit.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Word:
    """A single transcribed token with its timing, as emitted by the ASR stage."""

    text: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class SpeakerTurn:
    """A span during which one speaker is active, from the diarisation stage."""

    speaker: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class AttributedSegment:
    """A run of consecutive words attributed to one speaker (the reconciled output)."""

    speaker: str
    start: float
    end: float
    text: str
