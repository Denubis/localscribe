# pattern: Functional Core
"""Match diarised speaker clusters to enrolled voices, and rename segments.

Pure: operates on plain float vectors (speaker embeddings) and value objects. The
shell extracts the embeddings from audio; this module only compares and relabels.

Enrolment never changes who-spoke-when — the diarisation already decided that. It
only attaches a human name to a cluster when a known voice matches closely enough.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence

from .models import AttributedSegment

logger = logging.getLogger(__name__)


def match_speakers(
    cluster_embeddings: Mapping[str, Sequence[float]],
    enrolled: Mapping[str, Sequence[float]],
    *,
    threshold: float,
) -> dict[str, str]:
    """Map each cluster to its best-matching enrolled name, if above ``threshold``.

    Clusters whose closest enrolled voice scores below the threshold are omitted,
    so they keep their anonymous ``SPEAKER_xx`` label downstream.
    """
    mapping: dict[str, str] = {}
    for speaker, vector in cluster_embeddings.items():
        best_name: str | None = None
        best_score = threshold
        for name, reference in enrolled.items():
            score = _cosine(vector, reference)
            if score >= best_score:
                best_score = score
                best_name = name
        if best_name is not None:
            mapping[speaker] = best_name
            logger.debug("matched %s -> %s (score %.3f)", speaker, best_name, best_score)
    return mapping


def apply_names(
    segments: list[AttributedSegment], mapping: Mapping[str, str]
) -> list[AttributedSegment]:
    """Return segments with speakers renamed per ``mapping`` (others unchanged)."""
    return [
        AttributedSegment(
            speaker=mapping.get(segment.speaker, segment.speaker),
            start=segment.start,
            end=segment.end,
            text=segment.text,
        )
        for segment in segments
    ]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two vectors; 0.0 if either has zero magnitude."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
