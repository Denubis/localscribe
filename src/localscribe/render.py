# pattern: Functional Core
"""Render reconciled segments into output formats.

Pure string production: no file I/O here. The imperative shell writes these
strings to disk atomically. Two formats:

- JSON: the durable raw record (stable shape, machine-readable).
- Markdown: the human-facing speaker-labelled transcript.
"""

from __future__ import annotations

import json

from .models import AttributedSegment


def render_json(segments: list[AttributedSegment], *, source: str) -> str:
    """Render segments as the durable JSON record.

    ``ensure_ascii=False`` keeps accented names and non-ASCII text intact rather
    than escaping them, so the record stays faithful to what was said.
    """
    payload = {
        "source": source,
        "segments": [
            {"speaker": s.speaker, "start": s.start, "end": s.end, "text": s.text}
            for s in segments
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def format_timestamp(seconds: float) -> str:
    """Render a second offset as ``H:MM:SS`` (whole seconds).

    Hours are unpadded because a recording rarely runs to double-digit hours;
    minutes and seconds are zero-padded so columns line up when read.
    """
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def render_markdown(segments: list[AttributedSegment], *, source: str) -> str:
    """Render segments as a human-facing, speaker-labelled transcript."""
    lines = [f"# Transcript: {source}", ""]
    for segment in segments:
        lines.append(f"**{segment.speaker}** · {format_timestamp(segment.start)}")
        lines.append(segment.text)
        lines.append("")
    return "\n".join(lines)
