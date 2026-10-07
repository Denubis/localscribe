# pattern: Functional Core
"""Word timing checked against the sound itself.

Pure. Two jobs:

- ``fill_gaps``: a second transcription pass is the witness for the first. Where
  the first pass left a stretch with no words, the second pass's words for that
  stretch are taken in. (The ASR skips speech now and then; two passes with
  different windows rarely skip the same stretch.)
- ``trim_word_ends``: the ASR stamps words end to end, so a pause between two
  words is usually hidden inside the earlier word's span. Pulling each word's end
  back to where the sound stops makes pauses visible to whatever reads the
  timeline, such as paragraph breaks.
"""

from __future__ import annotations

from collections.abc import Sequence

from .models import Word

#: Audio quieter than the speaker's own loud level by more than this ratio counts
#: as a pause (20 dB; a tenth of the amplitude).
PAUSE_RATIO = 0.1

#: Which level counts as a stretch's loud level: its 90th percentile, so brief
#: peaks and any quiet frames in it both leave it alone.
LOUD_PERCENTILE = 0.9


def fill_gaps(
    primary: Sequence[Word], secondary: Sequence[Word], *, min_seconds: float
) -> list[Word]:
    """Return ``primary`` plus the ``secondary`` words centred inside its gaps.

    A gap is a stretch of at least ``min_seconds`` with no primary word, including
    before the first word. Shorter gaps are ordinary pauses and left alone.
    """
    if not primary:
        return list(secondary)
    edges = [0.0, *(time for word in primary for time in (word.start, word.end)), float("inf")]
    gaps = [
        (start, end)
        for start, end in zip(edges[::2], edges[1::2], strict=True)
        if end - start >= min_seconds
    ]
    taken = [
        word
        for word in secondary
        if any(start <= (word.start + word.end) / 2 <= end for start, end in gaps)
    ]
    return sorted([*primary, *taken], key=lambda word: word.start)


def trim_word_ends(
    words: Sequence[Word],
    levels: Sequence[float],
    *,
    frame_seconds: float,
    min_pause_seconds: float,
) -> list[Word]:
    """Pull each word's end back to where the sound stops, if a pause follows it.

    ``levels`` is one loudness value per ``frame_seconds``. Quiet is judged
    against the word's own loud level, so a soft speaker's pauses count the same
    as a loud speaker's. A word is trimmed only when the quiet run at the end of
    its span lasts at least ``min_pause_seconds``, and never to nothing.
    """
    if not levels:
        return list(words)
    min_frames = max(1, round(min_pause_seconds / frame_seconds))
    trimmed: list[Word] = []
    for word in words:
        first = round(word.start / frame_seconds)
        last = min(len(levels), round(word.end / frame_seconds))
        if last <= first:
            trimmed.append(word)
            continue
        threshold = loud_level(levels[first:last]) * PAUSE_RATIO
        end = last
        while end > first and levels[end - 1] <= threshold:
            end -= 1
        if end > first and last - end >= min_frames:
            trimmed.append(Word(text=word.text, start=word.start, end=end * frame_seconds))
        else:
            trimmed.append(word)
    return trimmed


def loud_level(levels: Sequence[float]) -> float:
    """The level speech in this recording is typically at (see ``LOUD_PERCENTILE``)."""
    ranked = sorted(levels)
    return ranked[round(LOUD_PERCENTILE * (len(ranked) - 1))]
