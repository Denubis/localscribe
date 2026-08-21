# pattern: Functional Core
"""Map ASR word timestamps onto diarisation speaker turns.

Pure. Given words (text + timing) and speaker turns (speaker + timing), produce
speaker-attributed segments. No word is ever dropped (guardrail #2): a word that
falls outside every turn is still attributed, to the nearest turn.
"""

from __future__ import annotations

from .models import AttributedSegment, SpeakerTurn, Word

#: Speaker label for words the diarisation could not place (e.g. it returned no
#: turns at all). Keeps the transcript whole rather than discarding words.
UNKNOWN_SPEAKER = "UNKNOWN"


def reconcile(words: list[Word], turns: list[SpeakerTurn]) -> list[AttributedSegment]:
    """Attribute each word to a speaker and merge consecutive same-speaker words.

    Words are attributed in order, then collapsed into runs: a maximal sequence of
    consecutive words sharing a speaker becomes one segment. Text is joined once per
    run, so a long single-speaker stretch stays O(n) rather than O(n^2).
    """
    if not words:
        return []

    attributed = [(word, _speaker_for(word, turns)) for word in words]

    segments: list[AttributedSegment] = []
    run = [attributed[0]]
    for word, speaker in attributed[1:]:
        if speaker == run[-1][1]:
            run.append((word, speaker))
        else:
            segments.append(_segment(run))
            run = [(word, speaker)]
    segments.append(_segment(run))
    return segments


def _segment(run: list[tuple[Word, str]]) -> AttributedSegment:
    """Build one segment from a run of consecutive same-speaker (word, speaker) pairs."""
    speaker = run[0][1]
    words = [word for word, _ in run]
    return AttributedSegment(
        speaker=speaker,
        start=words[0].start,
        end=words[-1].end,
        text=" ".join(word.text for word in words),
    )


def _speaker_for(word: Word, turns: list[SpeakerTurn]) -> str:
    """Attribute one word to a speaker by the turn covering its midpoint.

    The midpoint (rather than start or end) is robust to small timing drift at
    word boundaries. A word covered by no turn falls back to the nearest turn so
    it is never dropped.
    """
    if not turns:
        return UNKNOWN_SPEAKER
    midpoint = (word.start + word.end) / 2.0
    for turn in turns:
        if turn.start <= midpoint <= turn.end:
            return turn.speaker
    return _nearest_turn(midpoint, turns).speaker


def _nearest_turn(point: float, turns: list[SpeakerTurn]) -> SpeakerTurn:
    """Return the turn whose span is closest to ``point`` (0 distance if inside)."""
    return min(turns, key=lambda turn: _gap(point, turn))


def _gap(point: float, turn: SpeakerTurn) -> float:
    if point < turn.start:
        return turn.start - point
    if point > turn.end:
        return point - turn.end
    return 0.0
