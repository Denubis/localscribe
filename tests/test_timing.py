"""Word timing against the sound itself: gaps between passes, and word ends.

The ASR's word stamps touch end to end: measured 2026-10-06 on two meetings, 70%
of consecutive words had no gap at all and about a third of real 0.2-0.5 s pauses
were hidden inside a word's span. Paragraph breaks on pauses need those gaps to
exist, so word ends are pulled back to where the sound actually stops.
"""

from __future__ import annotations

from localscribe.models import AttributedSegment, SpeakerTurn, Word
from localscribe.reconcile import reconcile
from localscribe.timing import fill_gaps, trim_word_ends

FRAME = 0.02


def _levels(
    seconds: float, quiet: list[tuple[float, float]], *, loud: float = 1.0
) -> list[float]:
    frames = round(seconds / FRAME)
    return [
        0.001 if any(start <= index * FRAME < end for start, end in quiet) else loud
        for index in range(frames)
    ]


class TestFillGaps:
    def test_words_from_the_second_pass_fill_a_gap_in_the_first(self) -> None:
        primary = [Word("a", 0.0, 1.0), Word("d", 5.0, 6.0)]
        secondary = [
            Word("a", 0.0, 1.1), Word("b", 1.5, 2.0), Word("c", 2.5, 3.0), Word("d", 4.9, 6.0)
        ]
        assert fill_gaps(primary, secondary, min_seconds=1.0) == [
            Word("a", 0.0, 1.0), Word("b", 1.5, 2.0), Word("c", 2.5, 3.0), Word("d", 5.0, 6.0)
        ]

    def test_a_short_gap_is_left_alone(self) -> None:
        # Two passes disagree about short stretches constantly; only a real hole is filled.
        primary = [Word("a", 0.0, 1.0), Word("d", 1.8, 3.0)]
        secondary = [Word("x", 1.1, 1.7)]
        assert fill_gaps(primary, secondary, min_seconds=1.0) == primary

    def test_speech_before_the_first_word_is_a_gap_too(self) -> None:
        primary = [Word("b", 4.0, 5.0)]
        secondary = [Word("a", 1.0, 2.0), Word("b", 4.0, 5.0)]
        assert fill_gaps(primary, secondary, min_seconds=1.0) == [
            Word("a", 1.0, 2.0), Word("b", 4.0, 5.0)
        ]


class TestTrimWordEnds:
    def test_a_word_whose_span_runs_into_a_pause_ends_where_the_sound_does(self) -> None:
        levels = _levels(3.0, quiet=[(1.0, 1.6)])
        words = [Word("stop", 0.5, 1.6), Word("go", 1.6, 2.5)]
        trimmed = trim_word_ends(words, levels, frame_seconds=FRAME, min_pause_seconds=0.1)
        assert trimmed[0].end == 1.0
        assert trimmed[1] == Word("go", 1.6, 2.5)

    def test_a_dip_shorter_than_the_minimum_pause_is_not_a_pause(self) -> None:
        levels = _levels(3.0, quiet=[(1.54, 1.6)])
        words = [Word("stop", 0.5, 1.6)]
        assert trim_word_ends(words, levels, frame_seconds=FRAME, min_pause_seconds=0.1) == words

    def test_a_soft_speaker_is_judged_against_their_own_voice_not_the_loud_ones(self) -> None:
        # The loud speaker fills most of the recording; the soft one is 30 dB down,
        # which is below a tenth of the loud level. Their pause is still a pause.
        levels = _levels(60.0, quiet=[(51.0, 51.6)])
        for index in range(round(50.0 / FRAME), round(60.0 / FRAME)):
            levels[index] = min(levels[index], 0.03)
        words = [Word("soft", 50.5, 51.6)]
        trimmed = trim_word_ends(words, levels, frame_seconds=FRAME, min_pause_seconds=0.1)
        assert trimmed[0].end == 51.0

    def test_a_word_is_never_trimmed_to_nothing(self) -> None:
        levels = _levels(3.0, quiet=[(0.0, 3.0)])  # the whole span is quiet (a soft speaker)
        words = [Word("hmm", 0.5, 1.0)]
        assert trim_word_ends(words, levels, frame_seconds=FRAME, min_pause_seconds=0.1) == words


class TestPauseBreaks:
    def test_a_pause_over_the_limit_starts_a_new_segment_with_its_own_time(self) -> None:
        words = [Word("one.", 1.0, 1.4), Word("Two", 2.6, 3.0)]
        turns = [SpeakerTurn("S", 0.0, 4.0)]
        assert reconcile(words, turns, pause_seconds=1.0) == [
            AttributedSegment("S", 1.0, 1.4, "one."),
            AttributedSegment("S", 2.6, 3.0, "Two"),
        ]

    def test_a_pause_mid_sentence_does_not_break_the_paragraph(self) -> None:
        # The word before the pause ends no sentence; a hesitation (or a dropped
        # moment of audio) should not split "1500 is where / we start".
        words = [Word("where", 1.0, 1.4), Word("we", 2.6, 3.0)]
        turns = [SpeakerTurn("S", 0.0, 4.0)]
        assert reconcile(words, turns, pause_seconds=1.0) == [
            AttributedSegment("S", 1.0, 3.0, "where we")
        ]

    def test_a_long_pause_mid_sentence_still_breaks(self) -> None:
        words = [Word("where", 1.0, 1.4), Word("we", 4.0, 4.4)]
        turns = [SpeakerTurn("S", 0.0, 5.0)]
        assert reconcile(words, turns, pause_seconds=1.0, hesitation_seconds=2.5) == [
            AttributedSegment("S", 1.0, 1.4, "where"),
            AttributedSegment("S", 4.0, 4.4, "we"),
        ]

    def test_a_pause_under_the_limit_keeps_the_paragraph_together(self) -> None:
        words = [Word("one.", 1.0, 1.4), Word("Two", 2.3, 3.0)]
        turns = [SpeakerTurn("S", 0.0, 4.0)]
        assert reconcile(words, turns, pause_seconds=1.0) == [
            AttributedSegment("S", 1.0, 3.0, "one. Two")
        ]
