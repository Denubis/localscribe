"""Behaviour tests for the reconciliation core (word timing -> speaker segments)."""

from localscribe.models import AttributedSegment, SpeakerTurn, Word
from localscribe.reconcile import reconcile


def test_single_word_in_single_turn_is_attributed_to_that_speaker() -> None:
    words = [Word(text="hello", start=1.0, end=1.5)]
    turns = [SpeakerTurn(speaker="SPEAKER_00", start=0.0, end=2.0)]

    segments = reconcile(words, turns)

    assert segments == [AttributedSegment(speaker="SPEAKER_00", start=1.0, end=1.5, text="hello")]


def test_consecutive_words_same_speaker_merge_into_one_segment() -> None:
    words = [Word(text="hello", start=1.0, end=1.5), Word(text="world", start=1.6, end=2.0)]
    turns = [SpeakerTurn(speaker="SPEAKER_00", start=0.0, end=3.0)]

    segments = reconcile(words, turns)

    assert segments == [
        AttributedSegment(speaker="SPEAKER_00", start=1.0, end=2.0, text="hello world")
    ]


def test_speaker_change_splits_into_two_segments() -> None:
    words = [Word(text="hi", start=0.5, end=0.9), Word(text="there", start=2.5, end=2.9)]
    turns = [
        SpeakerTurn(speaker="SPEAKER_00", start=0.0, end=1.0),
        SpeakerTurn(speaker="SPEAKER_01", start=2.0, end=3.0),
    ]

    segments = reconcile(words, turns)

    assert segments == [
        AttributedSegment(speaker="SPEAKER_00", start=0.5, end=0.9, text="hi"),
        AttributedSegment(speaker="SPEAKER_01", start=2.5, end=2.9, text="there"),
    ]


def test_word_outside_all_turns_is_attributed_to_nearest_not_dropped() -> None:
    # Guardrail #2: a word covered by no turn is kept, attributed to the nearest turn.
    words = [Word(text="stray", start=5.0, end=5.4)]
    turns = [
        SpeakerTurn(speaker="SPEAKER_00", start=0.0, end=1.0),
        SpeakerTurn(speaker="SPEAKER_01", start=2.0, end=3.0),
    ]

    segments = reconcile(words, turns)

    assert segments == [AttributedSegment(speaker="SPEAKER_01", start=5.0, end=5.4, text="stray")]


def test_no_words_yields_no_segments() -> None:
    assert reconcile([], [SpeakerTurn(speaker="SPEAKER_00", start=0.0, end=1.0)]) == []


def test_words_with_no_turns_are_kept_under_unknown_speaker() -> None:
    # Guardrail #2: empty diarisation must not discard the transcript.
    words = [Word(text="hello", start=1.0, end=1.5), Word(text="world", start=1.6, end=2.0)]

    segments = reconcile(words, [])

    assert segments == [
        AttributedSegment(speaker="UNKNOWN", start=1.0, end=2.0, text="hello world")
    ]
