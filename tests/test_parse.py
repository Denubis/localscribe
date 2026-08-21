"""Behaviour tests for the boundary parsers (model output -> core value objects).

These encode guardrail #2 at the ingestion boundary: a single malformed entry
from the ASR or diarisation must be dropped and logged, never crash the run or
discard the whole transcript.
"""

from localscribe.models import SpeakerTurn, Word
from localscribe.parse import parse_turns, parse_words, stitch_windows


def test_parse_words_maps_nemo_segment_stamps() -> None:
    stamps = [
        {"segment": "hello", "start": 1.0, "end": 1.5},
        {"segment": "world", "start": 1.6, "end": 2.0},
    ]

    assert parse_words(stamps) == [
        Word(text="hello", start=1.0, end=1.5),
        Word(text="world", start=1.6, end=2.0),
    ]


def test_parse_words_accepts_word_key_as_text_fallback() -> None:
    # NeMo versions have varied between 'segment' and 'word' for the token text.
    assert parse_words([{"word": "hi", "start": 0.0, "end": 0.5}]) == [
        Word(text="hi", start=0.0, end=0.5)
    ]


def test_parse_words_skips_malformed_stamp_without_losing_the_rest() -> None:
    stamps = [
        {"segment": "good", "start": 1.0, "end": 1.5},
        {"start": 2.0, "end": 2.5},  # no text -> drop this one only
        {"segment": "also", "start": 3.0, "end": 3.5},
    ]

    assert parse_words(stamps) == [
        Word(text="good", start=1.0, end=1.5),
        Word(text="also", start=3.0, end=3.5),
    ]


def test_stitch_windows_offsets_each_window_and_concatenates_in_order() -> None:
    # Long audio is transcribed in windows; each window's timestamps are relative
    # to its own start and must be shifted back onto the full-recording timeline.
    window0 = [Word(text="hello", start=0.0, end=0.5), Word(text="world", start=1.0, end=1.5)]
    window1 = [Word(text="again", start=0.2, end=0.6)]

    result = stitch_windows([(0.0, window0), (600.0, window1)])

    assert result == [
        Word(text="hello", start=0.0, end=0.5),
        Word(text="world", start=1.0, end=1.5),
        Word(text="again", start=600.2, end=600.6),
    ]


def test_stitch_windows_handles_no_windows() -> None:
    assert stitch_windows([]) == []


def test_parse_turns_maps_spans() -> None:
    spans = [(1.0, 2.0, "SPEAKER_00"), (2.0, 3.0, "SPEAKER_01")]

    assert parse_turns(spans) == [
        SpeakerTurn(speaker="SPEAKER_00", start=1.0, end=2.0),
        SpeakerTurn(speaker="SPEAKER_01", start=2.0, end=3.0),
    ]


def test_parse_turns_skips_malformed_span_without_losing_the_rest() -> None:
    spans = [(1.0, 2.0, "S0"), (None, 3.0, "S1"), (4.0, 5.0, "S2")]

    assert parse_turns(spans) == [
        SpeakerTurn(speaker="S0", start=1.0, end=2.0),
        SpeakerTurn(speaker="S2", start=4.0, end=5.0),
    ]
