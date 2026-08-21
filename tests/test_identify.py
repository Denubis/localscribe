"""Behaviour tests for the speaker-identification core (cosine matching, renaming).

Pure: works on plain float vectors, no embedding model involved.
"""

from localscribe.identify import apply_names, match_speakers
from localscribe.models import AttributedSegment


def test_match_speakers_assigns_names_above_threshold() -> None:
    enrolled = {"Alice": [1.0, 0.0], "Bob": [0.0, 1.0]}
    clusters = {"SPEAKER_00": [0.9, 0.1], "SPEAKER_01": [0.1, 0.9]}

    assert match_speakers(clusters, enrolled, threshold=0.8) == {
        "SPEAKER_00": "Alice",
        "SPEAKER_01": "Bob",
    }


def test_match_speakers_leaves_unmatched_cluster_anonymous() -> None:
    enrolled = {"Alice": [1.0, 0.0]}
    clusters = {"SPEAKER_00": [0.0, 1.0]}  # orthogonal -> cosine 0, below threshold

    assert match_speakers(clusters, enrolled, threshold=0.5) == {}


def test_match_speakers_with_no_enrolled_voices_returns_empty() -> None:
    assert match_speakers({"SPEAKER_00": [1.0, 0.0]}, {}, threshold=0.5) == {}


def test_match_speakers_picks_best_of_several_enrolled() -> None:
    enrolled = {"Alice": [1.0, 0.0], "Bob": [0.7, 0.7]}
    clusters = {"SPEAKER_00": [0.8, 0.6]}  # closer to Bob

    assert match_speakers(clusters, enrolled, threshold=0.5) == {"SPEAKER_00": "Bob"}


def test_apply_names_renames_only_mapped_speakers() -> None:
    segments = [
        AttributedSegment(speaker="SPEAKER_00", start=0.0, end=1.0, text="hello"),
        AttributedSegment(speaker="SPEAKER_01", start=1.0, end=2.0, text="hi"),
    ]

    renamed = apply_names(segments, {"SPEAKER_00": "Alice"})

    assert renamed == [
        AttributedSegment(speaker="Alice", start=0.0, end=1.0, text="hello"),
        AttributedSegment(speaker="SPEAKER_01", start=1.0, end=2.0, text="hi"),
    ]
