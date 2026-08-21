"""Behaviour tests for the rendering core (segments -> markdown / json strings)."""

import json

from hypothesis import given
from hypothesis import strategies as st

from localscribe.models import AttributedSegment
from localscribe.render import format_timestamp, render_json, render_markdown


def test_render_json_emits_segments_and_source() -> None:
    segments = [AttributedSegment(speaker="SPEAKER_00", start=1.0, end=2.0, text="hello world")]

    data = json.loads(render_json(segments, source="meeting.flac"))

    assert data["source"] == "meeting.flac"
    assert data["segments"] == [
        {"speaker": "SPEAKER_00", "start": 1.0, "end": 2.0, "text": "hello world"}
    ]


def test_format_timestamp_renders_hours_minutes_seconds() -> None:
    assert format_timestamp(0.0) == "0:00:00"
    assert format_timestamp(65.0) == "0:01:05"
    assert format_timestamp(3661.0) == "1:01:01"


def test_render_markdown_labels_each_segment_with_speaker_and_time() -> None:
    segments = [
        AttributedSegment(speaker="SPEAKER_00", start=65.0, end=70.0, text="hello world"),
        AttributedSegment(speaker="SPEAKER_01", start=71.0, end=73.0, text="goodbye"),
    ]

    md = render_markdown(segments, source="meeting.flac")

    assert "meeting.flac" in md
    assert "SPEAKER_00" in md
    assert "0:01:05" in md
    assert "hello world" in md
    assert "SPEAKER_01" in md
    assert "goodbye" in md


def test_empty_segments_render_without_error() -> None:
    assert json.loads(render_json([], source="silent.flac"))["segments"] == []
    assert "silent.flac" in render_markdown([], source="silent.flac")


_finite = st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)
_segments = st.lists(
    st.builds(AttributedSegment, speaker=st.text(), start=_finite, end=_finite, text=st.text()),
    max_size=30,
)


@given(segments=_segments)
def test_render_json_roundtrips_segment_values(segments: list[AttributedSegment]) -> None:
    data = json.loads(render_json(segments, source="s"))
    assert data["segments"] == [
        {"speaker": s.speaker, "start": s.start, "end": s.end, "text": s.text} for s in segments
    ]


