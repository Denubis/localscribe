"""Tests for the guard that a capture actually contains audio.

Guardrail 4 says never silently record a dead channel. A zero-byte FLAC is the
limit case of that, and on 2026-08-01 one was recorded, hashed, pushed across the
network, digest-verified on both ends and queued for the GPU before anything
noticed. Every check it passed was a check that could not fail: an empty file
hashes perfectly consistently, and ffprobe still reports an audio *stream* for it.

So the fixtures below are the real output of the real ffprobe on the real dead
file, not invented. Note what the dead one lacks: no `duration` key at all,
rather than a zero one.

`ensure_has_audio` is exercised against real files in both directions. A gate
tested only against bad input cannot show it accepts good input, and one tested
only against good input is the gate that let this through.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from localscribe.capture import ensure_has_audio, parse_probe_duration
from localscribe.errors import CaptureError

# Verbatim ffprobe output for the zero-byte FLAC recorded on 2026-08-01.
DEAD_PROBE = """{
    "programs": [

    ],
    "streams": [
        {
            "codec_type": "audio",
            "sample_rate": "0"
        }
    ],
    "format": {
        "size": "0"
    }
}"""

# Verbatim ffprobe output for target/20260609.flac, a known-good recording.
LIVE_PROBE = """{
    "programs": [

    ],
    "streams": [
        {
            "codec_type": "audio",
            "sample_rate": "44100",
            "duration": "6322.984671"
        }
    ],
    "format": {
        "duration": "6322.984671",
        "size": "334466244"
    }
}"""


class TestParseProbeDuration:
    def test_a_real_recording_reports_its_length(self) -> None:
        assert parse_probe_duration(LIVE_PROBE) == pytest.approx(6322.984671)

    def test_the_zero_byte_capture_reports_nothing(self) -> None:
        # The exact shape that got through: a stream is present, sample_rate is
        # "0", and there is no duration key anywhere.
        assert parse_probe_duration(DEAD_PROBE) == 0.0

    def test_a_stream_with_no_sample_rate_reports_nothing(self) -> None:
        probe = '{"streams": [{"codec_type": "audio", "duration": "12.0"}]}'
        assert parse_probe_duration(probe) == 0.0

    def test_a_file_with_no_audio_stream_reports_nothing(self) -> None:
        probe = '{"streams": [{"codec_type": "video", "sample_rate": "0"}]}'
        assert parse_probe_duration(probe) == 0.0

    def test_no_streams_at_all_reports_nothing(self) -> None:
        assert parse_probe_duration('{"streams": []}') == 0.0

    def test_unparseable_output_reports_nothing_rather_than_raising(self) -> None:
        # Fail safe: if we cannot confirm audio, we have not confirmed audio.
        # The caller turns 0.0 into a refusal, so degrading here still stops.
        assert parse_probe_duration("ffprobe: command not understood") == 0.0

    def test_a_duration_of_na_reports_nothing(self) -> None:
        probe = '{"streams": [{"codec_type": "audio", "sample_rate": "48000", "duration": "N/A"}]}'
        assert parse_probe_duration(probe) == 0.0

    def test_falls_back_to_the_container_duration_when_the_stream_omits_it(self) -> None:
        probe = (
            '{"streams": [{"codec_type": "audio", "sample_rate": "48000"}],'
            ' "format": {"duration": "3.5"}}'
        )
        assert parse_probe_duration(probe) == pytest.approx(3.5)


class TestEnsureHasAudio:
    """Against real files, because the point is what ffprobe really says."""

    def test_the_zero_byte_case_is_refused(self, tmp_path: Path) -> None:
        dead = tmp_path / "dead.flac"
        dead.write_bytes(b"")
        with pytest.raises(CaptureError):
            ensure_has_audio(dead)

    def test_a_truncated_file_that_is_not_audio_is_refused(self, tmp_path: Path) -> None:
        rubbish = tmp_path / "rubbish.flac"
        rubbish.write_bytes(b"not a flac at all")
        with pytest.raises(CaptureError):
            ensure_has_audio(rubbish)

    def test_a_missing_file_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(CaptureError):
            ensure_has_audio(tmp_path / "never-existed.flac")

    def test_a_real_recording_passes(self, tmp_path: Path) -> None:
        # The positive control. Without it this whole class would pass just as
        # happily if ensure_has_audio rejected everything unconditionally.
        good = tmp_path / "tone.flac"
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-v", "error", "-nostdin",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=0.3",
                "-c:a", "flac", str(good),
            ],
            check=True,
        )
        ensure_has_audio(good)
