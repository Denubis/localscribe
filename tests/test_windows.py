"""Transcription windows: where a long recording is cut, and where each piece starts.

Measured 2026-10-06 on five 54-69 minute meetings: fed 20-minute windows, the ASR
model left 75 stretches of audible speech untranscribed (one of them 29 seconds at
normal speaking volume); fed windows of about 30 seconds it left 34. Short windows
mean many more cuts, so a cut has to land in a pause rather than through a word,
and every window's timestamps have to be shifted by where it really starts.

The ffmpeg tests run the real binary, because both things in doubt are its
behaviour: the level report's shape, and the fact that the segment muxer cuts on
packet boundaries rather than exactly where it is asked to.
"""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from localscribe.audio import (
    LEVEL_FRAME_SECONDS,
    choose_cut_points,
    extract_span,
    find_holes,
    measure_levels,
    parse_levels,
    split_into_windows,
)
from localscribe.models import Word
from localscribe.transcribe import fill_holes

FRAME = 0.1


def _levels(seconds: float, pauses: list[tuple[float, float]]) -> list[float]:
    """Constant speech-like level with silence inside each ``(start, end)`` pause."""
    frames = round(seconds / FRAME)
    return [
        0.0 if any(start <= index * FRAME < end for start, end in pauses) else 1.0
        for index in range(frames)
    ]


def _tone_with_silence(target: Path, *, seconds: float, silent: tuple[float, float]) -> Path:
    """A mono 16 kHz WAV, the shape the pipeline prepares, silent between two times."""
    start, end = silent
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi",
            "-i",
            f"sine=f=440:d={seconds}:r=16000,"
            f"volume='if(between(t,{start},{end}),0,1)':eval=frame",
            "-ac", "1", "-c:a", "pcm_s16le", str(target),
        ],
        check=True,
    )
    return target


def _frames(wav: Path) -> int:
    with wave.open(str(wav)) as handle:
        return handle.getnframes()


class TestChooseCutPoints:
    def test_each_cut_lands_inside_the_pause_before_its_window_would_overrun(self) -> None:
        pauses = [(25.0, 26.0), (52.0, 53.0)]
        cuts = choose_cut_points(
            _levels(80.0, pauses), frame_seconds=FRAME, target_seconds=30.0, search_seconds=10.0
        )
        assert len(cuts) == len(pauses)
        for cut, (start, end) in zip(cuts, pauses, strict=True):
            assert start <= cut < end

    def test_a_recording_that_fits_in_one_window_is_not_cut(self) -> None:
        cuts = choose_cut_points(
            _levels(35.0, [(10.0, 11.0)]),
            frame_seconds=FRAME, target_seconds=30.0, search_seconds=10.0,
        )
        assert cuts == []

    @given(
        levels=st.lists(st.floats(min_value=0.0, max_value=1.0), max_size=4000),
        target=st.integers(min_value=2, max_value=60),
        data=st.data(),
    )
    def test_no_window_is_longer_than_the_target_however_the_audio_looks(
        self, levels: list[float], target: int, data: st.DataObject
    ) -> None:
        # The bound is the guardrail: an unbounded window is what exhausts the GPU,
        # and a long one is what loses speech.
        search = data.draw(st.integers(min_value=1, max_value=target - 1))
        cuts = choose_cut_points(
            levels, frame_seconds=FRAME, target_seconds=float(target), search_seconds=float(search)
        )
        duration = len(levels) * FRAME
        edges = [0.0, *cuts, duration]
        lengths = [later - earlier for earlier, later in zip(edges, edges[1:], strict=False)]
        tolerance = FRAME / 2
        assert all(length <= target + tolerance for length in lengths[:-1])
        assert all(length >= target - search - tolerance for length in lengths[:-1])
        assert lengths[-1] <= target + search + tolerance

    def test_a_search_span_as_long_as_the_window_is_refused(self) -> None:
        # It would allow a zero-length window, and so no progress.
        with pytest.raises(ValueError, match="search"):
            choose_cut_points(
                [1.0] * 100, frame_seconds=FRAME, target_seconds=5.0, search_seconds=5.0
            )


class TestParseLevels:
    # Verbatim shape printed by ffmpeg 8.0.1's astats + ametadata for a tone that
    # falls silent: one frame header, then one level line, digital silence as -inf.
    REPORT = (
        "frame:19   pts:30400   pts_time:1.9\n"
        "lavfi.astats.Overall.RMS_level=-21.073765\n"
        "frame:20   pts:32000   pts_time:2\n"
        "lavfi.astats.Overall.RMS_level=-24.280228\n"
        "frame:21   pts:33600   pts_time:2.1\n"
        "lavfi.astats.Overall.RMS_level=-inf\n"
    )

    def test_levels_come_back_in_order_and_louder_frames_are_larger(self) -> None:
        loud, quieter, silent = parse_levels(self.REPORT)
        assert loud > quieter > silent == 0.0

    def test_an_unreadable_line_does_not_discard_the_rest(self) -> None:
        report = self.REPORT + "lavfi.astats.Overall.RMS_level=garbage\n" + self.REPORT
        assert len(parse_levels(report)) == 6


class TestRealFfmpeg:
    def test_the_measured_levels_put_the_cut_in_the_real_silence(self, tmp_path: Path) -> None:
        wav = _tone_with_silence(tmp_path / "tone.wav", seconds=5, silent=(2.0, 3.0))
        levels = measure_levels(wav)
        assert len(levels) == pytest.approx(5 / LEVEL_FRAME_SECONDS, abs=1)
        cuts = choose_cut_points(
            levels, frame_seconds=LEVEL_FRAME_SECONDS, target_seconds=3.0, search_seconds=1.5
        )
        assert len(cuts) == 1
        assert 2.0 <= cuts[0] <= 3.0

    def test_windows_lose_no_audio_and_report_where_each_really_starts(
        self, tmp_path: Path
    ) -> None:
        wav = _tone_with_silence(tmp_path / "tone.wav", seconds=5, silent=(2.0, 3.0))
        requested = [1.37, 3.9]
        windows = split_into_windows(wav, requested)
        try:
            lengths = [_frames(chunk) for _, chunk in windows]
            assert sum(lengths) == _frames(wav)
            # ffmpeg cuts at the next packet, not at the requested instant, so the
            # offset that keeps timestamps true is the audio actually before the window.
            starts = [sum(lengths[:index]) / 16000 for index in range(len(lengths))]
            assert [offset for offset, _ in windows] == pytest.approx(starts)
            assert starts[1:] == pytest.approx(requested, abs=0.3)
        finally:
            for _, chunk in windows:
                chunk.unlink(missing_ok=True)
            windows[0][1].parent.rmdir()

    def test_no_cuts_gives_the_whole_recording_as_one_window_at_zero(
        self, tmp_path: Path
    ) -> None:
        wav = _tone_with_silence(tmp_path / "tone.wav", seconds=5, silent=(2.0, 3.0))
        windows = split_into_windows(wav, [])
        try:
            assert [offset for offset, _ in windows] == [0.0]
            assert _frames(windows[0][1]) == _frames(wav)
        finally:
            windows[0][1].unlink(missing_ok=True)
            windows[0][1].parent.rmdir()


# Even with short windows the model still skipped speech mid-window (measured
# 2026-10-07, fifteen meetings: 35 stretches another model transcribed). Those show
# up as a silent stretch in the transcript while the audio is at speaking level;
# transcribing just that stretch on its own recovered most of them.


class TestFindHoles:
    def test_a_quiet_transcript_over_loud_audio_is_a_hole(self) -> None:
        levels = _levels(40.0, [(10.0, 12.0)])
        spans = [(0.0, 10.0), (12.0, 20.0), (30.0, 40.0)]  # nothing transcribed 20-30
        holes = find_holes(
            levels, spans, frame_seconds=FRAME, min_seconds=1.0, speech_fraction=0.4
        )
        assert holes == [(20.0, 30.0)]

    def test_a_hole_in_a_soft_speaker_is_judged_against_their_neighbourhood(self) -> None:
        # A loud speaker for most of the hour; a soft one, 30 dB down, for the last
        # minute, of which 20-30 s went untranscribed.
        levels = _levels(3600.0, [])
        for index in range(round(3540.0 / FRAME), round(3600.0 / FRAME)):
            levels[index] = 0.03
        spans = [(0.0, 3540.0), (3540.0, 3560.0), (3570.0, 3600.0)]
        holes = find_holes(
            levels, spans, frame_seconds=FRAME, min_seconds=1.0, speech_fraction=0.4
        )
        assert holes == [(3560.0, 3570.0)]

    def test_a_pause_is_not_a_hole(self) -> None:
        levels = _levels(40.0, [(20.0, 30.0)])
        spans = [(0.0, 20.0), (30.0, 40.0)]
        assert (
            find_holes(levels, spans, frame_seconds=FRAME, min_seconds=1.0, speech_fraction=0.4)
            == []
        )

    def test_a_gap_shorter_than_the_minimum_is_not_a_hole(self) -> None:
        levels = _levels(40.0, [])
        spans = [(0.0, 20.0), (20.8, 40.0)]
        assert (
            find_holes(levels, spans, frame_seconds=FRAME, min_seconds=1.0, speech_fraction=0.4)
            == []
        )

    def test_speech_before_the_first_and_after_the_last_word_counts(self) -> None:
        levels = _levels(40.0, [])
        holes = find_holes(
            levels, [(5.0, 35.0)], frame_seconds=FRAME, min_seconds=1.0, speech_fraction=0.4
        )
        assert holes == [(0.0, 5.0), (35.0, 40.0)]


class TestFillHoles:
    def test_filled_words_land_at_absolute_times_and_only_inside_the_hole(self) -> None:
        words = [Word("a", 0.0, 1.0), Word("b", 9.0, 10.0)]
        calls: list[tuple[float, float]] = []

        def transcribe_span(start: float, end: float) -> list[Word]:
            calls.append((start, end))
            # Relative to the clip: the first and last words overlap the clip's
            # padding and belong to the neighbours already transcribed.
            return [
                Word("a", 0.0, 0.6), Word("x", 1.0, 2.0), Word("y", 3.0, 4.0), Word("b", 8.9, 9.6)
            ]

        merged = fill_holes(words, [(1.0, 9.0)], padding=0.5, transcribe_span=transcribe_span)
        assert calls == [(0.5, 9.5)]
        assert [w.text for w in merged] == ["a", "x", "y", "b"]
        assert merged[1] == Word("x", 1.5, 2.5)

    def test_a_hole_at_the_very_start_is_not_padded_before_zero(self) -> None:
        calls: list[tuple[float, float]] = []

        def transcribe_span(start: float, end: float) -> list[Word]:
            calls.append((start, end))
            return []

        fill_holes(
            [Word("a", 3.0, 4.0)], [(0.0, 3.0)], padding=0.5, transcribe_span=transcribe_span
        )
        assert calls == [(0.0, 3.5)]


class TestExtractSpan:
    def test_the_clip_is_the_requested_stretch(self, tmp_path: Path) -> None:
        wav = _tone_with_silence(tmp_path / "tone.wav", seconds=5, silent=(2.0, 3.0))
        clip = extract_span(wav, 1.0, 3.5)
        try:
            assert _frames(clip) / 16000 == pytest.approx(2.5, abs=0.1)
            levels = measure_levels(clip)
            per_second = round(1 / LEVEL_FRAME_SECONDS)
            # The silence 2-3 s in the source is now 1-2 s into the clip.
            silent = levels[round(1.2 * per_second) : round(1.8 * per_second)]
            loud = levels[round(0.2 * per_second) : round(0.8 * per_second)]
            assert max(silent) < min(loud) / 100
        finally:
            clip.unlink(missing_ok=True)
