"""Audio window boundaries measured from real PCM files and ffmpeg output."""

from __future__ import annotations

import array
import shutil
import wave
from pathlib import Path

import pytest

from localscribe.audio import split_into_windows


@pytest.mark.parametrize("sample_rate", [16000, 48000])
def test_windows_preserve_every_sample_at_exact_bounded_offsets(
    tmp_path: Path, sample_rate: int
) -> None:
    source = tmp_path / "source.wav"
    # Nonconstant PCM makes duplicated, omitted, or reordered samples observable.
    samples = array.array("h", ((index % 30001) - 15000 for index in range(sample_rate * 4 + 7)))
    original = samples.tobytes()
    with wave.open(str(source), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(original)

    windows = split_into_windows(source, 1)
    assert windows
    try:
        frames_before = 0
        reconstructed = bytearray()
        for offset, path in windows:
            with wave.open(str(path), "rb") as chunk:
                assert offset == frames_before / sample_rate
                assert 0 < chunk.getnframes() <= sample_rate
                assert chunk.getframerate() == sample_rate
                frames_before += chunk.getnframes()
                reconstructed.extend(chunk.readframes(chunk.getnframes()))
        assert reconstructed == original
    finally:
        shutil.rmtree(windows[0][1].parent)
