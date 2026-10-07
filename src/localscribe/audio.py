# pattern: Imperative Shell
"""Audio preprocessing: downmix to mono 16 kHz via ffmpeg.

Both pyannote and Parakeet want mono 16 kHz. The meeting FLACs here are stereo,
and NeMo's loader does not downmix on this path, so we normalise once up front and
feed the result to both stages. ffmpeg streams the conversion, so even an
80-minute recording converts without loading the whole signal into memory.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import wave
from collections.abc import Sequence
from pathlib import Path

from .errors import AudioError

logger = logging.getLogger(__name__)


def to_mono_16k(recording: Path) -> Path:
    """Convert ``recording`` to a temporary mono 16 kHz WAV and return its path.

    The caller owns the returned file and must delete it when done.
    """
    handle, name = tempfile.mkstemp(suffix=".wav", prefix="localscribe-")
    os.close(handle)
    temp = Path(name)
    command = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(recording),
        "-ac", "1", "-ar", "16000",
        str(temp),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
    except FileNotFoundError as exc:
        temp.unlink(missing_ok=True)
        raise AudioError("ffmpeg not found on PATH; it is required to read audio") from exc
    except subprocess.CalledProcessError as exc:
        temp.unlink(missing_ok=True)
        detail = exc.stderr.decode(errors="replace").strip()
        raise AudioError(f"ffmpeg failed to preprocess {recording}: {detail}") from exc
    logger.debug("prepared mono 16k wav for %s at %s", recording, temp)
    return temp


#: Length of one level-measurement frame: fine enough to see the short pause
#: between two words; an hour of audio is 180,000 values.
LEVEL_FRAME_SECONDS = 0.02

#: Levels are averaged over this long before the quietest point is chosen, so a
#: cut prefers a real pause to a single quiet frame inside a word.
_SMOOTHING_SECONDS = 0.5

_LEVEL_KEY = "lavfi.astats.Overall.RMS_level="


def choose_cut_points(
    levels: Sequence[float],
    *,
    frame_seconds: float,
    target_seconds: float,
    search_seconds: float,
) -> list[float]:
    """Choose where to cut a recording into windows, preferring pauses.

    Pure. ``levels`` is one loudness value per ``frame_seconds`` of audio, larger
    meaning louder. Each window ends at the quietest point in the last
    ``search_seconds`` before it would reach ``target_seconds``, so no window is
    longer than the target and none is shorter than ``target - search``. The final
    window takes whatever is left, up to ``target + search``.
    """
    target = round(target_seconds / frame_seconds)
    search = round(search_seconds / frame_seconds)
    if not 1 <= search < target:
        raise ValueError(
            "search span must be at least one frame and shorter than the window: "
            f"search={search_seconds}s target={target_seconds}s"
        )
    smoothed = _moving_average(levels, max(1, round(_SMOOTHING_SECONDS / frame_seconds)))
    cuts: list[float] = []
    last = 0
    while len(levels) - last > target + search:
        last = min(range(last + target - search, last + target), key=smoothed.__getitem__)
        cuts.append(last * frame_seconds)
    return cuts


def _moving_average(values: Sequence[float], width: int) -> list[float]:
    """Centred moving average; the window shrinks at the two ends."""
    totals = [0.0]
    for value in values:
        totals.append(totals[-1] + value)
    half = width // 2
    averaged: list[float] = []
    for index in range(len(values)):
        low, high = max(0, index - half), min(len(values), index + half + 1)
        averaged.append((totals[high] - totals[low]) / (high - low))
    return averaged


#: Audio quieter than the surrounding speech by more than this ratio is not
#: counted as speech when looking for holes (20 dB; a tenth of the amplitude).
_SPEECH_RATIO = 0.1

#: Which level counts as the surrounding speech's loud level: its 90th
#: percentile, so brief peaks and the pauses in it both leave it alone.
_LOUD_PERCENTILE = 0.9

#: How much audio either side of a gap sets the speech level it is judged against,
#: so a soft speaker's gap is compared with that speaker, not the loudest one.
_NEIGHBOURHOOD_SECONDS = 10.0


def find_holes(
    levels: Sequence[float],
    spans: Sequence[tuple[float, float]],
    *,
    frame_seconds: float,
    min_seconds: float,
    speech_fraction: float,
) -> list[tuple[float, float]]:
    """Find stretches with no transcribed words where the audio is at speaking level.

    Pure. ``spans`` are the ``(start, end)`` times of transcribed words in order.
    A gap between them (or before the first, or after the last) is a hole when it
    lasts at least ``min_seconds`` and at least ``speech_fraction`` of its frames
    are within ``_SPEECH_RATIO`` of the loud level of the surrounding
    ``_NEIGHBOURHOOD_SECONDS`` of audio.
    """
    if not levels:
        return []
    duration = len(levels) * frame_seconds
    reach = round(_NEIGHBOURHOOD_SECONDS / frame_seconds)
    edges = [0.0, *(time for span in spans for time in span), duration]
    holes: list[tuple[float, float]] = []
    for start, end in zip(edges[::2], edges[1::2], strict=True):
        if end - start < min_seconds:
            continue
        first, last = round(start / frame_seconds), round(end / frame_seconds)
        frames = levels[first:last]
        around = sorted(levels[max(0, first - reach) : last + reach])
        threshold = around[round(_LOUD_PERCENTILE * (len(around) - 1))] * _SPEECH_RATIO
        if frames and sum(level > threshold for level in frames) >= speech_fraction * len(frames):
            holes.append((start, end))
    return holes


def parse_levels(report: str) -> list[float]:
    """Parse ffmpeg's per-frame RMS report into linear levels, one per frame.

    Pure. Digital silence is reported as ``-inf`` dB and becomes ``0.0``. A line
    that cannot be read is skipped rather than failing the whole recording.
    """
    levels: list[float] = []
    for line in report.splitlines():
        if not line.startswith(_LEVEL_KEY):
            continue
        try:
            decibels = float(line[len(_LEVEL_KEY):])
        except ValueError:
            logger.warning("skipping unreadable level line: %r", line)
            continue
        levels.append(10 ** (decibels / 20))
    return levels


def measure_levels(wav: Path) -> list[float]:
    """Measure the loudness of ``wav`` once per ``LEVEL_FRAME_SECONDS``."""
    samples = round(_sample_rate(wav) * LEVEL_FRAME_SECONDS)
    command = [
        "ffmpeg", "-nostdin", "-v", "error",
        "-i", str(wav),
        "-af",
        f"asetnsamples=n={samples}:p=0,astats=metadata=1:reset=1,"
        f"ametadata=mode=print:key={_LEVEL_KEY.rstrip('=')}:file=-",
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise AudioError(f"ffmpeg failed to measure levels of {wav}: {exc.stderr.strip()}") from exc
    return parse_levels(result.stdout)


def split_into_windows(wav: Path, cuts: Sequence[float]) -> list[tuple[float, Path]]:
    """Split ``wav`` at ``cuts`` (seconds), returning ``(start_offset, chunk_path)``.

    ffmpeg cuts at the first packet at or after each requested time, not exactly
    on it, so each offset is the audio actually written before that chunk. Chunks
    share a fresh temp directory; the caller must remove them (``shutil.rmtree``
    on the parent). No cuts yields the whole recording as one chunk at offset 0.
    """
    chunk_dir = Path(tempfile.mkdtemp(prefix="localscribe-chunks-"))
    if not cuts:
        whole = chunk_dir / "chunk_00000.wav"
        shutil.copyfile(wav, whole)
        return [(0.0, whole)]
    command = [
        "ffmpeg", "-nostdin", "-y", "-v", "error",
        "-i", str(wav),
        "-f", "segment", "-segment_times", ",".join(f"{cut:.3f}" for cut in cuts),
        "-c", "copy",
        str(chunk_dir / "chunk_%05d.wav"),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        shutil.rmtree(chunk_dir, ignore_errors=True)
        detail = exc.stderr.decode(errors="replace").strip()
        raise AudioError(f"ffmpeg failed to split {wav}: {detail}") from exc
    windows: list[tuple[float, Path]] = []
    offset = 0.0
    for chunk in sorted(chunk_dir.glob("chunk_*.wav")):
        windows.append((offset, chunk))
        offset += _duration_seconds(chunk)
    return windows


def extract_span(wav: Path, start: float, end: float) -> Path:
    """Copy the audio between ``start`` and ``end`` seconds into a new temp file.

    The caller must unlink the returned path.
    """
    handle, name = tempfile.mkstemp(prefix="localscribe-span-", suffix=".wav")
    os.close(handle)
    clip = Path(name)
    command = [
        "ffmpeg", "-nostdin", "-y", "-v", "error",
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
        "-i", str(wav),
        "-c", "copy",
        str(clip),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        clip.unlink(missing_ok=True)
        detail = exc.stderr.decode(errors="replace").strip()
        raise AudioError(
            f"ffmpeg failed to extract {start:.1f}-{end:.1f}s of {wav}: {detail}"
        ) from exc
    return clip


def _sample_rate(wav: Path) -> int:
    with wave.open(str(wav)) as handle:
        return handle.getframerate()


def _duration_seconds(wav: Path) -> float:
    with wave.open(str(wav)) as handle:
        return handle.getnframes() / handle.getframerate()
