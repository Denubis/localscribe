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


def split_into_windows(wav: Path, window_seconds: int) -> list[tuple[float, Path]]:
    """Split ``wav`` into chunks of at most ``window_seconds`` each.

    Returns ``(start_offset, chunk_path)`` pairs in order. Chunks share a fresh
    temp directory; the caller must remove them (``shutil.rmtree`` on the parent).
    A file shorter than one window yields a single chunk at offset 0.
    """
    # Packet-copy segmentation rounds cuts to input packet boundaries. Frame the
    # normalized PCM into exact seconds before segmenting, so reported offsets
    # agree with the samples and no chunk exceeds its integer-second window.
    try:
        with wave.open(str(wav), "rb") as source:
            sample_rate = source.getframerate()
    except (OSError, EOFError, wave.Error) as exc:
        raise AudioError(f"could not read PCM WAV {wav}: {exc}") from exc
    chunk_dir = Path(tempfile.mkdtemp(prefix="localscribe-chunks-"))
    pattern = str(chunk_dir / "chunk_%05d.wav")
    command = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(wav),
        "-f", "segment", "-segment_time", str(window_seconds),
        "-af", f"asetnsamples=n={sample_rate}:p=0", "-c:a", "pcm_s16le",
        pattern,
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        shutil.rmtree(chunk_dir, ignore_errors=True)
        detail = exc.stderr.decode(errors="replace").strip()
        raise AudioError(f"ffmpeg failed to split {wav}: {detail}") from exc
    chunks = sorted(chunk_dir.glob("chunk_*.wav"))
    return [(index * window_seconds, chunk) for index, chunk in enumerate(chunks)]
