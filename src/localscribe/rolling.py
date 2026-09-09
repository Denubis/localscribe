# pattern: Imperative Shell
"""Continuous audio capture with independently finalized, numbered FLAC parts.

The source stays open through a rollover. Its interleaved PCM frames go to exactly
one encoder each; closing an encoder's stdin finalizes that part while the next
encoder receives audio. A FIFO finalizer validates files before submitting them.
"""

from __future__ import annotations

import os
import queue
import selectors
import signal
import subprocess
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from .capture import (
    SilenceTracker,
    _resolve_default,
    build_capture_command,
    ensure_has_audio,
    resolve_mic,
)
from .errors import CaptureError

_RATE = 48000
_READ_SIZE = 65536
_IO_TIMEOUT = 15.0


@dataclass
class _Part:
    path: Path
    process: subprocess.Popen[bytes]
    diagnostics: BinaryIO
    frames: int = 0


def _spawn_source(command: list[str]) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        bufsize=0, start_new_session=True,
    )


def _read_pcm(fd: int, size: int) -> bytes:
    return os.read(fd, size)


def _new_part(path: Path, channels: int) -> _Part:
    # The inherited descriptor owns this exact inode even if a competing process
    # replaces the filename. FLAC can seek back to write its duration/checksum.
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o644)
    # Ownership passes to _Part, whose finalizer closes the diagnostics file.
    diagnostics = tempfile.TemporaryFile()  # noqa: SIM115
    try:
        process = subprocess.Popen(
            [
                "ffmpeg", "-hide_banner", "-v", "error", "-nostdin", "-y",
                "-f", "s16le", "-ar", str(_RATE), "-ac", str(channels),
                "-i", "pipe:0", "-c:a", "flac", "-f", "flac", f"/proc/self/fd/{fd}",
            ],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=diagnostics,
            pass_fds=(fd,), bufsize=0, start_new_session=True,
        )
    except BaseException:
        diagnostics.close()
        raise
    finally:
        os.close(fd)
    assert process.stdin is not None
    os.set_blocking(process.stdin.fileno(), False)
    return _Part(path, process, diagnostics)


def _write_pcm(part: _Part, block: bytes) -> None:
    assert part.process.stdin is not None
    fd = part.process.stdin.fileno()
    remaining = memoryview(block)
    deadline = time.monotonic() + _IO_TIMEOUT
    with selectors.DefaultSelector() as ready:
        ready.register(fd, selectors.EVENT_WRITE)
        while remaining:
            if part.process.poll() is not None:
                raise CaptureError(f"FLAC encoder stopped while recording {part.path}")
            if time.monotonic() >= deadline:
                raise CaptureError(f"FLAC encoder stopped accepting audio for {part.path}")
            if not ready.select(timeout=0.1):
                continue
            try:
                count = os.write(fd, remaining)
            except BlockingIOError:
                continue
            except BrokenPipeError as exc:
                raise CaptureError(f"FLAC encoder failed while recording {part.path}") from exc
            remaining = remaining[count:]
            deadline = time.monotonic() + _IO_TIMEOUT


def _finish_part(part: _Part) -> None:
    """EOF is the encoder's stop request: a literal q would become PCM audio."""
    assert part.process.stdin is not None
    part.process.stdin.close()
    try:
        try:
            part.process.wait(timeout=_IO_TIMEOUT)
        except subprocess.TimeoutExpired:
            part.process.send_signal(signal.SIGINT)
            try:
                part.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                part.process.kill()
                part.process.wait()
            raise CaptureError(f"FLAC encoder did not finalize promptly: {part.path}") from None
        part.diagnostics.seek(0)
        detail = part.diagnostics.read().decode(errors="replace").strip()
        if part.process.returncode:
            raise CaptureError(f"FLAC encoder failed for {part.path}: {detail}")
        ensure_has_audio(part.path)
    finally:
        part.diagnostics.close()


def _request_source_stop(source: subprocess.Popen[bytes]) -> None:
    if source.stdin is not None and not source.stdin.closed:
        with suppress(OSError):
            source.stdin.write(b"q")
        source.stdin.close()


def capture_parts(
    output: Path,
    *,
    mic: str | None = None,
    mono: bool = False,
    remote: bool = True,
    rollover: threading.Event,
    stop: threading.Event,
    on_complete: Callable[[Path], None],
    echo: Callable[[str], None] = print,
) -> list[Path]:
    """Capture until ``stop``; ``rollover`` closes one part and starts the next.

    Call on the controlling thread with a SIGINT handler that sets ``stop``.
    ``on_complete`` runs on a single finalizer thread, in recording order, after
    the file is validated. It should enqueue background work and return promptly.
    Completed parts are returned in order. A refused rollover warns and keeps
    recording into the current part; capture/finalization failures raise
    ``CaptureError`` after children stop, preserving audio written to disk.
    """
    mic_node = resolve_mic(mic)
    monitor = _resolve_default("@DEFAULT_AUDIO_SINK@") + ".monitor" if remote else None
    channels = 2 if remote and not mono else 1
    frame_size = channels * 2
    command = build_capture_command(mic_node, monitor, output, mono=mono)
    command = command[:-3] + ["-c:a", "pcm_s16le", "-ar", str(_RATE), "-f", "s16le", "pipe:1"]
    try:
        active = _new_part(output, channels)
    except FileExistsError as exc:
        raise CaptureError(f"recording output already exists: {output}") from exc
    except OSError as exc:
        raise CaptureError(f"could not start FLAC recording {output}: {exc}") from exc

    pending: queue.Queue[_Part | None] = queue.Queue()
    completed: list[Path] = []
    errors: list[str] = []

    def finalize() -> None:
        while (part := pending.get()) is not None:
            try:
                _finish_part(part)
                completed.append(part.path)
                on_complete(part.path)
            except Exception as exc:  # queue failures must not kill live capture
                message = f"could not finish/queue {part.path}: {exc}"
                errors.append(message)
                echo(message)

    finalizer = threading.Thread(target=finalize, name="scribe-finalize")
    finalizer.start()
    source: subprocess.Popen[bytes] | None = None
    reader: threading.Thread | None = None
    recent: deque[str] = deque(maxlen=20)
    next_number = 2
    try:
        source = _spawn_source(command)
        assert source.stdout is not None and source.stderr is not None
        tracker = SilenceTracker(mic_node)

        def diagnostics() -> None:
            assert source is not None and source.stderr is not None
            for raw in iter(source.stderr.readline, b""):
                line = raw.decode(errors="replace").rstrip()
                recent.append(line)
                if (warning := tracker.observe(line)) is not None:
                    echo(warning)

        reader = threading.Thread(target=diagnostics, name="scribe-source-stderr")
        reader.start()
        carry = b""
        stopping_at: float | None = None
        echo(f"recording -> {output}")
        with selectors.DefaultSelector() as ready:
            ready.register(source.stdout, selectors.EVENT_READ)
            while True:
                if stop.is_set() and stopping_at is None:
                    _request_source_stop(source)
                    stopping_at = time.monotonic()
                if stopping_at is not None and time.monotonic() - stopping_at > _IO_TIMEOUT:
                    raise CaptureError("audio source did not stop promptly")
                if not ready.select(timeout=0.1):
                    continue
                block = _read_pcm(source.stdout.fileno(), _READ_SIZE)
                if not block:
                    break
                carry += block
                whole = len(carry) // frame_size * frame_size
                block, carry = carry[:whole], carry[whole:]
                if not block:
                    continue
                if rollover.is_set():
                    rollover.clear()
                    if active.frames and stopping_at is None:
                        try:
                            while True:
                                candidate = output.with_name(
                                    f"{output.stem}-part{next_number:03d}{output.suffix}"
                                )
                                next_number += 1
                                try:
                                    replacement = _new_part(candidate, channels)
                                except FileExistsError:
                                    continue
                                break
                        except OSError as exc:
                            # The old encoder still owns a valid output. Keep
                            # this pending block and all following audio there.
                            echo(f"rollover failed: {exc}; continuing in {active.path}")
                        else:
                            pending.put(active)
                            active = replacement
                            echo(f"recording -> {candidate}")
                _write_pcm(active, block)
                active.frames += len(block) // frame_size
        if carry:
            raise CaptureError("audio source ended with an incomplete PCM frame")
        source.wait(timeout=5)
        if source.returncode:
            raise CaptureError("audio source exited unexpectedly:\n" + "\n".join(recent))
    except Exception as exc:  # always finalize recoverable audio before reporting
        errors.append(str(exc))
    finally:
        if source is not None:
            if source.poll() is None:
                _request_source_stop(source)
                # The normal stop drained every frame above. Error cleanup must
                # also drain the pipe so a blocked source cannot hang shutdown.
                try:
                    assert source.stdout is not None
                    source.stdout.close()
                    source.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    source.kill()
                    source.wait()
            for pipe in (source.stdin, source.stdout):
                if pipe is not None:
                    pipe.close()
            if reader is not None:
                reader.join()
            if source.stderr is not None:
                source.stderr.close()
        pending.put(active)
        pending.put(None)
        finalizer.join()
    if errors:
        raise CaptureError("\n".join(errors))
    return completed
