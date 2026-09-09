# pattern: Imperative Shell
"""Terminal controls for uninterrupted recording and queued local transcription."""

from __future__ import annotations

import os
import select
import signal
import sys
import threading
from collections.abc import Callable
from pathlib import Path

from .errors import LocalscribeError
from .runtime import require_gpu_extra


def _read_enter(
    rollover: threading.Event, stopped: threading.Event, echo: Callable[[str], None]
) -> None:
    """Read terminal lines without leaving a blocked reader after capture ends."""
    try:
        descriptor = sys.stdin.fileno()
        while not stopped.is_set():
            ready, _, _ = select.select([descriptor], [], [], 0.1)
            if not ready:
                continue
            data = os.read(descriptor, 4096)
            if not data:
                return
            if b"\n" in data or b"\r" in data:
                rollover.set()
    except (OSError, ValueError) as exc:
        echo(f"terminal input unavailable: {exc}; Ctrl-C still stops recording")


def record_interactively(
    target: Path,
    *,
    mic: str,
    mono: bool,
    remote: bool,
    no_transcribe: bool,
    echo: Callable[[str], None] = print,
) -> int:
    """Capture numbered parts; Enter rolls over, SIGINT stops and drains the queue."""
    from .jobs import TranscriptionQueue
    from .rolling import capture_parts

    jobs = None
    missing_stack = False
    if not no_transcribe:
        try:
            require_gpu_extra()
        except LocalscribeError as exc:
            missing_stack = True
            echo(f"recording audio only: {exc}")
        else:
            jobs = TranscriptionQueue(echo=echo)

    def completed(path: Path) -> None:
        echo(f"wrote {path}")
        if jobs is not None:
            jobs.submit(path)
        elif missing_stack:
            echo(f"after installing the model stack, retry: scribe {path}")

    rollover = threading.Event()
    stopped = threading.Event()
    previous = signal.getsignal(signal.SIGINT)
    reader = threading.Thread(target=_read_enter, args=(rollover, stopped, echo), daemon=True)
    reader_started = False
    try:
        try:
            signal.signal(signal.SIGINT, lambda signum, frame: stopped.set())
            reader.start()
            reader_started = True
            capture_parts(
                target.resolve(), mic=mic, mono=mono, remote=remote,
                rollover=rollover, stop=stopped, on_complete=completed, echo=echo,
            )
        finally:
            stopped.set()
            if reader_started:
                reader.join(timeout=1)
            signal.signal(signal.SIGINT, previous)

        if jobs is not None:
            return 1 if jobs.finish() else 0
        return 3 if missing_stack else 0
    except KeyboardInterrupt:
        if jobs is not None:
            jobs.cancel()
        echo("stopped waiting for transcription; recordings are kept for retry with scribe")
        return 130
    except BaseException:
        if jobs is not None:
            jobs.cancel()
        raise
