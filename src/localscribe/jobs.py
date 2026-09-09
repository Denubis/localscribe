# pattern: Imperative Shell
"""Serial transcription jobs, with a fresh model process for each finished recording."""

from __future__ import annotations

import os
import queue
import shlex
import signal
import subprocess
import sys
import threading
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path


class TranscriptionQueue:
    """Transcribe submitted files in FIFO order while the caller keeps recording.

    ``finish`` closes submission and drains the jobs. ``cancel`` is reserved for
    exceptional shutdown: it reaps the active process and preserves queued audio.
    """

    def __init__(self, echo: Callable[[str], None] = print) -> None:
        self._echo = echo
        self._jobs: queue.Queue[Path | None] = queue.Queue()
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._closed = False
        self._cancelled = False
        self._pending = 0
        self._failures = 0
        self._active: subprocess.Popen[str] | None = None
        self._thread = threading.Thread(target=self._work, name="transcription-queue", daemon=True)
        self._thread.start()

    def submit(self, path: Path) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("transcription queue is closed")
            self._pending += 1
            self._jobs.put(Path(path).resolve())

    def finish(self) -> int:
        """Wait for all submitted files, returning the number that failed."""
        with self._lock:
            if not self._closed:
                self._closed = True
                self._jobs.put(None)
            pending = self._pending
        if pending:
            self._echo(f"waiting for {pending} transcription job(s) to finish ...")
        # Event.wait stays interruptible without marking a still-running Thread
        # stopped when Ctrl-C interrupts Thread.join on affected Python versions.
        self._done.wait()
        self._thread.join()
        return self._failures

    def cancel(self) -> None:
        """Stop and reap the active child; report unprocessed files as retryable."""
        with self._lock:
            self._cancelled = True
            if not self._closed:
                self._closed = True
                self._jobs.put(None)
            active = self._active
        if active is not None:
            self._stop_process(active)
        self._done.wait()
        self._thread.join()

    @staticmethod
    def _stop_process(proc: subprocess.Popen[str]) -> None:
        with suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        finally:
            # A loader descendant may keep stdout open after the worker exits.
            # Reap the whole owned session even if its leader already stopped.
            with suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()

    def _retry(self, path: Path, reason: str) -> None:
        self._echo(f"transcription failed for {path.name}: {reason}")
        self._echo(f"the recording is safe at {path}; retry: {shlex.join(['scribe', str(path)])}")

    def _work(self) -> None:
        try:
            while (path := self._jobs.get()) is not None:
                try:
                    self._run_job(path)
                except Exception as exc:  # boundary: a failed child must not stop later jobs
                    self._failures += 1
                    self._retry(path, str(exc))
                finally:
                    with self._lock:
                        self._pending -= 1
        finally:
            self._done.set()

    def _run_job(self, path: Path) -> None:
        # Launch and cancellation share a lock so cancel cannot miss a child
        # created between checking the cancelled flag and publishing its handle.
        with self._lock:
            if self._cancelled:
                raise RuntimeError("cancelled before transcription")
            proc = subprocess.Popen(
                [sys.executable, "-m", "localscribe.jobs", str(path)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                bufsize=1,
                start_new_session=True,
            )
            self._active = proc
        try:
            assert proc.stdout is not None
            with proc.stdout:
                for line in proc.stdout:
                    self._echo(line.rstrip("\r\n"))
            code = proc.wait()
            if code:
                raise RuntimeError(f"worker exited with status {code}")
        finally:
            if proc.poll() is None:
                self._stop_process(proc)
            with self._lock:
                self._active = None


def run_worker(recording: Path) -> int:
    """Run one pipeline under the shared GPU lock; return a process exit code."""
    from .runtime import configure_cache, quiet_third_party, require_gpu_extra

    def echo(message: str) -> None:
        print(message, flush=True)

    try:
        configure_cache()
        quiet_third_party()
        require_gpu_extra()
        from .pipeline import transcribe_recording
        from .watch import gpu_lock

        with gpu_lock(waiting=lambda: echo("waiting for another transcription to finish ...")):
            transcribe_recording(recording, echo=echo)
    except Exception as exc:  # boundary: preserve model/import/I/O error text for the parent
        echo(f"transcription failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m localscribe.jobs RECORDING", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(run_worker(Path(sys.argv[1])))
