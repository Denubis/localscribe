"""Queue acceptance with real lightweight child processes, without model imports."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from localscribe import jobs


@pytest.fixture
def child_runner(monkeypatch, tmp_path):
    script = tmp_path / "child.py"
    script.write_text(
        "import os, sys, time\n"
        "from pathlib import Path\n"
        "path = Path(sys.argv[1])\n"
        "events = path.parent / 'events'\n"
        "with events.open('a') as log: log.write(f'start {path.name} {os.getpid()}\\n')\n"
        "print(f'processing {path.name}', flush=True)\n"
        "if path.stem == 'blocked':\n"
        "    print('ready', flush=True)\n"
        "    while True: time.sleep(1)\n"
        "if path.stem == 'bad':\n"
        "    print('CUDA out of memory', file=sys.stderr, flush=True)\n"
        "    sys.exit(1)\n"
        "path.with_suffix('.md').write_text('# transcript')\n"
        "path.with_suffix('.json').write_text('{}')\n"
        "with events.open('a') as log: log.write(f'end {path.name} {os.getpid()}\\n')\n"
    )
    popen = subprocess.Popen
    children = []

    def launch(command, **kwargs):
        assert command[:3] == [sys.executable, "-m", "localscribe.jobs"]
        assert kwargs["stdin"] == subprocess.DEVNULL
        assert kwargs["start_new_session"] is True
        proc = popen([sys.executable, str(script), command[3]], **kwargs)
        children.append(proc)
        return proc

    monkeypatch.setattr(jobs.subprocess, "Popen", launch)
    yield children
    for proc in children:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def recording(directory: Path, name: str) -> Path:
    path = directory / f"{name}.flac"
    path.write_bytes(b"original FLAC")
    return path


def test_jobs_run_serially_in_fresh_processes_and_finish_drains(child_runner, tmp_path):
    messages = []
    queue = jobs.TranscriptionQueue(echo=messages.append)
    first, second = recording(tmp_path, "first"), recording(tmp_path, "second")
    queue.submit(first)
    queue.submit(second)

    assert queue.finish() == 0

    events = [line.split() for line in (tmp_path / "events").read_text().splitlines()]
    assert [(kind, name) for kind, name, _ in events] == [
        ("start", "first.flac"), ("end", "first.flac"),
        ("start", "second.flac"), ("end", "second.flac"),
    ]
    assert events[0][2] != events[2][2]
    assert all(path.with_suffix(".md").exists() for path in (first, second))
    assert any("waiting" in message.lower() for message in messages)
    assert any("processing first.flac" in message for message in messages)
    assert queue.finish() == 0
    with pytest.raises(RuntimeError, match="closed"):
        queue.submit(first)


def test_failure_keeps_audio_reports_original_error_and_continues(child_runner, tmp_path):
    messages = []
    queue = jobs.TranscriptionQueue(echo=messages.append)
    bad, good = recording(tmp_path, "bad"), recording(tmp_path, "good")
    queue.submit(bad)
    queue.submit(good)

    assert queue.finish() == 1

    assert bad.read_bytes() == good.read_bytes() == b"original FLAC"
    assert good.with_suffix(".md").exists()
    assert not bad.with_suffix(".md").exists()
    assert any("CUDA out of memory" in message for message in messages)
    assert any(f"scribe {bad}" in message for message in messages)


def test_process_start_failure_does_not_strand_next_job(child_runner, monkeypatch, tmp_path):
    launch = jobs.subprocess.Popen
    messages = []
    first = True

    def fail_once(*args, **kwargs):
        nonlocal first
        if first:
            first = False
            raise OSError("process resources unavailable")
        return launch(*args, **kwargs)

    monkeypatch.setattr(jobs.subprocess, "Popen", fail_once)
    queue = jobs.TranscriptionQueue(echo=messages.append)
    bad, good = recording(tmp_path, "bad"), recording(tmp_path, "good")
    queue.submit(bad)
    queue.submit(good)

    assert queue.finish() == 1
    assert good.with_suffix(".md").exists()
    assert bad.read_bytes() == b"original FLAC"
    assert any("process resources unavailable" in message for message in messages)


def test_cancel_reaps_active_process_and_retains_waiting_audio(child_runner, tmp_path):
    ready = threading.Event()
    messages = []

    def echo(message):
        messages.append(message)
        if message == "ready":
            ready.set()

    queue = jobs.TranscriptionQueue(echo=echo)
    blocked, waiting = recording(tmp_path, "blocked"), recording(tmp_path, "waiting")
    try:
        queue.submit(blocked)
        queue.submit(waiting)
        assert ready.wait(timeout=5), messages
        child_pid = child_runner[0].pid
        queue.cancel()
        assert queue.finish() == 2
        assert child_runner[0].poll() is not None
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        assert len(child_runner) == 1
        assert blocked.read_bytes() == waiting.read_bytes() == b"original FLAC"
        assert any(f"scribe {waiting}" in message for message in messages)
    finally:
        queue.cancel()


def test_worker_sets_runtime_and_holds_gpu_lock(monkeypatch, tmp_path):
    from contextlib import contextmanager

    seen = []
    path = recording(tmp_path, "meeting")
    monkeypatch.setattr("localscribe.runtime.configure_cache", lambda: seen.append("cache"))
    monkeypatch.setattr("localscribe.runtime.quiet_third_party", lambda: seen.append("quiet"))
    monkeypatch.setattr("localscribe.runtime.require_gpu_extra", lambda: seen.append("extra"))

    @contextmanager
    def lock(**kwargs):
        seen.append("lock")
        yield
        seen.append("unlock")

    monkeypatch.setattr("localscribe.watch.gpu_lock", lock)
    monkeypatch.setattr("localscribe.pipeline.transcribe_recording",
                        lambda path, **kwargs: seen.append(path))

    assert jobs.run_worker(path) == 0
    assert seen == ["cache", "quiet", "extra", "lock", path, "unlock"]
