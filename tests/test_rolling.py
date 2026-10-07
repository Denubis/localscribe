"""Real ffmpeg rollovers preserve every interleaved PCM frame."""

from __future__ import annotations

import struct
import subprocess
import threading
from pathlib import Path

import pytest

from localscribe import rolling
from localscribe.errors import CaptureError


def _source(monkeypatch, tmp_path, *, channels=1, frames=48000, realtime=False):
    pcm = b"".join(
        struct.pack("<h", (index * 101 + channel * 997) % 65536 - 32768)
        for index in range(frames)
        for channel in range(channels)
    )
    raw = tmp_path / "source.raw"
    raw.write_bytes(pcm)
    spawn = rolling._spawn_source
    monkeypatch.setattr(
        rolling,
        "_spawn_source",
        lambda command: spawn([
            "ffmpeg", "-hide_banner", "-v", "error", *(["-re"] if realtime else []),
            "-f", "s16le",
            "-ar", "48000", "-ac", str(channels), "-i", str(raw),
            "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1",
        ]),
    )
    monkeypatch.setattr(rolling, "resolve_mic", lambda explicit: "MIC")
    monkeypatch.setattr(rolling, "_resolve_default", lambda node: "MON")
    return pcm


def _decode(path: Path) -> bytes:
    return subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", str(path), "-c:a", "pcm_s16le",
        "-f", "s16le", "pipe:1",
    ])


@pytest.mark.parametrize("remote", [False, True])
def test_rollovers_preserve_all_frames_and_finalize_before_callback(
    monkeypatch, tmp_path, remote
) -> None:
    channels = 2 if remote else 1
    expected = _source(monkeypatch, tmp_path, channels=channels)
    rollover, stop = threading.Event(), threading.Event()
    read = rolling._read_pcm
    consumed = 0

    def awkward_reads(fd, size):
        nonlocal consumed
        block = read(fd, 5003)  # Deliberately split samples and stereo frames.
        consumed += len(block)
        if consumed > 18000:
            rollover.set()
            consumed = 0
        return block

    monkeypatch.setattr(rolling, "_read_pcm", awkward_reads)
    completed = []
    decoded = []

    def complete(path):
        completed.append(path)
        decoded.append(_decode(path))  # Must already have a readable FLAC trailer.

    paths = rolling.capture_parts(
        tmp_path / "meeting.flac", remote=remote, rollover=rollover, stop=stop,
        on_complete=complete, echo=lambda message: None,
    )

    assert len(paths) >= 3
    assert paths == completed
    assert b"".join(decoded) == expected
    assert all(decoded)
    assert paths[0].name == "meeting.flac"
    assert paths[1].name == "meeting-part002.flac"


def test_initial_collision_preserves_existing_recording(monkeypatch, tmp_path) -> None:
    _source(monkeypatch, tmp_path)
    output = tmp_path / "meeting.flac"
    output.write_bytes(b"existing research recording")

    with pytest.raises(CaptureError, match="already exists"):
        rolling.capture_parts(
            output, remote=False, rollover=threading.Event(), stop=threading.Event(),
            on_complete=lambda path: pytest.fail("must not transcribe old recording"),
        )

    assert output.read_bytes() == b"existing research recording"


def test_failed_source_is_not_submitted_for_transcription(monkeypatch, tmp_path) -> None:
    _source(monkeypatch, tmp_path)
    monkeypatch.setattr(
        rolling, "_spawn_source",
        lambda command: subprocess.Popen(
            ["ffmpeg", "-v", "error", "-f", "invalid-input-format", "-i", "missing"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        ),
    )

    with pytest.raises(CaptureError):
        rolling.capture_parts(
            tmp_path / "meeting.flac", remote=False,
            rollover=threading.Event(), stop=threading.Event(),
            on_complete=lambda path: pytest.fail("failed audio must not be submitted"),
            echo=lambda message: None,
        )


def test_callback_failure_preserves_finalized_audio(monkeypatch, tmp_path) -> None:
    expected = _source(monkeypatch, tmp_path)

    def fail(path):
        raise RuntimeError("queue unavailable")

    with pytest.raises(CaptureError, match="queue unavailable"):
        rolling.capture_parts(
            tmp_path / "meeting.flac", remote=False,
            rollover=threading.Event(), stop=threading.Event(), on_complete=fail,
            echo=lambda message: None,
        )

    assert _decode(tmp_path / "meeting.flac") == expected


def test_stop_drains_all_remaining_pcm_and_closes_source(monkeypatch, tmp_path) -> None:
    expected = _source(monkeypatch, tmp_path, realtime=True, frames=96000)
    received = bytearray()
    stop = threading.Event()
    read = rolling._read_pcm

    def read_and_stop(fd, size):
        block = read(fd, size)
        received.extend(block)
        if len(received) > 20000:
            stop.set()
        return block

    monkeypatch.setattr(rolling, "_read_pcm", read_and_stop)
    paths = rolling.capture_parts(
        tmp_path / "meeting.flac", remote=False,
        rollover=threading.Event(), stop=stop, on_complete=lambda path: None,
        echo=lambda message: None,
    )

    assert stop.is_set()
    assert 0 < len(received) < len(expected), "stop must end the realtime source early"
    assert _decode(paths[0]) == received
    assert expected.startswith(received)


def test_numbered_collision_is_skipped_without_overwriting(monkeypatch, tmp_path) -> None:
    expected = _source(monkeypatch, tmp_path)
    collision = tmp_path / "meeting-part002.flac"
    collision.write_bytes(b"earlier recording")
    rollover = threading.Event()
    read = rolling._read_pcm
    calls = 0

    def request_one_rollover(fd, size):
        nonlocal calls
        block = read(fd, 5003)
        calls += 1
        if calls == 3:
            rollover.set()
        return block

    monkeypatch.setattr(rolling, "_read_pcm", request_one_rollover)
    paths = rolling.capture_parts(
        tmp_path / "meeting.flac", remote=False,
        rollover=rollover, stop=threading.Event(), on_complete=lambda path: None,
        echo=lambda message: None,
    )

    assert [p.name for p in paths] == ["meeting.flac", "meeting-part003.flac"]
    assert collision.read_bytes() == b"earlier recording"
    assert b"".join(_decode(p) for p in paths) == expected


def test_rollover_before_first_frame_does_not_create_an_empty_part(monkeypatch, tmp_path):
    expected = _source(monkeypatch, tmp_path)
    rollover = threading.Event()
    rollover.set()

    paths = rolling.capture_parts(
        tmp_path / "meeting.flac", remote=False,
        rollover=rollover, stop=threading.Event(), on_complete=lambda path: None,
        echo=lambda message: None,
    )

    assert len(paths) == 1
    assert _decode(paths[0]) == expected


def test_failed_encoder_stops_children_and_never_queues_audio(monkeypatch, tmp_path) -> None:
    _source(monkeypatch, tmp_path)
    children = []
    spawn_source = rolling._spawn_source

    def track_source(command):
        child = spawn_source(command)
        children.append(child)
        return child

    def broken_encoder(path, channels):
        import sys
        import tempfile

        path.write_bytes(b"unfinished")
        diagnostics = tempfile.TemporaryFile()  # noqa: SIM115 - owned by finalizer
        child = subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.stdin.buffer.read(10); sys.exit(7)"],
            stdin=subprocess.PIPE, stderr=diagnostics, bufsize=0, start_new_session=True,
        )
        children.append(child)
        return rolling._Part(path, child, diagnostics)

    monkeypatch.setattr(rolling, "_spawn_source", track_source)
    monkeypatch.setattr(rolling, "_new_part", broken_encoder)

    with pytest.raises(CaptureError, match="encoder"):
        rolling.capture_parts(
            tmp_path / "meeting.flac", remote=False,
            rollover=threading.Event(), stop=threading.Event(),
            on_complete=lambda path: pytest.fail("bad encoder output must not be queued"),
            echo=lambda message: None,
        )

    assert children and all(child.poll() is not None for child in children)
    assert (tmp_path / "meeting.flac").read_bytes() == b"unfinished"


@pytest.mark.parametrize("retry", [False, True])
def test_refused_rollover_keeps_every_frame_and_allows_later_retry(
    monkeypatch, tmp_path, retry
) -> None:
    import errno

    expected = _source(monkeypatch, tmp_path)
    rollover = threading.Event()
    new_part = rolling._new_part
    read = rolling._read_pcm
    attempts = 0
    reads = 0
    messages = []

    def refuse_once(path, channels):
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise OSError(errno.ENAMETOOLONG, "File name too long", str(path))
        return new_part(path, channels)

    def request_rollovers(fd, size):
        nonlocal reads
        block = read(fd, 5003)
        reads += 1
        if reads == 3 or (retry and reads == 8):
            rollover.set()
        return block

    monkeypatch.setattr(rolling, "_new_part", refuse_once)
    monkeypatch.setattr(rolling, "_read_pcm", request_rollovers)
    paths = rolling.capture_parts(
        tmp_path / "meeting.flac", remote=False, rollover=rollover,
        stop=threading.Event(), on_complete=lambda path: None, echo=messages.append,
    )

    assert len(paths) == (2 if retry else 1)
    assert b"".join(_decode(path) for path in paths) == expected
    assert any("rollover" in message and "continu" in message for message in messages)
    assert attempts == (3 if retry else 2)
