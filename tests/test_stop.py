"""Tests for stopping a capture without destroying it.

These run real ffmpeg against a lavfi tone, because the bug they pin was entirely
about how a real ffmpeg reacts to how it is asked to stop. No microphone is
involved.

Measured 2026-08-01, ffmpeg 6.1.1 here and 8.x on the laptop that failed:

| how it is stopped                | result                        |
|----------------------------------|-------------------------------|
| `q` on a pipe, no `-nostdin`     | exit 0, valid FLAC            |
| `q` on a pipe, with `-nostdin`   | ignored; killed; **0 bytes**  |

`-nostdin` makes ffmpeg refuse the only graceful stop it documents, leaving
signals as the only route. That would be survivable if it got exactly one, but
Ctrl-C in a terminal goes to the whole foreground process group, so ffmpeg got
the terminal's signal *and* the one the code sent. On ffmpeg 8 a signal arriving
during shutdown aborts the trailer write, and the recording is lost.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

from localscribe.capture import (
    build_capture_command,
    capture,
    ensure_has_audio,
    spawn_capture,
    stop_gracefully,
)
from localscribe.errors import CaptureError


def _tone_capture(target: Path, *, nostdin: bool = False) -> subprocess.Popen[str]:
    """An open-ended realtime capture, the shape `capture()` runs, from a tone."""
    argv = ["ffmpeg", "-hide_banner", "-v", "error"]
    if nostdin:
        argv.append("-nostdin")
    argv += [
        "-re", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
        "-c:a", "flac", str(target),
    ]
    return spawn_capture(argv)


class TestCaptureCommand:
    def test_nostdin_is_absent_because_it_disables_the_graceful_stop(self) -> None:
        # With -nostdin, ffmpeg ignores `q` and can only be signalled, which is
        # what destroyed the 2026-08-01 recording.
        assert "-nostdin" not in build_capture_command("MIC", "MON", Path("/out/m.flac"))

    @pytest.mark.parametrize("remote", [False, True])
    def test_existing_output_is_refused_without_waiting_for_stdin(
        self, tmp_path: Path, remote: bool
    ) -> None:
        target = tmp_path / "existing.flac"
        original = b"previous recording must survive"
        target.write_bytes(original)
        command = build_capture_command("MIC", "MON" if remote else None, target)
        # Keep the generated graph/output options, substituting synthetic audio
        # only at the hardware input boundary. stdin stays open as in capture().
        command = [
            "lavfi" if arg == "pulse" else
            "sine=frequency=440:duration=0.1" if arg in {"MIC", "MON"} else arg
            for arg in command
        ]
        proc = spawn_capture(command)
        try:
            proc.wait(timeout=3)
            # ffmpeg 8 can report refusal with exit 0; the observable contract
            # here is prompt-free refusal and preservation, not its exit status.
            assert proc.stderr is not None
            assert "already exists" in proc.stderr.read()
            assert target.read_bytes() == original
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    @pytest.mark.parametrize("remote", [False, True])
    def test_output_created_during_startup_is_not_reported_as_a_new_recording(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, remote: bool
    ) -> None:
        target = tmp_path / "raced.flac"
        previous_audio: list[bytes] = []

        def spawn_after_output_appears(command: list[str]) -> subprocess.Popen[str]:
            # Another writer creates valid audio after capture's preflight check.
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-v", "error", "-nostdin", "-f", "lavfi",
                 "-i", "sine=duration=0.1", str(target)],
                check=True,
            )
            previous_audio.append(target.read_bytes())
            command = [
                "lavfi" if arg == "pulse" else
                "sine=duration=0.1" if arg in {"MIC", "MON.monitor"} else arg
                for arg in command
            ]
            return spawn_capture(command)

        monkeypatch.setattr("localscribe.capture.spawn_capture", spawn_after_output_appears)
        monkeypatch.setattr("localscribe.capture._resolve_default", lambda node: "MON")
        with pytest.raises(CaptureError, match="already exists"):
            capture(target, mic="MIC", remote=remote)
        assert target.read_bytes() == previous_audio[0]


class TestSpawnCapture:
    def test_the_child_is_isolated_from_our_process_group(self) -> None:
        # Otherwise a terminal Ctrl-C reaches ffmpeg directly, as a second signal
        # alongside the one we send deliberately.
        proc = spawn_capture(["sleep", "30"])
        try:
            assert os.getpgid(proc.pid) != os.getpgid(0)
        finally:
            proc.kill()
            proc.wait()

    def test_stdin_is_a_pipe_so_q_can_be_delivered(self) -> None:
        proc = spawn_capture(["sleep", "30"])
        try:
            assert proc.stdin is not None
        finally:
            proc.kill()
            proc.wait()


class TestStopGracefully:
    def test_a_stopped_capture_is_a_playable_recording(self, tmp_path: Path) -> None:
        target = tmp_path / "tone.flac"
        proc = _tone_capture(target)
        time.sleep(1.5)
        stop_gracefully(proc)

        assert proc.returncode == 0, "a graceful stop should be a clean exit, not a signal death"
        ensure_has_audio(target)

    def test_the_recording_keeps_the_audio_captured_before_the_stop(
        self, tmp_path: Path
    ) -> None:
        # A trailer that aborts leaves 0 bytes, so a positive duration is the
        # thing that distinguishes a real stop from the failure being fixed.
        target = tmp_path / "tone.flac"
        proc = _tone_capture(target)
        time.sleep(1.5)
        stop_gracefully(proc)

        probe = subprocess.run(
            ["ffprobe", "-hide_banner", "-v", "error", "-show_entries",
             "format=duration", "-of", "csv=p=0", str(target)],
            capture_output=True, text=True, check=True,
        )
        assert float(probe.stdout.strip()) > 0.5

    def test_a_process_that_ignores_q_is_still_stopped_rather_than_hanging(
        self, tmp_path: Path
    ) -> None:
        # The escalation path. -nostdin deliberately reproduces an ffmpeg that
        # will not accept `q`; the stop must still return.
        target = tmp_path / "stubborn.flac"
        proc = _tone_capture(target, nostdin=True)
        time.sleep(1.0)

        started = time.monotonic()
        stop_gracefully(proc, timeout=2.0)
        elapsed = time.monotonic() - started

        assert proc.returncode is not None, "the process must not be left running"
        assert elapsed < 20.0, "escalation must not hang"

    def test_stopping_an_already_finished_process_is_harmless(self, tmp_path: Path) -> None:
        target = tmp_path / "short.flac"
        proc = spawn_capture([
            "ffmpeg", "-hide_banner", "-v", "error",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=0.2",
            "-c:a", "flac", str(target),
        ])
        proc.wait(timeout=30)
        stop_gracefully(proc)  # must not raise
        ensure_has_audio(target)


@pytest.mark.parametrize("nostdin", [False, True])
def test_the_measurement_this_fix_rests_on(tmp_path: Path, nostdin: bool) -> None:
    """Pin the finding itself: -nostdin is what breaks the graceful stop.

    Without this, a future edit could reintroduce -nostdin and every other test
    here would still pass, because they all go through spawn_capture.
    """
    target = tmp_path / "measured.flac"
    proc = _tone_capture(target, nostdin=nostdin)
    time.sleep(1.2)
    assert proc.stdin is not None
    try:
        proc.stdin.write("q")
        proc.stdin.flush()
    except (BrokenPipeError, ValueError):
        pass

    try:
        proc.wait(timeout=5)
        accepted_q = True
    except subprocess.TimeoutExpired:
        accepted_q = False
        proc.kill()
        proc.wait()

    assert accepted_q is (not nostdin), (
        "ffmpeg should honour `q` on a pipe, and ignore it under -nostdin"
    )
