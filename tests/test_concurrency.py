"""Cross-process tests for the two assumptions the design rests on.

`test_watch.py` exercises claiming in one process, which proves the bookkeeping
and nothing about the race. These use real separate processes and real rsync,
because what is being checked is that the operating system behaves the way the
design assumes:

- ``os.rename`` is atomic, so exactly one of several watchers can claim a
  recording however closely they collide.
- ``flock`` excludes across processes, so two watchers started in two project
  folders cannot put two model pipelines on one 24 GB card.
- rsync's in-flight file is invisible to ``is_ready``, so a watcher never opens a
  transfer that is still arriving.

No GPU and no models: every model call is stubbed, since none of these properties
involve one. Each test carries a positive control, because a test that only ever
observes "nothing was claimed" or "nothing was ready" would pass just as happily
if the mechanism were broken in the other direction.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from localscribe.inbox import CLAIMED_DIR, PARTIAL_DIR, build_rsync_command, select_next
from localscribe.watch import claim, gpu_lock

WORKERS = 8


def _claim_at(args: tuple[str, str, str, float]) -> str | None:
    """Wait for a shared deadline, then claim. Returns the claimed name, or None."""
    inbox, name, token, deadline = args
    while time.time() < deadline:
        time.sleep(0.001)
    claimed = claim(Path(inbox), name, token)
    return None if claimed is None else claimed.name


def _hold_lock(args: tuple[str, float]) -> tuple[float, float]:
    """Take the GPU lock at a shared deadline, hold it briefly, report the window."""
    lock_path, deadline = args
    while time.time() < deadline:
        time.sleep(0.001)
    with gpu_lock(Path(lock_path)):
        entered = time.time()
        time.sleep(0.15)
        return entered, time.time()


class TestClaimRaceAcrossProcesses:
    def test_exactly_one_process_wins_a_contested_recording(self, tmp_path: Path) -> None:
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        (inbox / "meeting-1.flac").write_bytes(b"audio")

        deadline = time.time() + 0.5
        work = [(str(inbox), "meeting-1.flac", f"tok{i:04d}", deadline) for i in range(WORKERS)]
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            results = list(pool.map(_claim_at, work))

        winners = [r for r in results if r is not None]
        assert len(winners) == 1, f"{len(winners)} processes claimed the same recording"
        assert len(list((inbox / CLAIMED_DIR).iterdir())) == 1

    def test_every_recording_is_claimed_when_there_are_enough_to_go_round(
        self, tmp_path: Path
    ) -> None:
        # The positive control. Without it the test above would pass if `claim`
        # simply always returned None.
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        for i in range(WORKERS):
            (inbox / f"meeting-{i}.flac").write_bytes(b"audio")

        deadline = time.time() + 0.5
        work = [
            (str(inbox), f"meeting-{i}.flac", f"tok{i:04d}", deadline) for i in range(WORKERS)
        ]
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            results = list(pool.map(_claim_at, work))

        assert [r for r in results if r is not None] != []
        assert len(list((inbox / CLAIMED_DIR).iterdir())) == WORKERS


class TestGpuLockAcrossProcesses:
    def test_no_two_processes_hold_the_lock_at_the_same_time(self, tmp_path: Path) -> None:
        lock = tmp_path / "gpu.lock"
        deadline = time.time() + 0.5
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            windows = sorted(pool.map(_hold_lock, [(str(lock), deadline)] * WORKERS))

        assert len(windows) == WORKERS
        for (_, earlier_end), (later_start, _) in zip(windows, windows[1:], strict=False):
            assert later_start >= earlier_end, "two processes held the GPU lock at once"

    def test_the_lock_serialises_rather_than_dropping_work(self, tmp_path: Path) -> None:
        # Positive control: blocking, not failing. A lock that raised instead of
        # waiting would make the test above trivially true and lose recordings.
        lock = tmp_path / "gpu.lock"
        deadline = time.time() + 0.3
        started = time.time()
        with ProcessPoolExecutor(max_workers=4) as pool:
            windows = list(pool.map(_hold_lock, [(str(lock), deadline)] * 4))

        assert len(windows) == 4, "every process must get its turn"
        assert time.time() - started >= 4 * 0.15, "holds must have been serial, not concurrent"


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync not installed")
class TestTransferInFlight:
    def test_a_watcher_ignores_a_transfer_that_is_still_arriving(self, tmp_path: Path) -> None:
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        source = tmp_path / "big.flac"
        source.write_bytes(os.urandom(6 * 1024 * 1024))

        # A local rsync takes the same in-flight/rename path as one over ssh.
        argv = [a for a in build_rsync_command(source, "", str(inbox)) if a != "-s"]
        argv[-1] = f"{inbox}/"
        argv.insert(1, "--bwlimit=2000")  # ~2 MB/s, so the transfer lasts seconds
        proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        try:
            saw_bytes_arriving = False
            for _ in range(60):
                time.sleep(0.05)
                entries = [(p.name, p.stat().st_mtime) for p in inbox.iterdir() if p.is_file()]
                partial = inbox / PARTIAL_DIR
                if any(p.stat().st_size > 0 for p in inbox.rglob("*") if p.is_file()):
                    saw_bytes_arriving = True
                # Whatever is on disk mid-transfer, it must never be selectable.
                assert select_next(entries, None) is None, (
                    f"a watcher would have taken an in-flight transfer: {entries} "
                    f"(partial dir present: {partial.exists()})"
                )
                if proc.poll() is not None:
                    break
            assert saw_bytes_arriving, "the transfer never started; the test proved nothing"
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()

    def test_a_killed_transfer_leaves_nothing_a_watcher_would_take(
        self, tmp_path: Path
    ) -> None:
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        source = tmp_path / "big.flac"
        source.write_bytes(os.urandom(6 * 1024 * 1024))

        argv = [a for a in build_rsync_command(source, "", str(inbox)) if a != "-s"]
        argv[-1] = f"{inbox}/"
        argv.insert(1, "--bwlimit=2000")
        proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.8)
        proc.kill()
        proc.wait()

        entries = [(p.name, p.stat().st_mtime) for p in inbox.iterdir() if p.is_file()]
        assert select_next(entries, None) is None

    def test_a_completed_transfer_is_taken(self, tmp_path: Path) -> None:
        # The control that makes the two above mean something: the same rsync,
        # allowed to finish, must produce a file the watcher will pick up.
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        source = tmp_path / "small.flac"
        source.write_bytes(os.urandom(64 * 1024))

        argv = [a for a in build_rsync_command(source, "", str(inbox)) if a != "-s"]
        argv[-1] = f"{inbox}/"
        subprocess.run(argv, check=True, capture_output=True)

        entries = [(p.name, p.stat().st_mtime) for p in inbox.iterdir() if p.is_file()]
        assert select_next(entries, None) == "small.flac"
