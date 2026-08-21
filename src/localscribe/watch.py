# pattern: Imperative Shell
"""Watch the inbox, transcribe what arrives, into the folder the watcher runs in.

`scribe-watch` is the receiving half of the split that lets a laptop with a
microphone and no GPU hand work to the GPU host. It polls rather than
using inotify: a poll costs one ``listdir`` every few seconds, works over any
filesystem including a network mount, and the latency is nothing against a job
that runs for minutes.

The cycle is claim, deliver, transcribe, and the order is deliberate. Delivering
before transcribing means the recording is already in its final home when the
model starts, so a crash, an OOM or a Ctrl-C during transcription leaves the audio
sitting where the user wanted it, retryable with a plain ``scribe`` (guardrail 2).
The reverse order would strand the audio in a working directory nobody looks at.

Only one transcription runs at a time, across every watcher on the machine,
enforced by an flock. Two watchers started in two project folders is a reasonable
thing to do and would otherwise put two 24 GB pipelines on one card.

Pure helpers (``unique_name``) and the filesystem cycle are tested; the flock's
cross-process exclusion is verified by hand.
"""

from __future__ import annotations

import fcntl
import os
import shutil
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from .errors import LocalscribeError
from .inbox import CLAIMED_DIR, claim_name, inbox_dir, original_name, select_next

#: One transcription at a time per machine. Two watchers in two folders is a
#: reasonable setup and would otherwise put two model pipelines on one 24 GB card.
_GPU_LOCK_NAME = "gpu.lock"

_POLL_SECONDS = 3.0


def gpu_lock_path() -> Path:
    """The lock file serialising model runs across every watcher on this machine."""
    return inbox_dir().parent / _GPU_LOCK_NAME


@contextmanager
def gpu_lock(
    path: Path | None = None, *, waiting: Callable[[], None] | None = None
) -> Iterator[None]:
    """Hold an exclusive lock for the duration of one model run.

    Blocks rather than failing: a second watcher should queue behind the first,
    not drop the recording it just claimed. ``waiting`` is called once if the lock
    is not immediately free, so the user learns why nothing is happening.
    """
    target = path or gpu_lock_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = os.open(target, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            if waiting is not None:
                waiting()
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        os.close(handle)


def claim(inbox: Path, name: str, token: str) -> Path | None:
    """Atomically take ownership of one queued recording.

    Returns the claimed path, or None if another watcher renamed it first. The
    rename is the whole mechanism: on a shared filesystem exactly one caller can
    win it, and the loser sees ``FileNotFoundError`` rather than a duplicate job.
    """
    claimed_dir = inbox / CLAIMED_DIR
    claimed_dir.mkdir(parents=True, exist_ok=True)
    target = claimed_dir / claim_name(name, token)
    try:
        os.rename(inbox / name, target)
    except FileNotFoundError:
        return None
    return target


def unique_name(name: str, exists: Callable[[str], bool]) -> str:
    """A name that does not collide, suffixing ``-2``, ``-3`` … before the extension.

    Recordings are timestamped to the second, so a collision means a genuine
    second file. Overwriting one would destroy audio, which this tool never does.
    """
    if not exists(name):
        return name
    stem, dot, suffix = name.rpartition(".")
    if not dot:
        stem, suffix = name, ""
    counter = 2
    while True:
        candidate = f"{stem}-{counter}{dot}{suffix}"
        if not exists(candidate):
            return candidate
        counter += 1


def deliver(claimed: Path, destination: Path) -> Path:
    """Move a claimed recording into ``destination`` under its original name.

    ``shutil.move`` rather than ``os.rename`` because the inbox and the destination
    are routinely on different filesystems.
    """
    destination.mkdir(parents=True, exist_ok=True)
    final_name = unique_name(
        original_name(claimed.name), lambda n: (destination / n).exists()
    )
    final = destination / final_name
    shutil.move(str(claimed), str(final))
    return final


def orphaned_claims(inbox: Path) -> list[str]:
    """Claims sitting in the working area, newest last.

    A non-empty list after a clean start means a watcher was killed between
    claiming and delivering. Reported rather than auto-recovered, because a live
    watcher's in-flight claim looks exactly the same from here.
    """
    claimed_dir = inbox / CLAIMED_DIR
    if not claimed_dir.is_dir():
        return []
    return sorted(p.name for p in claimed_dir.iterdir() if p.is_file())


def run_once(
    inbox: Path,
    destination: Path,
    *,
    transcribe: Callable[[Path], None],
    needle: str | None = None,
) -> Path | None:
    """Take at most one recording through the full cycle; return where it landed.

    Returns None when there is nothing to do. A failure inside ``transcribe``
    propagates *after* the recording has been delivered, so the caller can report
    it while the audio stays safe in the destination.
    """
    if not inbox.is_dir():
        return None
    entries = [(p.name, p.stat().st_mtime) for p in inbox.iterdir() if p.is_file()]
    name = select_next(entries, needle)
    if name is None:
        return None
    claimed = claim(inbox, name, uuid.uuid4().hex[:8])
    if claimed is None:
        return None
    final = deliver(claimed, destination)
    transcribe(final)
    return final


def watch(
    destination: Path,
    *,
    transcribe: Callable[[Path], None],
    needle: str | None = None,
    poll_seconds: float = _POLL_SECONDS,
    once: bool = False,
    echo: Callable[[str], None] = print,
    inbox: Path | None = None,
) -> int:
    """Poll the inbox until interrupted, transcribing each arrival into ``destination``.

    Returns the number of recordings that failed to transcribe. A failure is
    reported and the loop continues: one poison recording must not stop the queue,
    and the audio it came from is already safe in ``destination``.
    """
    box = inbox or inbox_dir()
    box.mkdir(parents=True, exist_ok=True)

    stranded = orphaned_claims(box)
    if stranded:
        echo(f"warning: {len(stranded)} claim(s) left in {box / CLAIMED_DIR} by an earlier run:")
        for name in stranded:
            echo(f"  {name}")
        echo("  another watcher may hold these; if not, move them back into the inbox.")

    failures = 0
    while True:
        try:
            final = run_once_locked(
                box, destination, transcribe=transcribe, needle=needle, echo=echo
            )
        except LocalscribeError as exc:
            failures += 1
            echo(f"transcription failed: {exc}")
            echo("  the recording is safe in the destination — rerun `scribe` on it to retry")
            final = None
        except KeyboardInterrupt:
            echo("stopped.")
            return failures

        if final is None and once:
            return failures
        if final is None:
            try:
                time.sleep(poll_seconds)
            except KeyboardInterrupt:
                echo("stopped.")
                return failures
        elif once:
            return failures


def run_once_locked(
    inbox: Path,
    destination: Path,
    *,
    transcribe: Callable[[Path], None],
    needle: str | None,
    echo: Callable[[str], None],
) -> Path | None:
    """``run_once``, announcing what it took and holding the GPU lock while it runs."""

    def announce(path: Path) -> None:
        echo(f"transcribing {path.name} -> {path.parent}")
        with gpu_lock(waiting=lambda: echo("  waiting for another watcher to finish ...")):
            transcribe(path)

    return run_once(inbox, destination, transcribe=announce, needle=needle)
