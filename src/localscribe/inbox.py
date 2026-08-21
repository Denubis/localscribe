# pattern: Functional Core
"""The drop-box that decouples recording from transcription.

A laptop with a microphone and no GPU records a FLAC and pushes it here; the GPU
host runs a watcher that picks it up. Neither end needs to know anything
about the other's filesystem beyond this one directory.

**Where a transcript lands is decided by where the watcher runs, not by the
sender.** ``cd ~/recordings/project-a && scribe-watch`` files everything into
that folder. The sender stays dumb, so nothing has to be configured twice and a
recording can never be addressed to a folder nobody is watching.

Two races matter here, and both are settled by naming rather than by locking:

*A half-transferred file must never be transcribed.* rsync writes to a
dot-prefixed temporary beside the target and renames it into place when the
transfer completes, so ``is_ready`` treats a leading dot as "still in flight".
The rename is atomic within one filesystem, which makes appearance-under-a-plain-
name the completion signal.

*Two watchers in two folders must never take the same recording.* Claiming is an
``os.rename`` into a subdirectory; the loser of the race gets ``FileNotFoundError``
rather than a second copy of the job. That lives in watch.py — this module only
supplies the name it renames to.

Everything here is pure except ``inbox_dir``, which reads the environment.
"""

from __future__ import annotations

import os
from pathlib import Path

from .errors import CaptureError

#: Where the sender pushes to, relative to the *remote* home directory. Relative
#: on purpose: an absolute path or a ``~`` would have to survive the remote login
#: shell (fish here), and rsync resolves a relative remote path against the home
#: directory without any shell expansion at all.
REMOTE_INBOX = ".local/share/localscribe/inbox"

#: rsync parks interrupted transfers here, inside the inbox. The leading dot keeps
#: them invisible to ``is_ready``, so a resumed transfer cannot be picked up early.
PARTIAL_DIR = ".partial"

#: Claimed recordings move here while a watcher owns them. Also dot-prefixed, so a
#: claim can never be re-claimed by a second watcher scanning the inbox.
CLAIMED_DIR = ".claimed"

_AUDIO_SUFFIX = ".flac"


def inbox_dir() -> Path:
    """The local inbox directory, honouring ``LOCALSCRIBE_INBOX`` then XDG."""
    override = os.environ.get("LOCALSCRIBE_INBOX")
    if override:
        return Path(override).expanduser()
    data_home = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(data_home) / "localscribe" / "inbox"


def remote_inbox_default() -> str:
    """The push target on the far side, relative to that machine's home directory."""
    return REMOTE_INBOX


def is_ready(name: str) -> bool:
    """Is ``name`` a completed recording a watcher may take?

    Only a plain ``.flac`` with no leading dot qualifies. Everything rsync creates
    mid-flight is dot-prefixed, and the transcripts a watcher writes are not
    ``.flac``, so this is the whole readiness test.
    """
    if name.startswith("."):
        return False
    return name.lower().endswith(_AUDIO_SUFFIX)


def matches(name: str, needle: str | None) -> bool:
    """Does ``name`` pass the watcher's ``--match`` filter? No filter takes everything.

    This is what lets two projects share one inbox: record with
    ``scribe-send -n project-a`` and the watcher started with
    ``--match project-a`` leaves
    everyone else's recordings alone.
    """
    if needle is None:
        return True
    return needle.lower() in name.lower()


def select_next(entries: list[tuple[str, float]], needle: str | None) -> str | None:
    """Choose the recording to take next from ``(name, mtime)`` pairs, or None.

    Oldest first, so a backlog drains in the order it was recorded; ties break on
    the name so two watchers scanning the same instant agree on the order and
    collide on one claim rather than deadlocking around each other.
    """
    ready = [
        (mtime, name) for name, mtime in entries if is_ready(name) and matches(name, needle)
    ]
    if not ready:
        return None
    return min(ready)[1]


def claim_name(original: str, token: str) -> str:
    """The name a watcher renames ``original`` to in order to claim it.

    The token makes the claim unique; the original name is preserved after the
    first hyphen so the recording keeps its identity through the hand-off and the
    transcript beside it is still recognisable.
    """
    if not token or "/" in token or "\\" in token or token.startswith("."):
        raise CaptureError(f"invalid claim token: {token!r}")
    return f"{token}-{original}"


def original_name(claimed: str) -> str:
    """Recover the pre-claim name written by ``claim_name``."""
    _, _, rest = claimed.partition("-")
    return rest or claimed


def build_rsync_command(local: Path, host: str, remote_inbox: str) -> list[str]:
    """Build the rsync argv that pushes one recording into a remote inbox.

    ``-s`` hands paths to the remote rsync over the protocol instead of letting the
    remote login shell re-split them, which matters because that shell is fish.
    ``--partial-dir`` keeps an interrupted transfer resumable and, being
    dot-prefixed, keeps it invisible to a watcher until it completes.
    """
    return [
        "rsync",
        "-a",
        "-s",
        "--chmod=F600,D700",
        f"--partial-dir={PARTIAL_DIR}",
        "--info=progress2",
        str(local),
        f"{host}:{remote_inbox}/",
    ]
