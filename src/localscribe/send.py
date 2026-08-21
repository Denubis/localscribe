# pattern: Imperative Shell
"""Push a recording to the machine that has the GPU.

The sending half of the split: a laptop records with `scribe-send`, this module
puts the FLAC in the remote inbox, and `scribe-watch` on the far side turns it
into a transcript. Nothing here imports torch, so a laptop installs plain
``localscribe`` and never downloads a CUDA wheel.

The recording is written locally first and only then pushed. A network drop
during a sixty-minute lecture therefore costs a retry, never the audio, and the
local copy is kept until the caller says otherwise.

A transfer is confirmed by comparing SHA-256 digests across the two machines.
That check is written so it can only pass on a genuine match: an ssh that failed,
a missing file, or any output that is not a digest raises rather than falling
through to "verified" (see ``parse_sha256``).

Pure helpers (``parse_sha256``, ``confirm_transfer``, the command builders) are
tested; running rsync/ssh is the boundary and is verified by hand.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from collections.abc import Callable
from pathlib import Path

from .capture import ensure_has_audio
from .errors import CaptureError
from .inbox import build_rsync_command, remote_inbox_default

_DIGEST_LENGTH = 64
_HEX = set("0123456789abcdef")


def staging_dir() -> Path:
    """Where recordings are written before they are pushed."""
    override = os.environ.get("LOCALSCRIBE_STAGING")
    if override:
        return Path(override).expanduser()
    cache_home = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(cache_home) / "localscribe" / "outgoing"


def default_host() -> str | None:
    """The transcription box, from ``LOCALSCRIBE_HOST``; None if unset."""
    return os.environ.get("LOCALSCRIBE_HOST") or None


def quote(value: str) -> str:
    """Single-quote a value for a remote shell, POSIX and fish alike.

    Always quotes rather than quoting only when necessary, so the argument a test
    pins is the argument that gets sent.
    """
    return "'" + value.replace("'", "'\"'\"'") + "'"


def remote_file_path(remote_inbox: str, filename: str) -> str:
    """Where ``filename`` will sit on the far side once rsync has finished."""
    return f"{remote_inbox.rstrip('/')}/{filename}"


def build_mkdir_command(host: str, remote_inbox: str) -> list[str]:
    """Create the inbox before pushing, so a missing directory is not an rsync error."""
    return ["ssh", host, f"mkdir -p {quote(remote_inbox)}"]


def build_checksum_command(host: str, remote_file: str) -> list[str]:
    """Ask the far side to hash what it received."""
    return ["ssh", host, f"sha256sum {quote(remote_file)}"]


def parse_sha256(output: str) -> str:
    """Pull the digest out of ``sha256sum`` output.

    Raises unless the first field really is a 64-character hex digest. This is the
    whole reason the verification means anything: an ssh that printed an error, or
    printed nothing at all, must not be readable as a successful comparison.
    """
    first = output.strip().split(maxsplit=1)
    candidate = first[0].lower() if first else ""
    if len(candidate) != _DIGEST_LENGTH or not set(candidate) <= _HEX:
        raise CaptureError(
            "could not read a SHA-256 digest from the remote host; got: "
            f"{output.strip()[:200]!r}"
        )
    return candidate


def digest_of(path: Path) -> str:
    """SHA-256 of a local file, read in chunks so a long recording fits in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def confirm_transfer(local_digest: str, remote_checksum_output: str) -> None:
    """Raise unless the remote copy is byte-identical to the local one."""
    remote_digest = parse_sha256(remote_checksum_output)
    if remote_digest != local_digest.lower():
        raise CaptureError(
            "the copy on the remote host does not match the local recording "
            f"(local {local_digest[:12]}…, remote {remote_digest[:12]}…); "
            "the local file is untouched, try sending again"
        )


def _run(argv: list[str], *, capture: bool = False) -> str:
    """Run a command, turning any failure into a CaptureError with its stderr."""
    try:
        result = subprocess.run(argv, check=True, capture_output=capture, text=True)
    except FileNotFoundError as exc:
        raise CaptureError(f"{argv[0]} not found on PATH; it is required to send") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip() or f"exit status {exc.returncode}"
        raise CaptureError(f"{argv[0]} failed: {detail}") from exc
    return result.stdout or "" if capture else ""


def push(
    local: Path,
    host: str,
    *,
    remote_inbox: str | None = None,
    echo: Callable[[str], None] = print,
) -> str:
    """Push one recording to ``host``'s inbox and verify it arrived intact.

    Returns the remote path. Raises ``CaptureError`` without touching the local
    file if anything goes wrong, so a failed send is always retryable.
    """
    inbox = remote_inbox or remote_inbox_default()

    # Before spending a transfer and a GPU slot on it. A digest comparison cannot
    # catch a dead recording: an empty file hashes identically on both machines,
    # so the verification below would confirm it arrived perfectly intact.
    ensure_has_audio(local)

    echo(f"hashing {local.name} ...")
    local_digest = digest_of(local)

    _run(build_mkdir_command(host, inbox))
    echo(f"sending {local.name} -> {host}:{inbox}/")
    _run(build_rsync_command(local, host, inbox))

    remote = remote_file_path(inbox, local.name)
    echo("verifying ...")
    confirm_transfer(local_digest, _run(build_checksum_command(host, remote), capture=True))
    echo(f"verified {local_digest[:12]}… on {host}")
    return remote
