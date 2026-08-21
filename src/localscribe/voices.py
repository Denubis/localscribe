# pattern: Imperative Shell
"""Voice enrolment: speaker embeddings and the on-disk voice registry.

Embeddings come from a pyannote embedding model (the same one for enrolment and
for embedding diarised clusters, so both live in one vector space). The registry
is a directory of ``<name>.json`` files under the user's data dir, overridable via
``LOCALSCRIBE_VOICES_DIR``.

Registry I/O is testable without the model; the embedding functions are exercised
through the live pipeline.
"""

from __future__ import annotations

import json
import logging
import os
from importlib import import_module
from pathlib import Path
from typing import Any

from .errors import EnrolmentError
from .models import SpeakerTurn

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"

#: Diarised turns shorter than this give noisy embeddings; skip them when
#: building a cluster's representative vector.
_MIN_TURN_SECONDS = 1.0
#: Cap how many turns we embed per speaker (longest first) to stay fast.
_MAX_TURNS_PER_SPEAKER = 15

_inference: Any = None


def voices_dir() -> Path:
    """Return the voice registry directory (overridable via env)."""
    override = os.environ.get("LOCALSCRIBE_VOICES_DIR")
    if override:
        return Path(override)
    data_home = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(data_home) / "localscribe" / "voices"


def save_voice(name: str, embedding: list[float], *, directory: Path | None = None) -> Path:
    """Persist an enrolled voice embedding under ``name``; overwrites if it exists."""
    directory = directory or voices_dir()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    path = directory / f"{_safe_name(name)}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.touch(mode=0o600, exist_ok=True)
    tmp.chmod(0o600)
    tmp.write_text(json.dumps({"name": name, "embedding": list(embedding)}), encoding="utf-8")
    os.replace(tmp, path)
    path.chmod(0o600)
    logger.debug("saved voice %r to %s", name, path)
    return path


def load_voices(*, directory: Path | None = None) -> dict[str, list[float]]:
    """Load all enrolled voices as ``{name: embedding}``. Unreadable files are skipped."""
    directory = directory or voices_dir()
    voices: dict[str, list[float]] = {}
    if not directory.exists():
        return voices
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            voices[data["name"]] = [float(x) for x in data["embedding"]]
        except (OSError, ValueError, KeyError, TypeError):
            logger.warning("skipping unreadable voice file %s", path)
    return voices


def embed_clip(sample: Path) -> list[float]:
    """Embed a whole audio clip into one speaker vector (for enrolment)."""
    try:
        vector = _embedding_inference()(str(sample))
    except Exception as exc:  # noqa: BLE001 - boundary: wrap model/audio failure
        raise EnrolmentError(f"failed to embed {sample}: {exc}") from exc
    return [float(x) for x in vector]


def embed_speaker_turns(audio: Path, turns: list[SpeakerTurn]) -> list[float] | None:
    """Build one embedding for a speaker by averaging their (longest) turns.

    Returns None if the speaker has no turn long enough to embed reliably.
    """
    np = import_module("numpy")
    Segment = import_module("pyannote.core").Segment

    inference = _embedding_inference()
    usable = sorted(
        (turn for turn in turns if turn.end - turn.start >= _MIN_TURN_SECONDS),
        key=lambda turn: turn.end - turn.start,
        reverse=True,
    )[:_MAX_TURNS_PER_SPEAKER]

    vectors = []
    for turn in usable:
        try:
            vectors.append(inference.crop(str(audio), Segment(turn.start, turn.end)))
        except Exception:  # noqa: BLE001 - one bad crop should not sink the speaker
            logger.warning("could not embed turn %.2f-%.2f of %s", turn.start, turn.end, audio)
    if not vectors:
        return None
    return np.mean(np.stack(vectors), axis=0).tolist()


def _embedding_inference() -> Any:
    """Lazily load and cache the embedding inference helper."""
    global _inference
    if _inference is None:
        pyannote_audio = import_module("pyannote.audio")
        Inference = pyannote_audio.Inference
        Model = pyannote_audio.Model

        model = Model.from_pretrained(EMBEDDING_MODEL, token=True)
        if model is None:
            raise EnrolmentError(f"could not load embedding model {EMBEDDING_MODEL}")
        _inference = Inference(model, window="whole")
    return _inference


def _safe_name(name: str) -> str:
    """Validate a voice name is usable as a filename (no path traversal)."""
    cleaned = name.strip()
    if not cleaned or cleaned in {".", ".."} or "/" in cleaned or "\\" in cleaned:
        raise EnrolmentError(f"invalid voice name: {name!r}")
    return cleaned
