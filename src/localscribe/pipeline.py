# pattern: Imperative Shell
"""Run one recording through the model pipeline to a transcript on disk.

Lifted out of the `scribe` command so `scribe-record` can reuse the exact same path
after a capture. Heavy stages (pyannote, NeMo, wespeaker) are imported lazily and
freed between stages (guardrail #1); diarisation runs once over the whole file for
consistent labels; outputs are written atomically beside the recording.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path


def transcribe_recording(
    recording: Path,
    *,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    name_threshold: float = 0.5,
    echo: Callable[[str], None] = print,
) -> None:
    """Diarise, transcribe, reconcile and render one recording to ``.md`` + ``.json``.

    Raises a ``LocalscribeError`` subclass if a stage fails; the caller decides what
    to do with the (untouched) recording. ``echo`` receives progress lines.
    """
    # Lazy: importing these pulls in torch / pyannote / NeMo.
    from .audio import to_mono_16k
    from .diarize import diarize
    from .identify import apply_names, match_speakers
    from .reconcile import reconcile
    from .render import render_json, render_markdown
    from .runtime import free_cuda, quiet_nemo_logging
    from .transcribe import transcribe
    from .voices import embed_speaker_turns, load_voices

    # NeMo installs its logger on import and overrides anything set beforehand,
    # so the level has to come down here rather than in quiet_third_party.
    quiet_nemo_logging()

    enrolled = load_voices()

    echo(f"preparing {recording.name} ...")
    prepared = to_mono_16k(recording)
    try:
        echo(f"diarising {recording.name} ...")
        turns = diarize(prepared, min_speakers=min_speakers, max_speakers=max_speakers)
        free_cuda()

        names: dict[str, str] = {}
        if enrolled:
            echo(f"matching speakers for {recording.name} ...")
            clusters: dict[str, list[float]] = {}
            for speaker in sorted({turn.speaker for turn in turns}):
                embedding = embed_speaker_turns(
                    prepared, [t for t in turns if t.speaker == speaker]
                )
                if embedding is not None:
                    clusters[speaker] = embedding
            names = match_speakers(clusters, enrolled, threshold=name_threshold)
            free_cuda()

        echo(f"transcribing {recording.name} ...")
        # Again: loading the ASR model reinstates NeMo's own log level, so the
        # call at the top of this function no longer holds by the time we get here.
        quiet_nemo_logging()
        words = transcribe(prepared)
        free_cuda()
    finally:
        prepared.unlink(missing_ok=True)

    segments = apply_names(reconcile(words, turns), names)
    _write_outputs(
        recording,
        render_markdown(segments, source=recording.name),
        render_json(segments, source=recording.name),
    )
    echo(f"wrote {recording.with_suffix('.md').name} and .json")


def _write_outputs(recording: Path, markdown: str, json_text: str) -> None:
    """Write the markdown and JSON transcripts atomically, beside the recording."""
    _atomic_write(recording.with_suffix(".md"), markdown)
    _atomic_write(recording.with_suffix(".json"), json_text)


def _atomic_write(path: Path, text: str) -> None:
    """Write via a temp file + rename so a crash never leaves a half-written file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
