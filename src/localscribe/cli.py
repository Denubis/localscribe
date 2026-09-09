# pattern: Imperative Shell
"""Command-line entry point for localscribe.

Kept cold: no torch / pyannote / NeMo imports at module load, so `scribe --help`
stays fast. The heavy stages are imported lazily, and only after the Hugging Face
cache has been selected (guardrail #3).

User messages go to stdout; the actual transcripts are written to files, so
stdout is free for progress and errors.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

from .errors import LocalscribeError
from .runtime import configure_cache, quiet_third_party, require_gpu_extra

app = typer.Typer(
    add_completion=False,
    help="Transcribe FLAC meeting recordings into speaker-attributed transcripts.",
)


@app.command()
def scribe(
    recordings: Annotated[
        list[Path], typer.Argument(help="One or more audio files (e.g. FLAC).")
    ],
    min_speakers: Annotated[int | None, typer.Option(help="Lower bound on speaker count.")] = None,
    max_speakers: Annotated[int | None, typer.Option(help="Upper bound on speaker count.")] = None,
    name_threshold: Annotated[
        float, typer.Option(help="Cosine threshold for matching enrolled voices (0-1).")
    ] = 0.5,
) -> None:
    """Write a speaker-attributed transcript (.md + .json) next to each recording."""
    missing = [r for r in recordings if not r.is_file()]
    if missing:
        for path in missing:
            typer.echo(f"error: file not found: {path}")
        raise typer.Exit(code=2)

    try:
        require_gpu_extra()
    except LocalscribeError as exc:
        typer.echo(f"error: {exc}")
        raise typer.Exit(code=3) from exc
    configure_cache()
    quiet_third_party()

    from .pipeline import transcribe_recording

    failures = 0
    for recording in recordings:
        try:
            transcribe_recording(
                recording,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                name_threshold=name_threshold,
                echo=typer.echo,
            )
        except LocalscribeError as exc:
            typer.echo(f"error processing {recording.name}: {exc}")
            failures += 1

    if failures:
        raise typer.Exit(code=1)


enrol_app = typer.Typer(
    add_completion=False,
    help="Enrol a speaker's voice so future transcripts use their real name.",
)


@enrol_app.command()
def enrol(
    sample: Annotated[Path, typer.Argument(help="A short clip of one speaker (e.g. 30s).")],
    name: Annotated[str, typer.Option(help="The speaker's name, e.g. Alice.")],
) -> None:
    """Register SAMPLE as the voice of NAME in the local registry."""
    if not sample.is_file():
        typer.echo(f"error: file not found: {sample}")
        raise typer.Exit(code=2)

    try:
        require_gpu_extra()
    except LocalscribeError as exc:
        typer.echo(f"error: {exc}")
        raise typer.Exit(code=3) from exc
    configure_cache()
    quiet_third_party()
    from .audio import to_mono_16k
    from .voices import embed_clip, save_voice

    try:
        prepared = to_mono_16k(sample)
        try:
            vector = embed_clip(prepared)
        finally:
            prepared.unlink(missing_ok=True)
        path = save_voice(name, vector)
    except LocalscribeError as exc:
        typer.echo(f"error: {exc}")
        raise typer.Exit(code=1) from exc

    typer.echo(f"enrolled {name} ({len(vector)}-d voiceprint) -> {path}")


record_app = typer.Typer(
    add_completion=False,
    help="Record your mic + the system output (e.g. a Teams call) to one FLAC.",
)


def _interactive_input() -> bool:
    """Whether this invocation has a terminal for the Enter rollover control."""
    return sys.stdin.isatty()


@record_app.command()
def record(
    output: Annotated[
        Path | None,
        typer.Argument(help="FLAC to write. Default: ./meeting-<timestamp>.flac"),
    ] = None,
    mic: Annotated[
        str | None,
        typer.Option(help="Mic source node name. Default: a USB source, else the system default."),
    ] = None,
    remote: Annotated[
        bool, typer.Option("--remote", help="Remote/hybrid call: mic + system audio (skip prompt).")
    ] = False,
    in_person: Annotated[
        bool, typer.Option("--in-person", help="In-person: mic only, mono (skip prompt).")
    ] = False,
    mono: Annotated[
        bool,
        typer.Option(help="Remote mode: mix mic + system to one channel instead of L/R."),
    ] = False,
    no_transcribe: Annotated[
        bool, typer.Option("--no-transcribe", help="Keep only the raw FLAC; skip transcription.")
    ] = False,
) -> None:
    """Record a meeting; Enter starts a new part, Ctrl-C finishes recording.

    Asks remote-or-in-person at the start unless --remote/--in-person is given.
    Remote taps your mic plus the system output (the call); in-person records the
    mic alone, so stray desktop audio cannot leak in as a phantom speaker. Pass
    --no-transcribe to stop at the FLAC. In a terminal, Enter finalises the current
    part for background transcription while recording continues in a new file.
    """
    from datetime import datetime

    from .capture import capture, default_capture_path, resolve_mic, resolve_record_mode

    target = output or default_capture_path(datetime.now(), Path.cwd())
    if not target.parent.is_dir():
        typer.echo(f"error: output directory does not exist: {target.parent}")
        raise typer.Exit(code=2)

    answer: str | None = None
    if not remote and not in_person and sys.stdin.isatty():
        answer = typer.prompt("Remote call or in-person? [r/i]", default="r")

    try:
        is_remote = resolve_record_mode(remote=remote, in_person=in_person, prompt_answer=answer)
        chosen_mic = resolve_mic(mic)
        channels = "mono" if mono else "stereo"
        mode_label = (
            f"remote (mic + system, {channels})" if is_remote else "in-person (mic only, mono)"
        )
        typer.echo(f"mic:  {chosen_mic}")
        typer.echo(f"mode: {mode_label}")
        if _interactive_input():
            from .record_session import record_interactively

            typer.echo(f"recording -> {target}   (Enter: next file; Ctrl-C: stop recording)")
            code = record_interactively(
                target, mic=chosen_mic, mono=mono, remote=is_remote,
                no_transcribe=no_transcribe, echo=typer.echo,
            )
            if code:
                raise typer.Exit(code=code)
            return
        typer.echo(f"recording -> {target}   (Ctrl-C to stop)")
        capture(target, mic=chosen_mic, mono=mono, remote=is_remote)
    except LocalscribeError as exc:
        typer.echo(f"error: {exc}")
        raise typer.Exit(code=1) from exc
    typer.echo(f"wrote {target.name}")

    if no_transcribe:
        return

    # Auto-transcribe on stop. A failure here must never cost the recording: the
    # FLAC stays on disk and the user is told how to retry.
    try:
        require_gpu_extra()
    except LocalscribeError as exc:
        typer.echo(f"not transcribing: {exc}")
        typer.echo(f"the recording is safe at {target}")
        typer.echo(f"  after installing the model stack locally, retry: scribe {target}")
        typer.echo("  to transcribe it on the GPU box: scribe-send " + str(target))
        raise typer.Exit(code=3) from exc
    configure_cache()
    quiet_third_party()
    from .pipeline import transcribe_recording

    try:
        transcribe_recording(target, echo=typer.echo)
    except LocalscribeError as exc:
        typer.echo(f"transcription failed: {exc}")
        typer.echo(f"the recording is safe at {target} — run `scribe {target}` to retry")
        raise typer.Exit(code=1) from exc
    typer.echo(f"transcript -> {target.with_suffix('.md')}")


send_app = typer.Typer(
    add_completion=False,
    help="Record here, then push the FLAC to the machine that has the GPU.",
)


@send_app.command()
def send(
    recording: Annotated[
        Path | None,
        typer.Argument(help="An existing FLAC to push. Omit to record one first."),
    ] = None,
    host: Annotated[
        str | None,
        typer.Option(help="Transcription box. Default: $LOCALSCRIBE_HOST."),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option(
            "--name",
            "-n",
            help="Label the recording, e.g. -n project-a. Doubles as the watcher's --match needle.",
        ),
    ] = None,
    mic: Annotated[
        str | None,
        typer.Option(help="Mic source node name. Default: a USB source, else the system default."),
    ] = None,
    remote: Annotated[
        bool, typer.Option("--remote", help="Remote/hybrid call: mic + system audio (skip prompt).")
    ] = False,
    in_person: Annotated[
        bool, typer.Option("--in-person", help="In-person: mic only, mono (skip prompt).")
    ] = False,
    mono: Annotated[
        bool, typer.Option(help="Remote mode: mix mic + system to one channel instead of L/R.")
    ] = False,
    remote_inbox: Annotated[
        str | None,
        typer.Option(help="Inbox path on the far side, relative to its home directory."),
    ] = None,
    discard_local: Annotated[
        bool,
        typer.Option("--discard-local", help="Delete the local copy once the transfer verifies."),
    ] = False,
) -> None:
    """Record a meeting and push it to the transcription box's inbox.

    The recording is written locally first, so a network drop costs a retry rather
    than the audio, and the local copy is kept unless --discard-local is given.
    Nothing is transcribed here: run `scribe-watch` on the far side, in whichever
    folder the transcript belongs in.
    """
    from datetime import datetime

    from .capture import capture, default_capture_path, resolve_mic, resolve_record_mode
    from .send import default_host, push, staging_dir

    target_host = host or default_host()
    if not target_host:
        typer.echo("error: no host given. Pass --host, or set LOCALSCRIBE_HOST.")
        raise typer.Exit(code=2)

    try:
        if recording is not None:
            if not recording.is_file():
                typer.echo(f"error: file not found: {recording}")
                raise typer.Exit(code=2)
            target = recording
        else:
            staging = staging_dir()
            staging.mkdir(parents=True, exist_ok=True)
            target = default_capture_path(datetime.now(), staging, name)

            answer: str | None = None
            if not remote and not in_person and sys.stdin.isatty():
                answer = typer.prompt("Remote call or in-person? [r/i]", default="r")

            is_remote = resolve_record_mode(
                remote=remote, in_person=in_person, prompt_answer=answer
            )
            chosen_mic = resolve_mic(mic)
            channels = "mono" if mono else "stereo"
            mode = (
                f"remote (mic + system, {channels})" if is_remote else "in-person (mic only, mono)"
            )
            typer.echo(f"mic:  {chosen_mic}")
            typer.echo(f"mode: {mode}")
            typer.echo(f"recording -> {target}   (Ctrl-C to stop)")
            capture(target, mic=chosen_mic, mono=mono, remote=is_remote)
            typer.echo(f"wrote {target.name}")

        remote_path = push(target, target_host, remote_inbox=remote_inbox, echo=typer.echo)
    except LocalscribeError as exc:
        typer.echo(f"error: {exc}")
        raise typer.Exit(code=1) from exc

    typer.echo(f"queued {target_host}:{remote_path}")
    if discard_local:
        target.unlink(missing_ok=True)
        typer.echo("local copy removed (verified on the remote first)")
    else:
        typer.echo(f"local copy kept at {target}")


watch_app = typer.Typer(
    add_completion=False,
    help="Transcribe recordings as they arrive, into the folder you run this in.",
)


@watch_app.command()
def watch_command(
    destination: Annotated[
        Path | None,
        typer.Argument(help="Where transcripts land. Default: the current directory."),
    ] = None,
    match: Annotated[
        str | None,
        typer.Option(help="Only take recordings whose name contains this, e.g. --match project-a."),
    ] = None,
    once: Annotated[
        bool, typer.Option("--once", help="Drain nothing: take at most one recording, then exit.")
    ] = False,
    poll: Annotated[float, typer.Option(help="Seconds between inbox scans.")] = 3.0,
    min_speakers: Annotated[int | None, typer.Option(help="Lower bound on speaker count.")] = None,
    max_speakers: Annotated[int | None, typer.Option(help="Upper bound on speaker count.")] = None,
    name_threshold: Annotated[
        float, typer.Option(help="Cosine threshold for matching enrolled voices (0-1).")
    ] = 0.5,
) -> None:
    """Watch the inbox and transcribe arrivals into DESTINATION.

    Where a transcript lands is decided here, by where you run this, so the
    recording side needs no configuration at all. Run one of these per project
    folder and use --match to keep them out of each other's way.
    """
    target = (destination or Path.cwd()).resolve()
    if destination is not None and not target.is_dir():
        typer.echo(f"error: destination directory does not exist: {target}")
        raise typer.Exit(code=2)

    try:
        require_gpu_extra()
    except LocalscribeError as exc:
        typer.echo(f"error: {exc}")
        raise typer.Exit(code=3) from exc
    configure_cache()
    quiet_third_party()

    from .inbox import inbox_dir
    from .pipeline import transcribe_recording
    from .watch import watch

    def transcribe(path: Path) -> None:
        transcribe_recording(
            path,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            name_threshold=name_threshold,
            echo=typer.echo,
        )

    box = inbox_dir()
    typer.echo(f"inbox:       {box}")
    typer.echo(f"destination: {target}")
    if match:
        typer.echo(f"match:       {match}")
    typer.echo("waiting for recordings ... (Ctrl-C to stop)")

    failures = watch(
        target,
        transcribe=transcribe,
        needle=match,
        poll_seconds=poll,
        once=once,
        echo=typer.echo,
        inbox=box,
    )
    if failures:
        raise typer.Exit(code=1)
