# pattern: Imperative Shell
"""Live capture: record your mic plus the system output to one FLAC.

The case this serves: a Teams call where remote participants come out of the
speakers. PipeWire exposes that system output as the default sink's ``.monitor``
source, so we tap two sources at once — your mic and that monitor — and let a
single ffmpeg process mix them under one clock (no cross-stream drift).

By default the two sources land on **separate channels**: you on the left,
everyone in the call on the right. That hands the transcription pipeline the
you-vs-them split for free; diarisation then only has to separate the remote
speakers from each other. ``mono=True`` collapses both into one channel instead.

Pure helpers (``parse_default_node``, ``build_capture_command``,
``default_capture_path``) are unit-tested. Running wpctl/ffmpeg is the external
boundary and stays here in the shell, verified by hand — the same split audio.py
draws.
"""

from __future__ import annotations

import json
import re
import signal
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .errors import CaptureError

# A default property line from `wpctl inspect`, e.g. `  * node.name = "alsa_..."`.
# Anchored on ``node.name `` so the neighbouring node.nick / node.description
# decoy lines never match.
_NODE_NAME = re.compile(r'^\s*\*?\s*node\.name = "([^"]+)"', re.MULTILINE)

# Each input is downmixed to mono and async-resampled so a late/early start on
# one source can't desynchronise the pair over a long meeting.
_DOWNMIX = "pan=mono|c0=0.5*c0+0.5*c1,aresample=async=1"

# A USB capture node (a webcam mic, a USB headset) cannot be rerouted by the
# analog combo jack's headphones/headset/mic picker the way `alsa_input.pci-...`
# can — so we prefer it as the recording source. See the robust-capture design.
_USB_PREFIX = "alsa_input.usb-"

# The mic leg carries a silencedetect guard. -50 dB detects *absence of signal*: a
# live mic in a room sits above this on ambient tone, a dead/muted/wrong-port channel
# below — so a warning means "the channel is dead", not "you stopped talking".
# ``d`` is the continuous dead air, in seconds, before silencedetect fires.
_SILENCE_NOISE_DB = 50
_SILENCE_MIN_SECONDS = 20
_SILENCE_FILTER = f"silencedetect=noise=-{_SILENCE_NOISE_DB}dB:d={_SILENCE_MIN_SECONDS}"

# silencedetect logs to stderr as ``silence_start: <t>`` and
# ``silence_end: <t> | silence_duration: <d>`` while passing audio through unchanged.
_SILENCE_START = re.compile(r"silence_start:\s*([0-9.]+)")
_SILENCE_END = re.compile(r"silence_end:\s*([0-9.]+)\s*\|\s*silence_duration:\s*([0-9.]+)")


def parse_default_node(wpctl_output: str) -> str:
    """Pull the ``node.name`` value out of ``wpctl inspect`` output.

    Raises ``CaptureError`` if no node.name line is present, rather than
    returning a half-right guess that ffmpeg would later reject.
    """
    match = _NODE_NAME.search(wpctl_output)
    if match is None:
        raise CaptureError("could not find a node.name in wpctl output")
    return match.group(1)


def parse_sources(pw_dump_json: str) -> list[str]:
    """Pull the ``node.name`` of every ``Audio/Source`` node out of ``pw-dump`` JSON.

    Sinks, monitors and malformed entries are skipped. A dump that does not parse
    degrades to an empty list rather than raising — enumeration is a best-effort
    convenience for the mic policy, never a hard failure (guardrail #2).
    """
    try:
        objects = json.loads(pw_dump_json)
    except (ValueError, TypeError):
        return []
    if not isinstance(objects, list):
        return []
    names: list[str] = []
    for obj in objects:
        props = (obj.get("info") or {}).get("props") or {} if isinstance(obj, dict) else {}
        if props.get("media.class") == "Audio/Source":
            name = props.get("node.name")
            if name:
                names.append(name)
    return names


def choose_mic(sources: list[str], explicit: str | None, default: str) -> str:
    """Pick the mic node to record from.

    Priority: an ``explicit`` override wins; otherwise keep the system ``default``
    if it is already a USB node; otherwise the first USB source on offer; otherwise
    fall back to ``default``. The point is to avoid the analog combo-jack source,
    whose input route a headphone plug can silently switch to a dead port.
    """
    if explicit:
        return explicit
    if default.startswith(_USB_PREFIX):
        return default
    for source in sources:
        if source.startswith(_USB_PREFIX):
            return source
    return default


def build_capture_command(
    mic: str, monitor: str | None, output: Path, *, mono: bool = False
) -> list[str]:
    """Build the ffmpeg argv that records ``mic`` (input 0), optionally mixing ``monitor``.

    ``monitor`` given (remote/hybrid call): a 2-channel FLAC, mic left / monitor
    right; ``mono=True`` mixes both down to a single channel instead.
    ``monitor`` is ``None`` (in-person): a single mono input, no system tap — so
    stray desktop audio cannot enter the recording as a phantom speaker.
    """
    # In a filter_complex, a comma chains filters on one pad and a semicolon separates
    # pads; so `{_DOWNMIX},{_SILENCE_FILTER}` runs the guard on the downmixed mic in place.
    if monitor is None:
        inputs = ["-f", "pulse", "-i", mic]
        graph = f"[0:a]{_DOWNMIX},{_SILENCE_FILTER}[out]"
    else:
        inputs = ["-f", "pulse", "-i", mic, "-f", "pulse", "-i", monitor]
        mix = "amix=inputs=2:normalize=0" if mono else "join=inputs=2:channel_layout=stereo"
        graph = (
            f"[0:a]{_DOWNMIX},{_SILENCE_FILTER}[me];"
            f"[1:a]{_DOWNMIX}[them];[me][them]{mix}[out]"
        )
    # No -nostdin. It makes ffmpeg ignore `q`, the only graceful stop it
    # documents, leaving signals as the only way to stop a capture. Measured
    # 2026-08-01: under -nostdin a stop had to escalate to SIGKILL and produced a
    # 0-byte file. stdin is a pipe, not the terminal, so ffmpeg cannot eat the
    # user's keystrokes.
    return [
        # stdin is reserved for stopping: never wait there for an overwrite answer.
        "ffmpeg", "-hide_banner", "-n",
        *inputs,
        "-filter_complex", graph,
        "-map", "[out]", "-c:a", "flac",
        str(output),
    ]


def sanitise_slug(slug: str | None) -> str:
    """Reduce a user-supplied recording name to something safe in a filename.

    Defaults to ``meeting``. The result also has to survive being a substring
    filter on the far side, since `scribe-watch --match` is how two projects share
    one inbox, so it stays lowercase alphanumerics and hyphens.
    """
    if slug is None:
        return "meeting"
    cleaned = re.sub(r"[^a-z0-9]+", "-", slug.lower()).strip("-")
    if not cleaned:
        raise CaptureError(f"recording name reduces to nothing usable: {slug!r}")
    return cleaned


def default_capture_path(now: datetime, directory: Path, slug: str | None = None) -> Path:
    """A timestamped FLAC name like ``meeting-20260610-155703.flac``."""
    return directory / f"{sanitise_slug(slug)}-{now:%Y%m%d-%H%M%S}.flac"


def resolve_record_mode(*, remote: bool, in_person: bool, prompt_answer: str | None) -> bool:
    """Decide the capture shape: ``True`` = remote (mic + system, stereo), ``False`` =
    in-person (mic only, mono).

    Explicit flags win and are mutually exclusive. With no flag, the interactive
    ``prompt_answer`` decides — any answer beginning with "i" means in-person — and a
    missing answer (non-interactive) defaults to remote, preserving prior behaviour.
    """
    if remote and in_person:
        raise CaptureError("--remote and --in-person cannot be combined")
    if remote:
        return True
    if in_person:
        return False
    if prompt_answer is None:
        return True
    return not prompt_answer.strip().lower().startswith("i")


@dataclass(frozen=True, slots=True)
class SilenceEvent:
    """A silence transition reported by ffmpeg's silencedetect on the mic leg."""

    kind: str  # "start" or "end"
    at: float  # seconds into the recording
    duration: float | None = None  # set only on "end"


def parse_silence_events(line: str) -> SilenceEvent | None:
    """Parse one ffmpeg stderr line into a SilenceEvent, or None if it is neither.

    Progress and banner lines return None, so the caller can feed every stderr line
    through unconditionally.
    """
    end = _SILENCE_END.search(line)
    if end is not None:
        return SilenceEvent("end", float(end.group(1)), float(end.group(2)))
    start = _SILENCE_START.search(line)
    if start is not None:
        return SilenceEvent("start", float(start.group(1)))
    return None


class SilenceTracker:
    """Turn a stream of ffmpeg stderr lines into at-most-N dead-channel warnings.

    A ``silence_start`` means the mic has carried no signal for the guard window and
    earns one warning (capped, so a flapping channel cannot spam). The matching
    ``silence_end`` earns a one-line "resumed" note, but only for a stretch we warned
    about, so the note always pairs with a visible warning.
    """

    def __init__(self, mic_label: str, *, max_warnings: int = 3) -> None:
        self._label = mic_label
        self._max_warnings = max_warnings
        self._warnings_emitted = 0
        self._warned_this_stretch = False

    def observe(self, line: str) -> str | None:
        """Consume one stderr line; return a warning/resume message, or None."""
        event = parse_silence_events(line)
        if event is None:
            return None
        if event.kind == "start":
            if self._warnings_emitted < self._max_warnings:
                self._warnings_emitted += 1
                self._warned_this_stretch = True
                return (
                    f"⚠ no signal on mic '{self._label}' for {_SILENCE_MIN_SECONDS}s+. "
                    "Check it is the right input."
                )
            self._warned_this_stretch = False
            return None
        if self._warned_this_stretch:
            self._warned_this_stretch = False
            return f"✓ mic '{self._label}' signal resumed ({event.at:.0f}s)."
        return None


def parse_probe_duration(probe_json: str) -> float:
    """Seconds of audio ffprobe found, or 0.0 if it found none.

    Reads the audio stream's duration, falling back to the container's. A stream
    with no sample rate does not count, because that is exactly what ffprobe
    reports for a zero-byte FLAC: an audio stream, ``sample_rate`` of "0", and no
    duration key at all. Checking merely that a stream exists would pass it.

    Unparseable output degrades to 0.0 rather than raising, which still stops the
    caller: not being able to confirm audio is not confirming audio.
    """
    try:
        probe = json.loads(probe_json)
    except (ValueError, TypeError):
        return 0.0
    if not isinstance(probe, dict):
        return 0.0

    streams = probe.get("streams") or []
    audio = None
    for stream in streams:
        if isinstance(stream, dict) and stream.get("codec_type") == "audio":
            audio = stream
            break
    if audio is None or _as_float(audio.get("sample_rate")) <= 0:
        return 0.0

    duration = _as_float(audio.get("duration"))
    if duration <= 0:
        duration = _as_float((probe.get("format") or {}).get("duration"))
    return max(duration, 0.0)


def _as_float(value: object) -> float:
    """Best-effort float, 0.0 for None, "N/A" and anything else unreadable."""
    if not isinstance(value, str | int | float):
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def ensure_has_audio(path: Path, *, context: str = "") -> None:
    """Raise ``CaptureError`` unless ``path`` holds a positive-duration audio stream.

    Guardrail 4 taken to its limit: a capture that recorded nothing at all must
    not be handed onwards. It costs nothing to check and it prevents a dead file
    being transferred across a network and queued onto the GPU before anything
    notices, which is what happened on 2026-08-01.
    """
    if not path.is_file():
        raise CaptureError(f"the recording was not written: {path}{context}")
    if path.stat().st_size == 0:
        raise CaptureError(
            f"the recording is empty (0 bytes): {path}\n"
            f"ffmpeg produced no output at all, so no audio was captured.{context}"
        )
    try:
        result = subprocess.run(
            [
                "ffprobe", "-hide_banner", "-v", "error",
                "-show_entries", "stream=codec_type,sample_rate,duration",
                "-show_entries", "format=duration",
                "-of", "json", str(path),
            ],
            check=False, capture_output=True, text=True,
        )
    except FileNotFoundError as exc:
        raise CaptureError("ffprobe not found on PATH; it ships with ffmpeg") from exc

    if parse_probe_duration(result.stdout) <= 0:
        raise CaptureError(
            f"the recording contains no audio: {path}\n"
            f"ffprobe found no playable stream in {path.stat().st_size} bytes.{context}"
        )


def _resolve_default(node: str) -> str:
    """Return the node.name of a wpctl default token (e.g. ``@DEFAULT_AUDIO_SINK@``)."""
    try:
        result = subprocess.run(
            ["wpctl", "inspect", node], check=True, capture_output=True, text=True
        )
    except FileNotFoundError as exc:
        raise CaptureError(
            "wpctl not found on PATH; it ships with WirePlumber and is required to "
            "find the default audio devices"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise CaptureError(f"wpctl could not inspect {node}: {exc.stderr.strip()}") from exc
    return parse_default_node(result.stdout)


def list_sources() -> list[str]:
    """Best-effort enumeration of Audio/Source node names via ``pw-dump``.

    Returns ``[]`` if pw-dump is missing or fails — the caller then falls back to
    the system default source. Never raises: source discovery is a convenience.
    """
    try:
        result = subprocess.run(["pw-dump"], check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return parse_sources(result.stdout)


def resolve_mic(explicit: str | None) -> str:
    """The concrete mic node to record: an explicit override, else the policy choice."""
    if explicit:
        return explicit
    default = _resolve_default("@DEFAULT_AUDIO_SOURCE@")
    return choose_mic(list_sources(), None, default)


def spawn_capture(command: list[str]) -> subprocess.Popen[str]:
    """Start a capture detached from the terminal's process group.

    ``start_new_session`` is the point. Ctrl-C in a terminal is delivered to every
    process in the foreground group, so without this ffmpeg receives the user's
    SIGINT *and* the stop we send deliberately. On ffmpeg 8 a signal arriving
    during shutdown aborts the trailer write and the recording is lost. Detached,
    the only stop ffmpeg ever sees is the one we choose to send.
    """
    try:
        return subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        raise CaptureError(f"{command[0]} not found on PATH; it is required to record") from exc


def request_stop(proc: subprocess.Popen[str]) -> None:
    """Ask ffmpeg to finish and write its trailer, in its own language.

    ``q`` on stdin is ffmpeg's documented graceful stop, and unlike a signal it
    lets the muxer close the file properly. Closing stdin afterwards is a second
    hint for anything that reads to EOF instead.
    """
    if proc.stdin is None or proc.stdin.closed:
        return
    try:
        proc.stdin.write("q")
        proc.stdin.flush()
        proc.stdin.close()
    except (BrokenPipeError, ValueError, OSError):
        pass  # already gone; the escalation below still applies


def stop_gracefully(proc: subprocess.Popen[str], *, timeout: float = 15.0) -> None:
    """Stop a capture cleanly, escalating only if it refuses to go.

    Escalation exists so a wedged ffmpeg cannot hang the caller forever, not
    because it is expected: reaching SIGINT here means the recording is probably
    already compromised, which ``ensure_has_audio`` will then catch.
    """
    if proc.poll() is not None:
        return
    request_stop(proc)
    for stage in (None, signal.SIGINT, signal.SIGKILL):
        if stage is not None:
            proc.send_signal(stage)
        try:
            proc.wait(timeout=timeout if stage is None else 5.0)
            return
        except subprocess.TimeoutExpired:
            continue
    proc.wait()


def capture(
    output: Path, *, mic: str | None = None, mono: bool = False, remote: bool = True
) -> Path:
    """Record to ``output`` until interrupted; return its path.

    ``mic`` defaults to the policy choice (a USB source over the analog jack).
    ``remote=True`` also taps the default sink's monitor for the far end of a call
    (stereo, or ``mono`` to mix); ``remote=False`` records the mic alone, mono, for
    an in-person meeting. The sink monitor is resolved at call time so switching the
    default output between runs still works. Press Ctrl-C to stop: the parent asks
    ffmpeg to quit through stdin and finalise the FLAC cleanly.
    """
    if output.exists():
        raise CaptureError(f"recording output already exists: {output}; choose a new filename")
    mic_node = resolve_mic(mic)
    monitor_node = _resolve_default("@DEFAULT_AUDIO_SINK@") + ".monitor" if remote else None
    command = build_capture_command(mic_node, monitor_node, output, mono=mono)
    proc = spawn_capture(command)
    assert proc.stderr is not None  # stderr=PIPE above

    tracker = SilenceTracker(mic_node)
    recent: deque[str] = deque(maxlen=20)  # keep a tail for a useful message if ffmpeg dies
    stopped_by_user = False
    output_failed = False
    # iter(readline, "") reads line-by-line with no read-ahead, so a silence_start
    # surfaces as soon as ffmpeg emits it rather than waiting for a buffer to fill.
    try:
        for line in iter(proc.stderr.readline, ""):
            recent.append(line.rstrip())
            output_failed = output_failed or "Error opening output file " in line
            warning = tracker.observe(line)
            if warning is not None:
                print(warning, file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        # Ask ffmpeg to finish rather than signalling it, so the FLAC trailer gets
        # written, then drain stderr until it closes. The watchdog only fires if
        # ffmpeg ignores the request, so draining cannot hang the stop.
        stopped_by_user = True
        watchdog = threading.Timer(15.0, lambda: stop_gracefully(proc, timeout=0.1))
        watchdog.daemon = True
        watchdog.start()
        try:
            request_stop(proc)
            for line in iter(proc.stderr.readline, ""):
                recent.append(line.rstrip())
                output_failed = output_failed or "Error opening output file " in line
        finally:
            watchdog.cancel()
    finally:
        proc.wait()

    # ffmpeg 8 reports an existing output with exit 0 under -n. The file may have
    # appeared after our preflight check, so its playable audio is not evidence
    # that this capture succeeded. Preserve it and report the output refusal.
    if output_failed:
        raise CaptureError("ffmpeg could not write the recording:\n" + "\n".join(recent))

    # A non-zero exit we did not ask for (e.g. the mic could not be opened) used to
    # vanish; surface it with the tail of ffmpeg's own diagnostics.
    if not stopped_by_user and proc.returncode:
        raise CaptureError("ffmpeg exited unexpectedly:\n" + "\n".join(recent))

    # Stopping with Ctrl-C used to mask a capture that never started: ffmpeg could
    # fail to open its inputs, write nothing, and still look like a clean stop
    # because the user asked for the stop. Guardrail 4's limit case, so the check
    # runs whether or not the stop was deliberate, and carries ffmpeg's own words.
    ensure_has_audio(output, context="\n\nffmpeg's last output:\n" + "\n".join(recent))
    return output
