"""Pure-core tests for live capture.

These never run ffmpeg or wpctl: the subprocess boundary lives in capture.py's
imperative shell and is exercised manually (the same convention audio.py follows).
What is pinned here is the logic worth pinning — which device name we extract,
the exact ffmpeg graph we build, and the timestamped output name.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from localscribe.capture import (
    SilenceEvent,
    SilenceTracker,
    build_capture_command,
    capture,
    choose_mic,
    default_capture_path,
    parse_default_node,
    parse_silence_events,
    parse_sources,
    resolve_record_mode,
    sanitise_slug,
)
from localscribe.errors import CaptureError

# Verbatim shape of `wpctl inspect @DEFAULT_AUDIO_SINK@` on the target box.
# node.description and node.nick are decoys: a naive parser grabs the wrong one.
WPCTL_SINK = (
    "id 53, type PipeWire:Interface:Node\n"
    '  * node.description = "Built-in Audio Analog Stereo"\n'
    '    node.driver = "true"\n'
    '  * node.name = "alsa_output.pci-0000_00_1f.3.analog-stereo"\n'
    '  * node.nick = "ALC3246 Analog"\n'
)


def test_parse_default_node_extracts_node_name_not_nick_or_description() -> None:
    assert parse_default_node(WPCTL_SINK) == "alsa_output.pci-0000_00_1f.3.analog-stereo"


def test_parse_default_node_raises_when_node_name_absent() -> None:
    with pytest.raises(CaptureError):
        parse_default_node('id 53\n  * node.nick = "no name here"\n')


def test_build_capture_command_default_puts_me_left_them_right() -> None:
    cmd = build_capture_command("MIC", "MON", Path("/out/m.flac"))

    assert cmd[0] == "ffmpeg"
    # Two PulseAudio inputs, mic first (left) and monitor second (right).
    assert cmd.count("pulse") == 2
    assert cmd.index("MIC") < cmd.index("MON")
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "join=inputs=2:channel_layout=stereo" in graph
    assert graph.index("[me]") < graph.index("[them]")
    assert cmd[cmd.index("-c:a") + 1] == "flac"
    assert cmd[-1] == "/out/m.flac"


def test_build_capture_command_mono_mixes_to_one_channel() -> None:
    cmd = build_capture_command("MIC", "MON", Path("/out/m.flac"), mono=True)

    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "amix=inputs=2" in graph
    assert "join=" not in graph


def test_default_capture_path_is_timestamped_flac_in_directory() -> None:
    when = datetime(2026, 6, 10, 15, 57, 3)

    assert default_capture_path(when, Path("/x")) == Path("/x/meeting-20260610-155703.flac")


def test_default_capture_path_uses_a_slug_in_place_of_meeting() -> None:
    when = datetime(2026, 6, 10, 15, 57, 3)

    assert default_capture_path(when, Path("/x"), "project-a") == Path(
        "/x/project-a-20260610-155703.flac"
    )


def test_sanitise_slug_defaults_to_meeting() -> None:
    assert sanitise_slug(None) == "meeting"


def test_sanitise_slug_reduces_punctuation_and_case_to_a_filter_safe_token() -> None:
    # The slug doubles as the `scribe-watch --match` needle, so it has to stay
    # something a user can type back exactly.
    assert sanitise_slug("Project A / prep!") == "project-a-prep"


def test_sanitise_slug_rejects_a_name_that_reduces_to_nothing() -> None:
    with pytest.raises(CaptureError):
        sanitise_slug("///")


# --- mic-selection policy (robust-capture.AC1) ---

# Shape mirrors `pw-dump`: a top-level array of objects, each with info.props.
# A sink and a class-less node are decoys the parser must skip.
_PW_DUMP = json.dumps(
    [
        {
            "id": 53,
            "type": "PipeWire:Interface:Node",
            "info": {
                "props": {
                    "media.class": "Audio/Sink",
                    "node.name": "alsa_output.pci-0000_00_1f.3.analog-stereo",
                }
            },
        },
        {
            "id": 54,
            "type": "PipeWire:Interface:Node",
            "info": {
                "props": {
                    "media.class": "Audio/Source",
                    "node.name": "alsa_input.pci-0000_00_1f.3.analog-stereo",
                }
            },
        },
        {
            "id": 33,
            "type": "PipeWire:Interface:Node",
            "info": {
                "props": {
                    "media.class": "Audio/Source",
                    "node.name": "alsa_input.usb-046d_C930c_EXAMPLE-02.analog-stereo",
                }
            },
        },
        {"id": 99, "type": "PipeWire:Interface:Node", "info": {"props": {"x": "no class"}}},
    ]
)

_USB = "alsa_input.usb-046d_C930c_EXAMPLE-02.analog-stereo"
_JACK = "alsa_input.pci-0000_00_1f.3.analog-stereo"


def test_parse_sources_returns_only_audio_source_node_names() -> None:
    # The sink and the class-less node are excluded; order is preserved.
    assert parse_sources(_PW_DUMP) == [_JACK, _USB]


def test_parse_sources_tolerates_malformed_json() -> None:
    # Guardrail: a bad dump degrades to "no sources", never raises.
    assert parse_sources("not json {{") == []


def test_choose_mic_prefers_usb_over_analog_jack() -> None:
    # The jack source is the one a headphone plug can reroute to a dead port.
    assert choose_mic([_JACK, _USB], explicit=None, default=_JACK) == _USB


def test_choose_mic_explicit_overrides_policy() -> None:
    assert choose_mic([_USB], explicit="my-mic", default=_JACK) == "my-mic"


def test_choose_mic_falls_back_to_default_when_no_usb() -> None:
    assert choose_mic([_JACK], explicit=None, default=_JACK) == _JACK


def test_choose_mic_keeps_default_when_default_is_already_usb() -> None:
    assert choose_mic([_JACK, _USB], explicit=None, default=_USB) == _USB


# --- capture modes (robust-capture.AC2) ---


def test_build_capture_command_mic_only_is_single_input_mono() -> None:
    cmd = build_capture_command("MIC", None, Path("/out/m.flac"))

    # One PulseAudio input, no monitor, no two-source mixing.
    assert cmd.count("pulse") == 1
    assert "MON" not in cmd
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "join=" not in graph
    assert "amix=" not in graph
    assert "[them]" not in graph
    assert cmd[cmd.index("-c:a") + 1] == "flac"
    assert cmd[-1] == "/out/m.flac"


def test_resolve_record_mode_flags_win_and_default_is_remote() -> None:
    assert resolve_record_mode(remote=True, in_person=False, prompt_answer=None) is True
    assert resolve_record_mode(remote=False, in_person=True, prompt_answer=None) is False
    # Non-interactive (no answer), no flag -> remote, preserving today's behaviour.
    assert resolve_record_mode(remote=False, in_person=False, prompt_answer=None) is True
    # Interactive answer: anything starting with "i" is in-person, else remote.
    assert resolve_record_mode(remote=False, in_person=False, prompt_answer="i") is False
    assert resolve_record_mode(remote=False, in_person=False, prompt_answer="In person") is False
    assert resolve_record_mode(remote=False, in_person=False, prompt_answer="r") is True


def test_resolve_record_mode_rejects_conflicting_flags() -> None:
    with pytest.raises(CaptureError):
        resolve_record_mode(remote=True, in_person=True, prompt_answer=None)


# --- silence guard (robust-capture.AC3) ---

# Verbatim shapes emitted live by ffmpeg's silencedetect filter (captured from a
# real 4s run); a progress line is the decoy the parser must ignore.


def test_parse_silence_start_event() -> None:
    assert parse_silence_events("[silencedetect @ 0x55] silence_start: 2.58754") == SilenceEvent(
        "start", 2.58754
    )


def test_parse_silence_end_event_with_duration() -> None:
    line = "[silencedetect @ 0x55] silence_end: 4.0725 | silence_duration: 1.48496"
    assert parse_silence_events(line) == SilenceEvent("end", 4.0725, 1.48496)


def test_parse_silence_ignores_progress_lines() -> None:
    assert parse_silence_events("size=N/A time=00:00:03.16 bitrate=N/A speed=1.01x") is None


def test_silence_tracker_warns_on_start_then_notes_resume() -> None:
    tracker = SilenceTracker("USB_MIC")

    warning = tracker.observe("[silencedetect @ 0x1] silence_start: 1.0")
    assert warning is not None and "USB_MIC" in warning

    assert tracker.observe("size=N/A time=2 bitrate=N/A") is None  # progress: nothing

    resume = tracker.observe("[silencedetect @ 0x1] silence_end: 30.0 | silence_duration: 29.0")
    assert resume is not None and "resum" in resume.lower()


def test_silence_tracker_rate_limits_warnings_and_resume_notes() -> None:
    tracker = SilenceTracker("M", max_warnings=2)

    messages: list[str] = []
    for i in range(5):
        for line in (f"silence_start: {i}.0", f"silence_end: {i}.5 | silence_duration: 0.5"):
            message = tracker.observe(line)
            if message is not None:
                messages.append(message)

    # Capped at 2: a stretch past the cap emits neither a warning nor a resume note,
    # so a flapping channel can spam neither.
    assert len([m for m in messages if "no signal" in m]) == 2
    assert len([m for m in messages if "resumed" in m]) == 2


def test_build_capture_command_mic_leg_carries_silence_guard() -> None:
    cmd = build_capture_command("MIC", "MON", Path("/o.flac"))
    graph = cmd[cmd.index("-filter_complex") + 1]
    branches = graph.split(";")

    assert branches[0].startswith("[0:a]")  # the mic leg
    assert "silencedetect=noise=-50dB:d=20" in branches[0]
    assert "silencedetect" not in branches[1]  # not the monitor leg


def test_build_capture_command_mic_only_carries_silence_guard() -> None:
    cmd = build_capture_command("MIC", None, Path("/o.flac"))
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "silencedetect=noise=-50dB:d=20" in graph


def test_capture_refuses_existing_output_before_opening_devices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "existing.flac"
    target.write_bytes(b"previous recording")

    def unexpected_resolution(explicit: str | None) -> str:
        pytest.fail("existing output must be refused before resolving devices")

    monkeypatch.setattr("localscribe.capture.resolve_mic", unexpected_resolution)
    with pytest.raises(CaptureError, match="already exists"):
        capture(target)
    assert target.read_bytes() == b"previous recording"
