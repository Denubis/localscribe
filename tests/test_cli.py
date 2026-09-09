"""CLI boundary tests that do not touch the heavy model stages.

Path validation happens before any pyannote/NeMo import, so these run fast.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from localscribe.cli import app, enrol_app, record_app, send_app, watch_app
from localscribe.errors import TranscriptionError

runner = CliRunner()


@pytest.mark.parametrize("mode", ["remote", "in-person"])
def test_interactive_record_uses_continuous_session(monkeypatch, tmp_path, mode) -> None:
    import sys

    seen = {}

    def session(target, **kwargs):
        seen.update(target=target, **kwargs)
        return 0

    def old_capture(*args, **kwargs):
        raise AssertionError("interactive recording must support rollover")

    monkeypatch.setitem(sys.modules, "localscribe.record_session", SimpleNamespace(
        record_interactively=session,
    ))
    monkeypatch.setattr("localscribe.capture.resolve_mic", lambda explicit: "USB_MIC")
    monkeypatch.setattr("localscribe.capture.capture", old_capture)
    # CliRunner supplies its own stdin object; patch the CLI's terminal query directly.
    monkeypatch.setattr("localscribe.cli._interactive_input", lambda: True, raising=False)
    target = tmp_path / "meeting.flac"

    result = runner.invoke(record_app, [str(target), f"--{mode}", "--no-transcribe"])

    assert result.exit_code == 0, result.output
    assert seen["target"] == target
    assert seen["remote"] is (mode == "remote")
    assert seen["mic"] == "USB_MIC"
    assert seen["no_transcribe"] is True
    assert "Enter" in result.output


def test_scribe_reports_missing_file_and_exits_nonzero(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("localscribe.runtime.find_spec", lambda name: None)
    missing = tmp_path / "nope.flac"

    result = runner.invoke(app, [str(missing)])

    assert result.exit_code == 2
    assert "not found" in result.output.lower()


def test_scribe_with_no_arguments_exits_nonzero() -> None:
    result = runner.invoke(app, [])

    assert result.exit_code != 0


def test_enrol_reports_missing_sample(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("localscribe.runtime.find_spec", lambda name: None)
    result = runner.invoke(enrol_app, [str(tmp_path / "nope.wav"), "--name", "Alice"])

    assert result.exit_code == 2
    assert "not found" in result.output.lower()


def test_watch_reports_missing_destination_before_loading_models(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("localscribe.runtime.find_spec", lambda name: None)
    result = runner.invoke(watch_app, [str(tmp_path / "absent"), "--once"])

    assert result.exit_code == 2
    assert "destination directory does not exist" in result.output


def test_record_reports_missing_output_directory_before_recording(tmp_path) -> None:
    # Parent dir absent: must exit (without ever launching ffmpeg/wpctl).
    target = tmp_path / "no_such_dir" / "out.flac"

    result = runner.invoke(record_app, [str(target)])

    assert result.exit_code != 0
    assert "directory" in result.output.lower()


@pytest.mark.parametrize(
    ("flags", "remote", "mono", "label"),
    [
        (["--in-person"], False, False, "in-person (mic only, mono)"),
        (["--remote"], True, False, "remote (mic + system, stereo)"),
        (["--remote", "--mono"], True, True, "remote (mic + system, mono)"),
    ],
)
@pytest.mark.parametrize("command", [record_app, send_app], ids=["record", "send"])
def test_record_mode_reaches_capture(
    monkeypatch, tmp_path, flags, remote, mono, label, command
) -> None:
    # Capture owns the in-person mono policy; the CLI forwards the mode and chosen mic.
    seen: dict[str, object] = {}

    def fake_capture(output, *, mic, mono, remote):
        seen["mic"], seen["remote"], seen["mono"] = mic, remote, mono
        return output

    monkeypatch.setattr("localscribe.capture.resolve_mic", lambda explicit: "USB_MIC")
    monkeypatch.setattr("localscribe.capture.capture", fake_capture)

    # --no-transcribe keeps this focused on capture wiring (record auto-transcribes
    # by default; that path is covered by its own tests below).
    out = tmp_path / "m.flac"
    if command is record_app:
        args = [str(out), *flags, "--no-transcribe"]
    else:
        monkeypatch.setattr("localscribe.send.staging_dir", lambda: tmp_path)
        monkeypatch.setattr("localscribe.send.push", lambda *a, **kw: "inbox/m.flac")
        args = [*flags, "--host", "test-host"]
    result = runner.invoke(command, args)

    assert result.exit_code == 0, result.output
    assert seen == {"mic": "USB_MIC", "remote": remote, "mono": mono}
    assert "USB_MIC" in result.output
    assert label in result.output


def test_scribe_delegates_to_pipeline_for_each_recording(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("localscribe.cli.require_gpu_extra", lambda: None)
    a, b = tmp_path / "a.flac", tmp_path / "b.flac"
    a.write_bytes(b"")
    b.write_bytes(b"")
    seen: list[Path] = []
    monkeypatch.setattr(
        "localscribe.pipeline.transcribe_recording", lambda rec, **kw: seen.append(rec)
    )
    monkeypatch.setattr("localscribe.cli.require_gpu_extra", lambda: None)

    result = runner.invoke(app, [str(a), str(b)])

    assert result.exit_code == 0, result.output
    assert seen == [a, b]


def test_scribe_reports_pipeline_failure_and_exits_nonzero(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("localscribe.cli.require_gpu_extra", lambda: None)
    rec = tmp_path / "a.flac"
    rec.write_bytes(b"")

    def boom(recording, **kw):
        raise TranscriptionError("model fell over")

    monkeypatch.setattr("localscribe.pipeline.transcribe_recording", boom)
    monkeypatch.setattr("localscribe.cli.require_gpu_extra", lambda: None)

    result = runner.invoke(app, [str(rec)])

    assert result.exit_code == 1
    assert "model fell over" in result.output


@pytest.mark.parametrize("mode", ["--remote", "--in-person"])
def test_record_auto_transcribes_and_prints_md_path(monkeypatch, tmp_path, mode) -> None:
    monkeypatch.setattr("localscribe.cli.require_gpu_extra", lambda: None)
    out = tmp_path / "m.flac"

    def fake_capture(output, *, mic, mono, remote):
        Path(output).write_bytes(b"flac")
        return Path(output)

    transcribed: dict[str, Path] = {}

    def fake_transcribe(recording, **kw):
        transcribed["rec"] = recording
        recording.with_suffix(".md").write_text("# transcript")
        recording.with_suffix(".json").write_text("{}")

    monkeypatch.setattr("localscribe.capture.resolve_mic", lambda explicit: "MIC")
    monkeypatch.setattr("localscribe.capture.capture", fake_capture)
    monkeypatch.setattr("localscribe.pipeline.transcribe_recording", fake_transcribe)
    monkeypatch.setattr("localscribe.cli.require_gpu_extra", lambda: None)

    result = runner.invoke(record_app, [str(out), mode])

    assert result.exit_code == 0, result.output
    assert transcribed["rec"] == out
    assert out.with_suffix(".md").exists()
    assert out.with_suffix(".json").exists()
    assert str(out.with_suffix(".md")) in result.output


def test_record_no_transcribe_keeps_only_the_flac(monkeypatch, tmp_path) -> None:
    out = tmp_path / "m.flac"

    def fake_capture(output, *, mic, mono, remote):
        Path(output).write_bytes(b"flac")
        return Path(output)

    def must_not_run(*args, **kwargs):
        raise AssertionError("transcription must not run under --no-transcribe")

    monkeypatch.setattr("localscribe.capture.resolve_mic", lambda explicit: "MIC")
    monkeypatch.setattr("localscribe.capture.capture", fake_capture)
    monkeypatch.setattr("localscribe.pipeline.transcribe_recording", must_not_run)

    result = runner.invoke(record_app, [str(out), "--remote", "--no-transcribe"])

    assert result.exit_code == 0, result.output
    assert out.exists()
    assert not out.with_suffix(".md").exists()


@pytest.mark.parametrize("mode", ["--remote", "--in-person"])
def test_record_keeps_flac_and_points_to_scribe_when_transcription_fails(
    monkeypatch, tmp_path, mode
) -> None:
    monkeypatch.setattr("localscribe.cli.require_gpu_extra", lambda: None)
    out = tmp_path / "m.flac"

    def fake_capture(output, *, mic, mono, remote):
        Path(output).write_bytes(b"flac")
        return Path(output)

    def boom(recording, **kw):
        raise TranscriptionError("CUDA OOM")

    monkeypatch.setattr("localscribe.capture.resolve_mic", lambda explicit: "MIC")
    monkeypatch.setattr("localscribe.capture.capture", fake_capture)
    monkeypatch.setattr("localscribe.pipeline.transcribe_recording", boom)

    result = runner.invoke(record_app, [str(out), mode])

    assert result.exit_code == 1
    assert out.read_bytes() == b"flac"
    assert "transcription failed: CUDA OOM" in result.output
    assert f"scribe {out}" in result.output


@pytest.mark.parametrize("mode", ["--remote", "--in-person"])
def test_record_without_model_stack_preserves_flac_and_explains_retry(
    monkeypatch, tmp_path, mode
) -> None:
    out = tmp_path / "m.flac"

    def fake_capture(output, **kwargs):
        output.write_bytes(b"flac")
        return output

    monkeypatch.setattr("localscribe.runtime.find_spec", lambda name: None)
    monkeypatch.setattr("localscribe.capture.resolve_mic", lambda explicit: "MIC")
    monkeypatch.setattr("localscribe.capture.capture", fake_capture)

    result = runner.invoke(record_app, [str(out), mode])

    assert result.exit_code == 3
    assert out.read_bytes() == b"flac"
    assert "./install.sh --gpu" in result.output
    assert f"scribe {out}" in result.output


@pytest.mark.parametrize("command", [app, enrol_app, watch_app], ids=["scribe", "enrol", "watch"])
def test_transcription_commands_explain_missing_model_stack(monkeypatch, tmp_path, command) -> None:
    monkeypatch.setattr("localscribe.runtime.find_spec", lambda name: None)
    sample = tmp_path / "m.flac"
    sample.write_bytes(b"flac")
    args = [str(tmp_path), "--once"] if command is watch_app else [str(sample)]
    if command is enrol_app:
        args.extend(["--name", "Alice"])

    result = runner.invoke(command, args)

    assert result.exit_code == 3
    assert "./install.sh --gpu" in result.output
