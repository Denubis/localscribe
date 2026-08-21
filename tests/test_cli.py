"""CLI boundary tests that do not touch the heavy model stages.

Path validation happens before any pyannote/NeMo import, so these run fast.
"""

from pathlib import Path

from typer.testing import CliRunner

from localscribe.cli import app, enrol_app, record_app
from localscribe.errors import TranscriptionError

runner = CliRunner()


def test_scribe_reports_missing_file_and_exits_nonzero(tmp_path) -> None:
    missing = tmp_path / "nope.flac"

    result = runner.invoke(app, [str(missing)])

    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_scribe_with_no_arguments_exits_nonzero() -> None:
    result = runner.invoke(app, [])

    assert result.exit_code != 0


def test_enrol_reports_missing_sample(tmp_path) -> None:
    result = runner.invoke(enrol_app, [str(tmp_path / "nope.wav"), "--name", "Alice"])

    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_record_reports_missing_output_directory_before_recording(tmp_path) -> None:
    # Parent dir absent: must exit (without ever launching ffmpeg/wpctl).
    target = tmp_path / "no_such_dir" / "out.flac"

    result = runner.invoke(record_app, [str(target)])

    assert result.exit_code != 0
    assert "directory" in result.output.lower()


def test_record_in_person_flag_records_mono_mic_only(monkeypatch, tmp_path) -> None:
    # --in-person must reach capture() as remote=False, using the policy-chosen mic,
    # which is echoed at startup. ffmpeg/wpctl are stubbed so no audio server is touched.
    seen: dict[str, object] = {}

    def fake_capture(output, *, mic, mono, remote):
        seen["mic"], seen["remote"] = mic, remote
        return output

    monkeypatch.setattr("localscribe.capture.resolve_mic", lambda explicit: "USB_MIC")
    monkeypatch.setattr("localscribe.capture.capture", fake_capture)

    # --no-transcribe keeps this focused on capture wiring (record auto-transcribes
    # by default; that path is covered by its own tests below).
    out = tmp_path / "m.flac"
    result = runner.invoke(record_app, [str(out), "--in-person", "--no-transcribe"])

    assert result.exit_code == 0, result.output
    assert seen == {"mic": "USB_MIC", "remote": False}
    assert "USB_MIC" in result.output


def test_scribe_delegates_to_pipeline_for_each_recording(monkeypatch, tmp_path) -> None:
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
    rec = tmp_path / "a.flac"
    rec.write_bytes(b"")

    def boom(recording, **kw):
        raise TranscriptionError("model fell over")

    monkeypatch.setattr("localscribe.pipeline.transcribe_recording", boom)
    monkeypatch.setattr("localscribe.cli.require_gpu_extra", lambda: None)

    result = runner.invoke(app, [str(rec)])

    assert result.exit_code == 1
    assert "model fell over" in result.output


def test_record_auto_transcribes_and_prints_md_path(monkeypatch, tmp_path) -> None:
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

    result = runner.invoke(record_app, [str(out), "--in-person"])

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


def test_record_keeps_flac_and_points_to_scribe_when_transcription_fails(
    monkeypatch, tmp_path
) -> None:
    out = tmp_path / "m.flac"

    def fake_capture(output, *, mic, mono, remote):
        Path(output).write_bytes(b"flac")
        return Path(output)

    def boom(recording, **kw):
        raise TranscriptionError("CUDA OOM")

    monkeypatch.setattr("localscribe.capture.resolve_mic", lambda explicit: "MIC")
    monkeypatch.setattr("localscribe.capture.capture", fake_capture)
    monkeypatch.setattr("localscribe.pipeline.transcribe_recording", boom)

    result = runner.invoke(record_app, [str(out), "--remote"])

    assert result.exit_code != 0
    assert out.exists()  # the recording is never lost to a failed transcription
    assert "scribe" in result.output.lower()  # told how to retry
