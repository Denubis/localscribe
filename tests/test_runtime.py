"""Runtime-boundary tests that stay independent of the optional GPU stack."""

from pathlib import Path

import pytest

from localscribe import runtime
from localscribe.errors import MissingExtraError


def test_localscribe_cache_override_wins(monkeypatch, tmp_path) -> None:
    chosen = tmp_path / "models"
    monkeypatch.setenv("LOCALSCRIBE_HF_HOME", str(chosen))
    monkeypatch.setenv("HF_HOME", "/standard/huggingface")

    runtime.configure_cache()

    assert runtime.os.environ["HF_HOME"] == str(chosen)


def test_standard_hf_home_is_preserved_without_localscribe_override(monkeypatch) -> None:
    monkeypatch.delenv("LOCALSCRIBE_HF_HOME", raising=False)
    monkeypatch.setenv("HF_HOME", "/standard/huggingface")

    runtime.configure_cache()

    assert runtime.os.environ["HF_HOME"] == "/standard/huggingface"


def test_cache_defaults_to_xdg_when_no_explicit_home(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("LOCALSCRIBE_HF_HOME", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    runtime.configure_cache()

    assert runtime.os.environ["HF_HOME"] == str(tmp_path / "huggingface")


def test_missing_gpu_extra_points_to_the_supported_installer(monkeypatch) -> None:
    monkeypatch.setattr(runtime, "find_spec", lambda _name: None)

    with pytest.raises(MissingExtraError) as caught:
        runtime.require_gpu_extra()

    message = str(caught.value)
    assert "./install.sh --gpu" in message
    assert "uv tool install" not in message


def test_default_cache_uses_the_user_cache_directory(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("LOCALSCRIBE_HF_HOME", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    runtime.configure_cache()

    assert Path(runtime.os.environ["HF_HOME"]) == tmp_path / ".cache" / "huggingface"


@pytest.mark.parametrize(
    ("key", "ambient", "disabled"),
    [
        ("PYANNOTE_METRICS_ENABLED", "true", "false"),
        ("HF_HUB_DISABLE_TELEMETRY", "0", "1"),
        ("OTEL_SDK_DISABLED", "false", "true"),
        ("WANDB_MODE", "online", "disabled"),
    ],
)
def test_runtime_disables_telemetry_despite_ambient_opt_in(
    monkeypatch, key, ambient, disabled
) -> None:
    monkeypatch.setenv(key, ambient)

    runtime.quiet_third_party()

    assert runtime.os.environ[key] == disabled
