"""Registry tests for the voice store (save/load embeddings). No model involved."""

import stat

import pytest

from localscribe.errors import EnrolmentError
from localscribe.voices import load_voices, save_voice


def test_save_and_load_voice_roundtrip(tmp_path) -> None:
    save_voice("Alice", [0.1, 0.2, 0.3], directory=tmp_path)
    save_voice("Bob", [0.4, 0.5, 0.6], directory=tmp_path)

    assert load_voices(directory=tmp_path) == {
        "Alice": [0.1, 0.2, 0.3],
        "Bob": [0.4, 0.5, 0.6],
    }


def test_load_voices_from_missing_dir_is_empty(tmp_path) -> None:
    assert load_voices(directory=tmp_path / "does-not-exist") == {}


def test_resaving_a_name_overwrites(tmp_path) -> None:
    save_voice("Alice", [1.0, 0.0], directory=tmp_path)
    save_voice("Alice", [0.0, 1.0], directory=tmp_path)

    assert load_voices(directory=tmp_path) == {"Alice": [0.0, 1.0]}


def test_save_voice_rejects_unsafe_name(tmp_path) -> None:
    with pytest.raises(EnrolmentError):
        save_voice("../evil", [0.1], directory=tmp_path)


def test_voice_registry_is_private_even_when_parent_started_permissive(tmp_path) -> None:
    registry = tmp_path / "voices"
    registry.mkdir(mode=0o755)

    path = save_voice("Alice", [0.1, 0.2], directory=registry)

    assert stat.S_IMODE(registry.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_resaving_a_voice_tightens_an_existing_permissive_file(tmp_path) -> None:
    path = save_voice("Alice", [0.1], directory=tmp_path)
    path.chmod(0o644)

    save_voice("Alice", [0.2], directory=tmp_path)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
