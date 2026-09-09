"""NeMo configuration at the windowed inference boundary, without loading CUDA."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from localscribe.models import Word
from localscribe.transcribe import transcribe


def test_repeated_windows_disable_decoder_graphs_and_keep_word_timestamps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[bool, bool]] = []

    class Model:
        def __init__(self) -> None:
            self.cfg = SimpleNamespace(
                decoding=SimpleNamespace(greedy=SimpleNamespace(max_symbols_per_step=10))
            )
            self.graphs_enabled = True

        def change_attention_model(self, *args: object) -> None:
            pass

        def change_decoding_strategy(self, config: SimpleNamespace, **kwargs: object) -> None:
            self.graphs_enabled = config.greedy.use_cuda_graph_decoder
            assert config.greedy.max_symbols_per_step == 10

        def transcribe(self, paths: list[str], *, timestamps: bool) -> list[SimpleNamespace]:
            calls.append((self.graphs_enabled, timestamps))
            stamp = {"word": "hello", "start": 0.1, "end": 0.5}
            return [SimpleNamespace(timestamp={"word": [stamp]})]

    model = Model()
    nemo = ModuleType("nemo")
    collections = ModuleType("nemo.collections")
    asr = ModuleType("nemo.collections.asr")
    monkeypatch.setattr(nemo, "collections", collections, raising=False)
    monkeypatch.setattr(collections, "asr", asr, raising=False)
    monkeypatch.setattr(
        asr, "models",
        SimpleNamespace(ASRModel=SimpleNamespace(from_pretrained=lambda **kw: model)),
        raising=False,
    )
    for module in (nemo, collections, asr):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    omega = ModuleType("omegaconf")
    monkeypatch.setattr(omega, "open_dict", nullcontext, raising=False)
    monkeypatch.setitem(sys.modules, "omegaconf", omega)
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    windows = [(0.0, chunk_dir / "first.wav"), (1200.0, chunk_dir / "second.wav")]
    monkeypatch.setattr("localscribe.transcribe.split_into_windows", lambda *args: windows)
    monkeypatch.setattr("localscribe.transcribe.free_cuda", lambda: None)

    words = transcribe(tmp_path / "meeting.wav")

    assert calls == [(False, True), (False, True)]
    assert words == [Word("hello", 0.1, 0.5), Word("hello", 1200.1, 1200.5)]
    assert not chunk_dir.exists()
