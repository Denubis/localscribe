"""Tests for the watcher's claim/deliver/transcribe cycle.

These use a real temporary directory but never a real model: `transcribe` is
injected, so the whole cycle is exercised without a GPU. The filesystem is the
thing under test here — claiming *is* an ``os.rename``, and a test that faked it
would be testing nothing.

Not covered, and verified by hand instead: that the flock actually excludes a
second process (needs two processes and a 24 GB model to be meaningful), and that
rsync's rename lands the way `is_ready` assumes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from localscribe.errors import TranscriptionError
from localscribe.inbox import CLAIMED_DIR
from localscribe.watch import claim, deliver, orphaned_claims, run_once, unique_name


@pytest.fixture
def inbox(tmp_path: Path) -> Path:
    box = tmp_path / "inbox"
    box.mkdir()
    return box


@pytest.fixture
def destination(tmp_path: Path) -> Path:
    dest = tmp_path / "recordings"
    dest.mkdir()
    return dest


def _recording(inbox: Path, name: str, content: bytes = b"FLAC-ish") -> Path:
    path = inbox / name
    path.write_bytes(content)
    return path


class TestClaim:
    def test_the_winner_gets_the_file_out_of_the_inbox(self, inbox: Path) -> None:
        _recording(inbox, "meeting-1.flac")
        claimed = claim(inbox, "meeting-1.flac", "a1b2c3")
        assert claimed is not None
        assert claimed.read_bytes() == b"FLAC-ish"
        assert not (inbox / "meeting-1.flac").exists()

    def test_only_one_of_two_watchers_can_claim_the_same_recording(self, inbox: Path) -> None:
        _recording(inbox, "meeting-1.flac")
        first = claim(inbox, "meeting-1.flac", "aaaaaa")
        second = claim(inbox, "meeting-1.flac", "bbbbbb")
        assert first is not None
        assert second is None

    def test_claiming_something_already_gone_is_not_an_error(self, inbox: Path) -> None:
        assert claim(inbox, "never-existed.flac", "a1b2c3") is None

    def test_a_claim_is_invisible_to_a_watcher_rescanning_the_inbox(self, inbox: Path) -> None:
        _recording(inbox, "meeting-1.flac")
        claim(inbox, "meeting-1.flac", "a1b2c3")
        assert [p.name for p in inbox.iterdir() if not p.name.startswith(".")] == []


class TestUniqueName:
    def test_an_unused_name_is_left_alone(self) -> None:
        assert unique_name("meeting-1.flac", lambda _: False) == "meeting-1.flac"

    def test_a_collision_is_suffixed_rather_than_overwritten(self) -> None:
        taken = {"meeting-1.flac"}
        assert unique_name("meeting-1.flac", lambda n: n in taken) == "meeting-1-2.flac"

    def test_the_suffix_keeps_climbing_past_repeated_collisions(self) -> None:
        taken = {"meeting-1.flac", "meeting-1-2.flac", "meeting-1-3.flac"}
        assert unique_name("meeting-1.flac", lambda n: n in taken) == "meeting-1-4.flac"


class TestDeliver:
    def test_the_recording_lands_in_the_destination_under_its_original_name(
        self, inbox: Path, destination: Path
    ) -> None:
        _recording(inbox, "meeting-20260801-150703.flac")
        claimed = claim(inbox, "meeting-20260801-150703.flac", "a1b2c3")
        assert claimed is not None
        final = deliver(claimed, destination)
        assert final == destination / "meeting-20260801-150703.flac"
        assert final.read_bytes() == b"FLAC-ish"

    def test_an_existing_recording_of_the_same_name_is_not_clobbered(
        self, inbox: Path, destination: Path
    ) -> None:
        (destination / "meeting-1.flac").write_bytes(b"the older one")
        _recording(inbox, "meeting-1.flac", b"the newer one")
        claimed = claim(inbox, "meeting-1.flac", "a1b2c3")
        assert claimed is not None
        final = deliver(claimed, destination)
        assert (destination / "meeting-1.flac").read_bytes() == b"the older one"
        assert final.read_bytes() == b"the newer one"


class TestOrphanedClaims:
    def test_an_empty_claim_area_reports_nothing(self, inbox: Path) -> None:
        assert orphaned_claims(inbox) == []

    def test_a_claim_left_by_a_killed_watcher_is_reported(self, inbox: Path) -> None:
        _recording(inbox, "meeting-1.flac")
        claim(inbox, "meeting-1.flac", "a1b2c3")
        assert orphaned_claims(inbox) == ["a1b2c3-meeting-1.flac"]


class TestRunOnce:
    def test_an_empty_inbox_is_a_no_op(self, inbox: Path, destination: Path) -> None:
        assert run_once(inbox, destination, transcribe=lambda _: None) is None

    def test_the_delivered_recording_is_what_gets_transcribed(
        self, inbox: Path, destination: Path
    ) -> None:
        _recording(inbox, "meeting-1.flac")
        seen: list[Path] = []
        final = run_once(inbox, destination, transcribe=seen.append)
        assert final == destination / "meeting-1.flac"
        assert seen == [destination / "meeting-1.flac"]

    def test_a_recording_the_filter_excludes_is_left_for_another_watcher(
        self, inbox: Path, destination: Path
    ) -> None:
        _recording(inbox, "project-b-1.flac")
        assert (
            run_once(inbox, destination, needle="project-a", transcribe=lambda _: None)
            is None
        )
        assert (inbox / "project-b-1.flac").exists()

    def test_a_transcription_failure_still_leaves_the_audio_in_the_destination(
        self, inbox: Path, destination: Path
    ) -> None:
        _recording(inbox, "meeting-1.flac")

        def explode(_: Path) -> None:
            raise TranscriptionError("CUDA out of memory")

        with pytest.raises(TranscriptionError):
            run_once(inbox, destination, transcribe=explode)
        assert (destination / "meeting-1.flac").read_bytes() == b"FLAC-ish"

    def test_a_failed_recording_is_not_left_stranded_in_the_claim_area(
        self, inbox: Path, destination: Path
    ) -> None:
        _recording(inbox, "meeting-1.flac")

        def explode(_: Path) -> None:
            raise TranscriptionError("CUDA out of memory")

        with pytest.raises(TranscriptionError):
            run_once(inbox, destination, transcribe=explode)
        assert orphaned_claims(inbox) == []

    def test_a_failed_recording_is_not_picked_up_again_on_the_next_pass(
        self, inbox: Path, destination: Path
    ) -> None:
        # Guardrail 2 cuts both ways: never lose the audio, but never loop on a
        # poison file either. Once delivered, it is out of the inbox for good.
        _recording(inbox, "meeting-1.flac")

        def explode(_: Path) -> None:
            raise TranscriptionError("CUDA out of memory")

        with pytest.raises(TranscriptionError):
            run_once(inbox, destination, transcribe=explode)
        assert run_once(inbox, destination, transcribe=lambda _: None) is None

    def test_the_claim_area_is_not_mistaken_for_a_queued_recording(
        self, inbox: Path, destination: Path
    ) -> None:
        (inbox / CLAIMED_DIR).mkdir()
        (inbox / CLAIMED_DIR / "someone-elses.flac").write_bytes(b"in flight")
        assert run_once(inbox, destination, transcribe=lambda _: None) is None
