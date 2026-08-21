"""Pure-core tests for the drop-box shared by `scribe-send` and `scribe-watch`.

These never touch the network or a real inbox: the rsync/ssh boundary lives in
send.py's shell and the claim rename in watch.py's, both exercised by hand. What
is pinned here is the logic worth pinning — which files a watcher is allowed to
see, which one it takes next, and the exact rsync argv that puts a recording in
front of it.

The expected values are written out from the contract (rsync's CLI, the
timestamped naming `scribe-record` already uses), never read back out of the code
under test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from localscribe.errors import CaptureError
from localscribe.inbox import (
    build_rsync_command,
    claim_name,
    is_ready,
    matches,
    remote_inbox_default,
    select_next,
)


class TestIsReady:
    """A watcher must never open a file that is still being written."""

    def test_a_plain_flac_is_ready(self) -> None:
        assert is_ready("meeting-20260801-150703.flac") is True

    def test_rsyncs_in_flight_temp_file_is_not_ready(self) -> None:
        # rsync writes `.<basename>.<random>` beside the target and renames on
        # completion, so a leading dot is the in-flight marker.
        assert is_ready(".meeting-20260801-150703.flac.Hx3kQa") is False

    def test_a_dotfile_that_ends_in_flac_is_still_not_ready(self) -> None:
        assert is_ready(".half-sent.flac") is False

    def test_a_partial_suffix_is_not_ready(self) -> None:
        assert is_ready("meeting-20260801-150703.flac.part") is False

    def test_a_transcript_sitting_in_the_inbox_is_not_a_recording(self) -> None:
        assert is_ready("meeting-20260801-150703.md") is False

    def test_the_case_of_the_extension_does_not_decide_readiness(self) -> None:
        assert is_ready("MEETING.FLAC") is True


class TestMatches:
    """`--match` lets two projects share one inbox without a manifest."""

    def test_no_filter_takes_everything(self) -> None:
        assert matches("meeting-20260801-150703.flac", None) is True

    def test_a_substring_of_the_name_matches(self) -> None:
        assert matches("project-a-20260801-150703.flac", "project-a") is True

    def test_a_name_without_the_substring_is_left_alone(self) -> None:
        assert matches("project-b-20260801-150703.flac", "project-a") is False

    def test_matching_ignores_case(self) -> None:
        assert matches("PROJECT-A-20260801.flac", "project-a") is True


class TestSelectNext:
    """Oldest first, so a queue drains in the order it was recorded."""

    def test_returns_none_when_the_inbox_is_empty(self) -> None:
        assert select_next([], None) is None

    def test_takes_the_oldest_by_modification_time(self) -> None:
        entries = [("newer.flac", 200.0), ("older.flac", 100.0)]
        assert select_next(entries, None) == "older.flac"

    def test_breaks_a_timestamp_tie_on_the_name_so_the_order_is_deterministic(self) -> None:
        entries = [("b.flac", 100.0), ("a.flac", 100.0)]
        assert select_next(entries, None) == "a.flac"

    def test_skips_files_that_are_not_ready(self) -> None:
        entries = [(".inflight.flac.Hx3kQa", 100.0), ("done.flac", 200.0)]
        assert select_next(entries, None) == "done.flac"

    def test_a_filter_can_leave_an_older_file_for_another_watcher(self) -> None:
        entries = [("project-b-1.flac", 100.0), ("project-a-1.flac", 200.0)]
        assert select_next(entries, "project-a") == "project-a-1.flac"

    def test_returns_none_when_the_filter_matches_nothing(self) -> None:
        assert select_next([("project-b-1.flac", 100.0)], "project-a") is None


class TestClaimName:
    """Two watchers in two folders race on one inbox; the rename decides the winner."""

    def test_the_claim_carries_the_token_and_the_original_name(self) -> None:
        assert claim_name("meeting-20260801-150703.flac", "a1b2c3") == (
            "a1b2c3-meeting-20260801-150703.flac"
        )

    def test_the_original_name_survives_a_round_trip(self) -> None:
        claimed = claim_name("project-a-20260801.flac", "deadbe")
        assert claimed.split("-", 1)[1] == "project-a-20260801.flac"

    def test_a_token_that_would_forge_a_path_is_rejected(self) -> None:
        with pytest.raises(CaptureError):
            claim_name("meeting.flac", "../../etc")


class TestRemoteInboxDefault:
    """The path is relative so the remote shell, not the sender, expands the home."""

    def test_is_relative_so_no_tilde_ever_reaches_a_remote_shell(self) -> None:
        default = remote_inbox_default()
        assert not default.startswith("~")
        assert not default.startswith("/")

    def test_lands_under_the_xdg_data_directory_localscribe_owns(self) -> None:
        assert remote_inbox_default() == ".local/share/localscribe/inbox"


class TestBuildRsyncCommand:
    def test_sends_the_file_into_the_remote_inbox_directory(self) -> None:
        argv = build_rsync_command(
            Path("/home/me/.cache/localscribe/meeting-20260801-150703.flac"),
            "gpu-box",
            ".local/share/localscribe/inbox",
        )
        assert argv[0] == "rsync"
        assert argv[-2] == "/home/me/.cache/localscribe/meeting-20260801-150703.flac"
        assert argv[-1] == "gpu-box:.local/share/localscribe/inbox/"

    def test_remote_recording_and_partial_directory_are_private(self) -> None:
        argv = build_rsync_command(Path("/tmp/x.flac"), "host", "inbox")
        assert "--chmod=F600,D700" in argv

    def test_an_interrupted_transfer_resumes_rather_than_restarting(self) -> None:
        argv = build_rsync_command(Path("/tmp/x.flac"), "host", "inbox")
        assert "--partial-dir=.partial" in argv

    def test_the_partial_directory_is_hidden_so_a_watcher_never_sees_a_partial(self) -> None:
        argv = build_rsync_command(Path("/tmp/x.flac"), "host", "inbox")
        partial = next(a for a in argv if a.startswith("--partial-dir="))
        assert is_ready(partial.split("=", 1)[1]) is False

    def test_arguments_are_protected_from_the_remote_shell(self) -> None:
        # The remote login shell here is fish; -s hands paths over the protocol
        # instead of letting a shell re-split them.
        argv = build_rsync_command(Path("/tmp/x.flac"), "host", "inbox")
        assert "-s" in argv
