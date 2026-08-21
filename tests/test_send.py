"""Pure-core tests for pushing a recording to the transcription box.

rsync and ssh are the external boundary and stay in send.py's shell. What is
pinned here is the part that decides whether a transfer is trustworthy, because a
verification step that cannot fail is worse than none: it reports success for a
truncated file just as readily as for a good one.
"""

from __future__ import annotations

import pytest

from localscribe.errors import CaptureError
from localscribe.send import (
    build_checksum_command,
    build_mkdir_command,
    confirm_transfer,
    parse_sha256,
    remote_file_path,
)

DIGEST = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


class TestParseSha256:
    def test_takes_the_digest_from_sha256sums_two_column_output(self) -> None:
        assert parse_sha256(f"{DIGEST}  .local/share/localscribe/inbox/x.flac\n") == DIGEST

    def test_a_path_containing_spaces_does_not_confuse_the_split(self) -> None:
        assert parse_sha256(f"{DIGEST}  a path/with spaces.flac\n") == DIGEST

    def test_empty_output_raises_rather_than_reporting_a_match(self) -> None:
        # An ssh that failed silently must not look like a verified transfer.
        with pytest.raises(CaptureError):
            parse_sha256("")

    def test_an_error_message_where_a_digest_should_be_raises(self) -> None:
        with pytest.raises(CaptureError):
            parse_sha256("sha256sum: x.flac: No such file or directory\n")

    def test_something_hex_but_the_wrong_length_raises(self) -> None:
        with pytest.raises(CaptureError):
            parse_sha256("deadbeef  x.flac\n")


class TestConfirmTransfer:
    def test_matching_digests_pass_quietly(self) -> None:
        confirm_transfer(DIGEST, f"{DIGEST}  x.flac\n")

    def test_a_differing_digest_raises(self) -> None:
        other = "0" * 64
        with pytest.raises(CaptureError):
            confirm_transfer(DIGEST, f"{other}  x.flac\n")

    def test_unreadable_remote_output_raises_rather_than_passing(self) -> None:
        with pytest.raises(CaptureError):
            confirm_transfer(DIGEST, "")

    def test_the_comparison_is_not_fooled_by_case(self) -> None:
        confirm_transfer(DIGEST, f"{DIGEST.upper()}  x.flac\n")


class TestRemoteCommands:
    def test_the_inbox_is_created_before_the_push_so_rsync_cannot_fail_obscurely(self) -> None:
        argv = build_mkdir_command("gpu-box", ".local/share/localscribe/inbox")
        assert argv[:2] == ["ssh", "gpu-box"]
        assert "mkdir -p" in argv[2]
        assert ".local/share/localscribe/inbox" in argv[2]

    def test_a_remote_path_is_quoted_so_a_space_cannot_split_it(self) -> None:
        argv = build_checksum_command("host", "in box/x.flac")
        assert argv[2] == "sha256sum 'in box/x.flac'"

    def test_a_single_quote_in_a_path_cannot_break_out_of_the_quoting(self) -> None:
        argv = build_checksum_command("host", "o'brien.flac")
        assert argv[2] == "sha256sum 'o'\"'\"'brien.flac'"

    def test_the_remote_file_sits_directly_under_the_inbox(self) -> None:
        assert remote_file_path(".local/share/localscribe/inbox", "meeting-1.flac") == (
            ".local/share/localscribe/inbox/meeting-1.flac"
        )

    def test_a_trailing_slash_on_the_inbox_does_not_double_up(self) -> None:
        assert remote_file_path("inbox/", "meeting-1.flac") == "inbox/meeting-1.flac"
