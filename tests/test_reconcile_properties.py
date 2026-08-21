"""Property tests for reconciliation invariants (Hypothesis).

The headline invariant encodes guardrail #2: whatever the diarisation looks like,
every transcribed word survives reconciliation, in its original order.
"""

from hypothesis import given
from hypothesis import strategies as st

from localscribe.models import SpeakerTurn, Word
from localscribe.reconcile import reconcile

# Tokens never contain spaces, so a segment's joined text splits back to its words.
_token = st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=6)
_time = st.floats(min_value=0.0, max_value=1e4, allow_nan=False, allow_infinity=False)
_words = st.lists(st.builds(Word, text=_token, start=_time, end=_time), max_size=50)
_speaker = st.sampled_from(["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"])
_turns = st.lists(st.builds(SpeakerTurn, speaker=_speaker, start=_time, end=_time), max_size=20)


@given(words=_words, turns=_turns)
def test_reconcile_never_drops_or_reorders_words(
    words: list[Word], turns: list[SpeakerTurn]
) -> None:
    segments = reconcile(words, turns)
    recovered = [token for segment in segments for token in segment.text.split(" ")]
    assert recovered == [word.text for word in words]


@given(words=_words, turns=_turns)
def test_every_segment_speaker_is_a_real_speaker_or_unknown(
    words: list[Word], turns: list[SpeakerTurn]
) -> None:
    segments = reconcile(words, turns)
    allowed = {turn.speaker for turn in turns} | {"UNKNOWN"}
    assert all(segment.speaker in allowed for segment in segments)
