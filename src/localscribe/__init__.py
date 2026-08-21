"""localscribe — local speaker-attributed transcription.

FLAC meeting recording in, a speaker-labelled transcript out. Transcription runs
locally: audio is never sent to a public-cloud transcription service. ``scribe-send``
can transfer a recording over SSH to a transcription host the user explicitly chooses.

No import side effects here: importing the package must stay cheap and pure so the
functional core (reconciliation, rendering) is testable without heavy deps. Cache
configuration and model loading live in the imperative shell.
"""

__version__ = "0.1.0"
