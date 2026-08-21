# pattern: Functional Core
"""Typed error hierarchy.

A single base lets the CLI catch everything localscribe raises and turn it into a
clean message + exit code, while the subclasses say which stage failed.
"""

from __future__ import annotations


class LocalscribeError(Exception):
    """Base for every error localscribe raises deliberately."""


class AudioNotFoundError(LocalscribeError):
    """A requested recording does not exist or is not a file."""


class AudioError(LocalscribeError):
    """Audio could not be read or preprocessed (e.g. ffmpeg failure)."""


class DiarizationError(LocalscribeError):
    """The diarisation stage (pyannote) failed to load or run."""


class TranscriptionError(LocalscribeError):
    """The transcription stage (NeMo/Parakeet) failed to load or run."""


class EnrolmentError(LocalscribeError):
    """Voice enrolment failed (bad name, unreadable sample, embedding failure)."""


class CaptureError(LocalscribeError):
    """Live capture failed (no audio server, missing tool, ffmpeg/wpctl failure)."""


class MissingExtraError(LocalscribeError):
    """A transcription command was run on an install without the ``gpu`` extra.

    Recording and sending need only ffmpeg, so a laptop installs localscribe
    without the model stack. Hitting this means a transcribing command ran there.
    """
