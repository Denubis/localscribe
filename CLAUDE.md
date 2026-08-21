# LocalScribe

Last verified: 2026-08-21

## Purpose

Local Linux CLI for speaker-attributed transcription of meetings. Inference runs
on hardware the user controls. `scribe-send` may transfer audio over SSH to a
transcription host the user explicitly chooses; audio never goes to a
public-cloud transcription service.

## Tech stack

- Python 3.13, uv, Typer, pytest, Ruff, and ty.
- ASR: `nvidia/parakeet-tdt-0.6b-v3` via NeMo, with word timestamps.
- Diarisation: pyannote.audio 4.x with
  `pyannote/speaker-diarization-community-1`.
- Audio/capture: ffmpeg, PipeWire, WirePlumber.
- Transfer: OpenSSH and rsync.
- Known-working GPU: RTX 4090 with 24 GB VRAM. This is a reference, not a
  measured minimum.

## Commands

- `uv sync --locked` — install the light development environment.
- `uv run --frozen pytest` — run the complete automated suite without GPU deps.
- `uv run --frozen ruff check .` — lint.
- `uv run --frozen ty check` — type-check in the light environment.
- `uv build --no-sources` — build public distributions.
- `./install.sh` — install capture/send commands without CUDA.
- `./install.sh --gpu` — install the locked CUDA model stack.

## Architecture

- Functional core: `models`, `reconcile`, `render`, `parse`, and `identify`.
- Imperative shell: `cli`, `pipeline`, `capture`, `audio`, `diarize`,
  `transcribe`, `voices`, `runtime`, `send`, and `watch`.
- Heavy optional dependencies are imported lazily so help, tests, lint, and type
  checking remain usable without CUDA packages.
- Outputs are `<recording>.md` and `<recording>.json`, written atomically beside
  the recording.

## Invariants

1. Audio inference stays local to a user-controlled machine. Only an explicit
   SSH transfer may move audio between such machines.
2. ASR windows long recordings and stitches timestamps; never run long audio as
   one unbounded model pass.
3. One malformed segment must not discard an otherwise usable transcript.
4. Select `HF_HOME` before model imports: `LOCALSCRIBE_HF_HOME`, then `HF_HOME`,
   then Hugging Face's XDG-compatible default.
5. Load heavy models sequentially and release CUDA memory between stages.
6. Prefer a USB microphone over a route-volatile analog input and warn on a dead
   mic channel without aborting the other channel.
7. Voiceprint directories and files remain private (`0700`/`0600`).
8. Participant recordings, transcripts, and voiceprints stay outside version
   control.

## Project structure

- `src/localscribe/` — application package.
- `tests/` — unit, property, process, ffmpeg, and rsync checks.
- `docs/design-plans/` — accepted design records.
- `.github/workflows/ci.yml` — public light-environment verification.
- `README.md` — user-facing install, privacy, and usage contract.

## Boundaries

- Preserve the lock-derived `install.sh --gpu` flow: ordinary tool installation
  cannot see the project's CUDA source configuration.
- Treat GPU transcription quality and live capture as hardware UAT, separate
  from automated checks.
- Keep private machine paths, participant identities, recordings, transcripts,
  credentials, and voiceprints out of tracked files.
