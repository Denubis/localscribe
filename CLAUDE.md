# LocalScribe

Last verified: 2026-09-22

## Purpose

Local Linux CLI for speaker-attributed transcription of meetings. Inference runs
on hardware the user controls. `scribe-send` may transfer audio over SSH to a
transcription host the user explicitly chooses; audio never goes to a
public-cloud transcription service.

Model downloads and Hugging Face authentication may use the network; model
telemetry is disabled before the heavy imports.

## Tech stack

- Python 3.13, uv, Typer, pytest, Ruff, and ty.
- ASR: `nvidia/parakeet-tdt-0.6b-v3` via NeMo, with word timestamps.
- Diarisation: pyannote.audio 4.x with
  `pyannote/speaker-diarization-community-1`.
- Audio/capture: ffmpeg, PipeWire, WirePlumber.
- Transfer: OpenSSH and rsync.
- Known-working GPUs: RTX 4090 with 24 GB VRAM, and an RTX 5080 laptop with
  16 GB VRAM on a synthetic 90-minute recording. These are references, not a
  measured minimum; size GPU work for the smaller card, and do not treat a short
  smoke test as long-meeting capacity.

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
- Imperative shell: `cli`, `pipeline`, `capture`, `rolling`, `record_session`,
  `jobs`, `audio`, `diarize`, `transcribe`, `voices`, `runtime`, `send`, and
  `watch`.
- Heavy optional dependencies are imported lazily so help, tests, lint, and type
  checking remain usable without CUDA packages.
- Outputs are `<recording>.md` and `<recording>.json`, written atomically beside
  the recording.
- Continuous recording: a terminal `scribe-record` holds one persistent capture
  source (`rolling`). Whole interleaved PCM frames go to exactly one FLAC part;
  Enter switches the destination encoder without reopening the microphone or
  monitor. Only validated, finished parts reach the transcription queue.
- Background transcription: `jobs` processes finished parts first-in first-out,
  one fresh process per recording, under the same GPU lock as `scribe-watch`. A
  failed job keeps its audio, prints a retry command, and lets later jobs run.
- ASR decoding keeps NeMo's decoder CUDA graphs disabled in the model's
  persistent decoding configuration. Graph reuse across windows caused an
  illegal-memory abort on an RTX 5080, and requesting timestamps can rebuild the
  decoder from that configuration, so a runtime-only disable is not enough.

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
   mic channel without aborting the other channel. Select the mic once at
   startup (explicit override, USB default, any USB source, system default) and
   keep it fixed for the recording.
7. Voiceprint directories and files remain private (`0700`/`0600`).
8. Participant recordings, transcripts, and voiceprints stay outside version
   control.
9. A failed or cancelled transcription never discards the recording; it reports
   how to retry with `scribe`.

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
