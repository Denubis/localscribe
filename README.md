# LocalScribe

LocalScribe is a Linux command-line tool for speaker-attributed meeting
transcription: audio in, Markdown and JSON transcripts out, with speaker labels
and timestamps.

Transcription and diarisation run on hardware you control. LocalScribe does not
send audio to a public-cloud transcription API. Its optional `scribe-send`
command transfers recordings over SSH to a transcription host you explicitly
choose—for example, from a laptop with a microphone to your own GPU workstation.

LocalScribe is alpha software. It has been exercised on real meetings up to 90
minutes and five speakers, but its known-working hardware is not a minimum
requirement or a general compatibility claim.

## What it provides

- `scribe`: transcribe one or more existing recordings.
- `scribe-enrol`: store a local speaker voiceprint so transcripts can use a name.
- `scribe-record`: record a remote or in-person meeting, then transcribe it.
- `scribe-send`: record locally or take an existing FLAC, then send it over SSH.
- `scribe-watch`: receive queued recordings and transcribe them into the current
  project directory.

Each transcription produces `<recording>.md` for reading and
`<recording>.json` for durable structured data.

## Supported environment

The current supported and tested environment is:

- Linux and Python 3.13;
- `ffmpeg`/`ffprobe` for audio preparation and validation;
- PipeWire, WirePlumber (`wpctl`), and `pw-dump` for live recording;
- OpenSSH and `rsync` for `scribe-send`;
- an NVIDIA CUDA GPU for transcription.

An RTX 4090 with 24 GB VRAM is the known-working reference. LocalScribe has not
established a minimum GPU specification. ASR is processed in 20-minute windows
to bound VRAM use on long recordings.

## Install

Install [uv](https://docs.astral.sh/uv/), clone this repository, then choose one
installation:

```console
./install.sh
```

This light installation supports recording and sending without installing the
CUDA model stack.

```console
./install.sh --gpu
```

The GPU installation adds transcription, diarisation, enrolment, and the inbox
watcher. Use the script from a source checkout: it derives exact constraints
from `uv.lock` and supplies the CUDA wheel index that ordinary wheel metadata
cannot preserve.

Before the first GPU run:

1. Accept the access conditions for
   [pyannote Community-1](https://huggingface.co/pyannote/speaker-diarization-community-1).
2. Authenticate with Hugging Face using its current CLI or token mechanism.
3. Ensure the model cache has enough space; the complete stack uses many
   gigabytes.

Cache selection is deterministic:

1. `LOCALSCRIBE_HF_HOME`, when set;
2. standard `HF_HOME`, when set;
3. `$XDG_CACHE_HOME/huggingface`, or `~/.cache/huggingface` when XDG is unset.

This lets a GPU workstation point LocalScribe at an existing large model cache
without embedding one developer's filesystem layout in the package.

## Use

Transcribe an existing recording:

```console
scribe meeting.flac
```

Record an in-person meeting from the preferred USB microphone:

```console
scribe-record recordings/meeting.flac --in-person
```

Record a remote meeting, capturing the microphone and system output:

```console
scribe-record recordings/meeting.flac --remote
```

The microphone is selected once at startup and printed before recording: an
explicit `--mic` wins, then the current USB default, then any USB source, then
the system default. Connect the intended microphone before starting; the chosen
source stays fixed for the whole recording.

In a terminal, press Enter to finish the current file and transcribe it in the
background while recording continues into `meeting-part002.flac`,
`meeting-part003.flac`, and so on. The microphone stays open across file
changes. Background jobs run one at a time, in a fresh process each, sharing the
GPU lock with `scribe-watch`. Press Ctrl-C once to stop recording, finish the
last file, and wait for queued transcripts; press Ctrl-C again to cancel the
remaining jobs and keep the FLACs for a later `scribe`. Without a terminal, the
command records one file and transcribes it when stopped. `--no-transcribe`
keeps the same Enter and Ctrl-C controls and produces only the numbered FLACs.

If a transcription fails, the recording is kept and the command prints the error
and the `scribe` command to retry it.

Enrol a speaker from a clean single-speaker clip:

```console
scribe-enrol speaker-sample.wav --name Alice
```

Send an existing recording to a GPU workstation you control:

```console
scribe-send recordings/meeting.flac --host gpu-box --name project-a
```

On that workstation, run a watcher from the destination directory:

```console
cd recordings/project-a
scribe-watch --match project-a
```

The sender does not choose a destination project. The watcher decides where a
transcript lands by the directory in which it runs.

## Privacy and data handling

Meeting recordings, transcripts, and enrolled voice embeddings can all be
sensitive research or personal data.

- No audio is sent to a public-cloud transcription service.
- `scribe-send` uses the local `ssh`/`rsync` configuration and only contacts the
  host supplied by the user or `LOCALSCRIBE_HOST`.
- Model and package downloads contact their respective repositories, but audio
  is not uploaded as part of model loading.
- Voiceprints default to `$XDG_DATA_HOME/localscribe/voices` or
  `~/.local/share/localscribe/voices`; LocalScribe enforces mode `0700` on the
  registry and `0600` on saved voiceprint files.
- Remote rsync copies are requested with file mode `0600` and directory mode
  `0700`.
- Local recordings are retained unless `scribe-send --discard-local` is used.
  LocalScribe does not implement a retention schedule or secure erasure.

The repository ignores common audio formats, `recordings/`, `.localscribe/`,
and any directory named `voices/`. Keep real participant data under an ignored
data directory rather than beside source files; generic Markdown and JSON cannot
be ignored globally because the repository itself uses both.

## Models and licenses

LocalScribe's source code is licensed under Apache-2.0. It does not bundle model
weights. The models it downloads have their own licenses, access conditions,
and attribution requirements. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
before redistributing a configured system or model files.

## Development

The default development environment intentionally excludes CUDA packages, so
ordinary contributors can run the complete automated suite:

```console
uv sync --locked
uv run --frozen pytest
uv run --frozen ruff check .
uv run --frozen ty check
uv build --no-sources
```

GPU end-to-end validation remains a separate hardware acceptance check. The
automated suite covers pure transformations, CLI boundaries, capture command
construction, transfer invariants, real cross-process locking, and real local
rsync rename behavior; it does not establish transcription quality on arbitrary
recordings or GPUs.

## Known limitations

- Dense overlapping speech and very short interjections can be assigned to the
  wrong speaker.
- The default speaker-enrolment threshold is an initial operating value, not a
  broadly calibrated biometric threshold.
- Live recording currently targets Linux PipeWire environments.
- Voiceprints and transcripts are protected by filesystem permissions, not
  encrypted at rest by LocalScribe.
