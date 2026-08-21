# Robust Capture Design

**GitHub Issue:** None

## Summary

`scribe-record` silently lost the user's own voice when earbuds were plugged into the
combo jack mid-session. Investigation showed the failure is one level below where the tool
reasons: it resolves `@DEFAULT_AUDIO_SOURCE@`/`@DEFAULT_AUDIO_SINK@` to a concrete node name
once at start and pins ffmpeg there, but the volatility is in the device's *input route*, not
the node identity. The default source stayed node 54 while its route flipped to a dead
`analog-input-headset-mic` port. Neither pinning the node nor "following the default node"
helps, because the default node never moved.

This design makes capture resist that class of failure rather than chase the moving route:
record from a reroute-immune USB mic by default, ask the user up front whether the meeting is
remote or in-person (two different capture shapes), warn loudly if the mic channel goes dead
instead of failing silently, and finish a recording as a pasteable markdown transcript in one
command.

## Definition of Done

1. **Your voice survives device shuffling.** `scribe-record` records from the C930c USB mic by
   default — a separate USB card that no combo-jack picker can reroute. Selection is a policy
   ("prefer a USB source over the volatile analog-jack source, else the system default"), not a
   brittle hardcoded name; the chosen mic is printed at startup. `--mic <node>` overrides it.
   *Observable:* a recording made while plugging/unplugging earbuds keeps the user's audio
   throughout. *Excludes:* chasing PipeWire's moving defaults mid-recording.

2. **The tool asks remote-or-in-person at the start.** *Remote/hybrid* → mic + system monitor,
   stereo (you = L, call = R). *In-person* → C930c only, mono, no monitor tap (so stray desktop
   audio can't enter as a phantom speaker). Flags `--remote`/`--in-person` skip the prompt for
   scripts; non-interactive with no flag defaults to remote (today's behaviour).
   *Observable:* choosing in-person yields a 1-channel file; remote yields the 2-channel file.

3. **A dead channel shouts, never goes silent.** While recording, the mic leg is watched; if it
   carries no signal (below −50 dB) for longer than the threshold (20 s), a clear warning is
   printed **and recording continues** — aborting would lose the other side too.
   *Observable:* a dead mic produces a visible warning within ~20 s; recording keeps running.
   *Excludes:* warning on a quiet-but-live mic (room tone sits above −50 dB).

4. **One command, record → pasteable `.md`.** Ctrl-C stops recording, then the existing
   transcribe pipeline runs and writes `.md` (+ `.json`) beside the FLAC; the `.md` path is
   printed. The **FLAC is kept** (ethics-governed research data). `--no-transcribe` keeps only
   the raw FLAC. If transcription fails, the FLAC is retained and the user is told to run
   `scribe <flac>` later. *Observable:* one `scribe-record` invocation ends with a `.md` on disk.

**Out of scope:** following PipeWire's moving defaults mid-recording; auto-detecting the meeting
type; changing the transcription/diarisation models; improving IRL diarisation quality.

## Acceptance Criteria

- **robust-capture.AC1.1** — `choose_mic` prefers an `alsa_input.usb-*` source over an
  `alsa_input.pci-*` source. Fail case: given only a PCI source, returns the default.
- **robust-capture.AC1.2** — an explicit `--mic` value is returned unchanged regardless of policy.
- **robust-capture.AC1.3** — source enumeration parses `pw-dump` JSON into source node names; a
  malformed/absent dump falls back to the system default source rather than raising.
- **robust-capture.AC2.1** — `build_capture_command(..., monitor=None)` yields a single-input,
  mono FLAC graph (no second `-f pulse` input, no `join`). Fail case: a two-input graph.
- **robust-capture.AC2.2** — `monitor` provided yields the two-input you=L/them=R stereo graph
  (unchanged from today). `--mono` still mixes to one channel.
- **robust-capture.AC2.3** — `--in-person` selects mono/no-monitor; `--remote` selects
  stereo/mic+monitor; both bypass the prompt. Non-TTY with neither flag → remote.
- **robust-capture.AC3.1** — `parse_silence_events` extracts `(start)` from
  `silence_start: 2.58754` and `(end, duration)` from
  `silence_end: 4.0725 | silence_duration: 1.48496`; non-matching lines → None.
- **robust-capture.AC3.2** — `SilenceTracker` emits a warning on `silence_start`, a recovery note
  on `silence_end`, and rate-limits repeated warnings. Fail case: silent on a progress line.
- **robust-capture.AC3.3** — the mic branch of the graph contains
  `silencedetect=noise=-50dB:d=20`; the monitor branch does not.
- **robust-capture.AC4.1** — `scribe-record --no-transcribe` writes the FLAC and no `.md`.
- **robust-capture.AC4.2** — with the heavy stages monkeypatched, `scribe-record` (default) writes
  `.md` and `.json` beside the FLAC; `scribe <flac>` writes the same (shared `transcribe_recording`).
- **robust-capture.AC4.3** — if `transcribe_recording` raises `LocalscribeError`, the FLAC remains
  on disk, a "run scribe later" message is printed, and exit code is non-zero.

## Architecture

**FCIS placement.** All new logic that can be pure is pure and lives with value-equality tests;
the ffmpeg/pw-dump/CUDA boundaries stay in the imperative shell, hand-verified (the convention
`capture.py` already follows).

- `capture.py` (shell + pure helpers):
  - *pure, new:* `parse_sources(pw_dump_json) -> list[str]` (source node names from `pw-dump`
    JSON); `choose_mic(sources, explicit, default) -> str` (policy); `parse_silence_events(line)
    -> SilenceEvent | None`; `SilenceTracker` (state machine over silence events → warning
    strings).
  - *pure, changed:* `build_capture_command(mic, monitor, output, *, mono)` — `monitor` becomes
    `str | None`; `None` → one-input mono graph; the mic branch gains `silencedetect=...`.
  - *shell, new/changed:* `list_sources()` (runs `pw-dump`, falls back on failure); `capture()`
    grows mode/monitor handling, reads ffmpeg stderr live through the tracker, prints warnings.
- `pipeline.py` (shell, new): `transcribe_recording(recording, *, min_speakers, max_speakers,
  name_threshold)` — the per-recording pipeline lifted verbatim from `cli.scribe`, called by both
  `scribe` and `record`.
- `cli.py` (shell): `scribe` delegates to `transcribe_recording`; `record` gains the
  remote/in-person prompt, `--remote`/`--in-person`/`--no-transcribe` flags, mic-policy wiring,
  and the post-stop transcribe step.

**Silence guard mechanism (verified live).** `silencedetect` is `A->A` (passes audio through), so
it sits inline on the mic branch. It emits `silence_start`/`silence_end` to stderr *as they
occur* during encoding (confirmed: events arrived mid-stream at t≈2.6 s and t≈4.1 s while ffmpeg
kept running). `noise=-50dB` detects absence of signal — a live mic in a room carries ambient
tone above −50 dB, a dead/muted/wrong-port channel sits below — so the guard separates "mic is
dead" from "you're listening quietly". `d=20` requires 20 s of continuous dead air before a
`silence_start` fires, which the shell turns into one rate-limited warning.

**Reading stderr + Ctrl-C.** `silence_start`/`silence_end` lines are `\n`-terminated and arrive
live, so a `for line in proc.stderr` loop suffices (no need to parse `\r` progress). The existing
SIGINT→ffmpeg→clean-FLAC-trailer contract is preserved: KeyboardInterrupt in the read loop
forwards SIGINT to ffmpeg, then drains and waits.

## Existing Patterns Followed

- Pure helpers in `capture.py` unit-tested; subprocess boundary hand-verified (current convention).
- Typed errors via `CaptureError`/`LocalscribeError`; CLI turns them into clean message + exit code.
- Lazy heavy imports (torch/pyannote/NeMo only inside the transcribe path) so `--help` stays cold.
- Atomic writes for outputs (`_atomic_write`).
- Guardrails: tolerate-and-continue on a bad `pw-dump` (fall back to default); never lose the FLAC.

## Implementation Phases

1. **Mic-selection policy** — `parse_sources`, `choose_mic`, `list_sources` (shell) + tests; wire
   into `capture()`, print chosen mic.
2. **Capture modes** — `build_capture_command` monitor optional + mono one-input graph; `record`
   prompt + `--remote`/`--in-person` + tests.
3. **Silence guard** — `parse_silence_events` + `SilenceTracker` + `silencedetect` in the graph +
   live stderr reader in `capture()` + tests.
4. **record → md** — characterisation test for `scribe` pipeline; extract `pipeline.py`; wire
   `record` auto-transcribe + `--no-transcribe` + failure-keeps-FLAC + tests.
5. **Verification + docs** — pytest/ruff/ty green; update `.notes/project_status.md` and the
   `CLAUDE.md` capture contract; write the UAT note (unit-tested vs needs live mic + GPU).

## Additional Considerations

- **False positives on the guard** are accepted by design: an occasional benign nudge beats
  silently losing the user's voice (DoD 3). The −50 dB / 20 s defaults minimise them.
- **`pw-dump` dependency:** ships with PipeWire alongside the already-required `wpctl`; parsed
  with stdlib `json` (no `jq`). Absence falls back to the default source.
- **Non-interactive default = remote** preserves the current behaviour for any existing scripts.

## Glossary

- **Node vs route:** a PipeWire *node* (e.g. `alsa_input.pci-...analog-stereo`) is a stable
  capture endpoint; the device's *route* selects which physical port (internal mic, headset mic)
  feeds it. The bug lived in a route change under a stable node.
- **Monitor source:** the `.monitor` of a sink — what is *played* to that sink, used to capture
  the far end of a call.
- **Combo jack:** the 3.5 mm jack whose insertion triggers GNOME's headphones/headset/mic picker
  and switches the analog card's input route.
- **silencedetect:** ffmpeg audio filter that logs silence start/end to stderr while passing audio
  through unchanged.
- **FCIS:** Functional Core / Imperative Shell — pure logic separated from side-effecting IO.
