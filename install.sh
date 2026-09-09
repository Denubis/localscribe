#!/usr/bin/env bash
# install.sh — install LocalScribe's commands with uv.
#
#   ./install.sh          light install: scribe-record + scribe-send, ffmpeg only.
#                         This is what a laptop wants — no CUDA, ~8 packages.
#   ./install.sh --gpu    full install: adds scribe, scribe-enrol, scribe-watch.
#                         Needs a supported NVIDIA/CUDA transcription host.
#
# Why --gpu is not just `uv tool install '.[gpu]'`:
#
# `uv tool install` resolves from *built wheel metadata*, and `[tool.uv]` is
# project configuration that wheel metadata does not carry. So the cu128 index
# and the torch/torchaudio `sources` pins are invisible to it. Without them
# torch==2.11.0+cu128 is unreachable, the resolver backtracks looking for a
# consistent set, the machine-wide `exclude-newer` narrows the field further, and
# it settles on numba 0.53.1 -> llvmlite 0.36.0 — sdist-only, and guarded at
# *build* time for Python <3.10. The result is a build failure that reads like a
# broken package rather than a lost index.
#
# The fix is to hand the resolver the answer it already has: every version is
# pinned from uv.lock, which is resolved in project mode and therefore does see
# the index. The constraints are regenerated here on every run, so they cannot go
# stale against the lock.
set -uo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo" || exit 1

command -v uv >/dev/null || { echo "uv not on PATH" >&2; exit 127; }

case "${1:-}" in
  "") want_gpu=0 ;;
  --gpu) want_gpu=1 ;;
  *) echo "usage: ./install.sh [--gpu]" >&2; exit 2 ;;
esac

if (( ! want_gpu )); then
  echo "installing localscribe (light: record + send, no model stack)"
  uv tool install --force "$repo" || exit 1
  echo
  echo "installed. On this machine you can record and push:"
  echo "  scribe-send --host <box> -n <label>"
  echo "  (set LOCALSCRIBE_HOST to skip --host)"
  exit 0
fi

constraints="$(mktemp -t localscribe-gpu-constraints.XXXXXX.txt)"
trap 'rm -f "$constraints"' EXIT

echo "exporting the locked resolution ..."
# Export the committed answer, regardless of machine-wide resolver settings.
uv export --frozen --quiet --extra gpu --no-dev --no-emit-project --no-hashes \
  --format requirements-txt -o "$constraints" || exit 1

# The pytorch index has to be named on the command line, and naming a second
# index makes uv refuse to look past it for packages both indexes carry (its
# dependency-confusion guard, which filelock trips). unsafe-best-match lifts
# that. It is only safe here because every version above is pinned exactly, so
# nothing can drift to a surprise — regenerate the constraints, never hand-edit.
echo "installing localscribe[gpu] ..."
uv tool install --force \
  --index https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match \
  --constraints "$constraints" \
  "${repo}[gpu]" || exit 1

# An install that silently landed a CPU torch would pass every other check and
# fail at the first diarisation, so verify the wheel rather than trust it.
echo
echo "verifying ..."
tool_dir="$(uv tool dir)" || exit 1
tool_bin_dir="$(uv tool dir --bin)" || exit 1
tool_python="$tool_dir/localscribe/bin/python"
[[ -x "$tool_python" ]] || { echo "FAIL: no interpreter at $tool_python" >&2; exit 1; }

torch_version="$("$tool_python" -c 'import torch; print(torch.__version__)' 2>/dev/null)"
if [[ "$torch_version" != *"+cu"* ]]; then
  echo "FAIL: torch is '$torch_version', expected a +cuXXX build." >&2
  echo "  The index pin was lost. Do not use this install for transcription." >&2
  exit 1
fi
echo "  torch $torch_version"

missing=0
for command in scribe scribe-enrol scribe-record scribe-send scribe-watch; do
  if [[ -x "$tool_bin_dir/$command" ]]; then
    echo "  $tool_bin_dir/$command"
  else
    echo "  MISSING: $command" >&2
    missing=1
  fi
done
(( missing )) && exit 1

echo
echo "installed. To receive recordings, run this in the folder the transcripts belong in:"
echo "  cd ~/recordings/<project> && scribe-watch"
