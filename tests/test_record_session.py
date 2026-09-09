"""Terminal events must preserve captured parts and drain queued transcription."""

import os
import signal
import sys
import threading

import pytest

from localscribe import record_session
from localscribe.errors import CaptureError, MissingExtraError


def test_enter_reader_coalesces_lines_and_exits_after_stop(monkeypatch) -> None:
    read_fd, write_fd = os.pipe()
    rollover, stopped = threading.Event(), threading.Event()
    with os.fdopen(read_fd) as source:
        monkeypatch.setattr(sys, 'stdin', source)
        reader = threading.Thread(
            target=record_session._read_enter, args=(rollover, stopped, print)
        )
        reader.start()
        try:
            os.write(write_fd, b'\n\n')
            assert rollover.wait(timeout=2), 'Enter did not request a new part'
        finally:
            stopped.set()
            reader.join(timeout=2)
            os.close(write_fd)
        assert not reader.is_alive()


@pytest.mark.parametrize('failure', [None, 'capture', 'transcription', 'interrupt'])
def test_session_stops_capture_then_finishes_or_cancels_jobs(
    monkeypatch, tmp_path, failure
) -> None:
    events = []
    original_handler = signal.getsignal(signal.SIGINT)

    class Jobs:
        def __init__(self, echo):
            pass

        def submit(self, path):
            assert path.is_file(), 'Unfinalized file was queued'
            events.append(('queued', path.name))

        def finish(self):
            assert signal.getsignal(signal.SIGINT) == original_handler
            events.append('finished')
            if failure == 'interrupt':
                raise KeyboardInterrupt
            return int(failure == 'transcription')

        def cancel(self):
            events.append('cancelled')

    def capture(output, *, stop, on_complete, **kwargs):
        for path in [output, output.with_stem(output.stem + '-part002')]:
            path.write_bytes(b'finalized audio')
            on_complete(path)
        signal.raise_signal(signal.SIGINT)
        assert stop.is_set(), 'Ctrl-C did not request graceful capture shutdown'
        events.append('capture stopped')
        if failure == 'capture':
            raise CaptureError('input failed')
        return []

    monkeypatch.setattr('localscribe.jobs.TranscriptionQueue', Jobs)
    monkeypatch.setattr('localscribe.rolling.capture_parts', capture)
    monkeypatch.setattr(record_session, 'require_gpu_extra', lambda: None)
    monkeypatch.setattr(record_session, '_read_enter', lambda *args: None)
    output = tmp_path / 'meeting.flac'

    def record():
        return record_session.record_interactively(
            output, mic='USB', mono=False, remote=True,
            no_transcribe=False, echo=lambda text: None,
        )

    if failure == 'capture':
        with pytest.raises(CaptureError, match='input failed'):
            record()
    else:
        code = record()
        assert code == {'transcription': 1, 'interrupt': 130}.get(failure, 0)
    assert signal.getsignal(signal.SIGINT) == original_handler
    assert events[:3] == [('queued', 'meeting.flac'), ('queued', 'meeting-part002.flac'),
                          'capture stopped']
    assert events[-1] == ('cancelled' if failure in {'capture', 'interrupt'} else 'finished')
    assert output.read_bytes() == b'finalized audio'


@pytest.mark.parametrize('no_transcribe', [True, False])
def test_light_install_keeps_recording_without_launching_jobs(monkeypatch, tmp_path, no_transcribe):
    def missing():
        raise MissingExtraError('install with ./install.sh --gpu')

    def no_jobs(**kwargs):
        pytest.fail('Light/audio-only recording must not launch model workers')

    def capture(output, *, on_complete, **kwargs):
        output.write_bytes(b'finalized audio')
        on_complete(output)
        return [output]

    monkeypatch.setattr(record_session, 'require_gpu_extra', missing)
    monkeypatch.setattr('localscribe.jobs.TranscriptionQueue', no_jobs)
    monkeypatch.setattr('localscribe.rolling.capture_parts', capture)
    monkeypatch.setattr(record_session, '_read_enter', lambda *args: None)
    messages = []
    output = tmp_path / 'meeting.flac'
    code = record_session.record_interactively(output, mic='USB', mono=False, remote=False,
                                               no_transcribe=no_transcribe, echo=messages.append)
    assert code == (0 if no_transcribe else 3)
    assert output.read_bytes() == b'finalized audio'
    if not no_transcribe:
        assert any('scribe ' + str(output) in message for message in messages)


def test_failed_terminal_reader_start_restores_signal_and_cancels_jobs(monkeypatch, tmp_path):
    original_handler = signal.getsignal(signal.SIGINT)
    cancelled = []

    class Jobs:
        def __init__(self, echo):
            pass

        def cancel(self):
            cancelled.append(True)

    def fail_start(self):
        raise RuntimeError('cannot start reader')

    monkeypatch.setattr(record_session, 'require_gpu_extra', lambda: None)
    monkeypatch.setattr('localscribe.jobs.TranscriptionQueue', Jobs)
    monkeypatch.setattr(threading.Thread, 'start', fail_start)
    try:
        with pytest.raises(RuntimeError, match='cannot start reader'):
            record_session.record_interactively(tmp_path / 'm.flac', mic='USB', mono=False,
                                                 remote=True, no_transcribe=False)
        assert signal.getsignal(signal.SIGINT) == original_handler
        assert cancelled == [True]
    finally:
        signal.signal(signal.SIGINT, original_handler)
