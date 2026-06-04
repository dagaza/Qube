"""Regression: idle TTS thread must not emit playback_finished on wake-up."""


class _BareTTSWorker:
    """Minimal stand-in so we can call _signal_playback_finished without QThread init."""

    _playback_active = False
    playback_level = None
    playback_finished = None


def test_signal_playback_finished_is_noop_when_idle():
    from workers.tts_worker import TTSWorker

    worker = _BareTTSWorker()
    worker._playback_active = False
    emitted: list[object] = []

    class _Level:
        def emit(self, value: float) -> None:
            emitted.append(value)

    class _Finished:
        def emit(self) -> None:
            emitted.append("finished")

    worker.playback_level = _Level()
    worker.playback_finished = _Finished()

    TTSWorker._signal_playback_finished(worker)

    assert emitted == []


def test_signal_playback_finished_clears_active_flag():
    from workers.tts_worker import TTSWorker

    worker = _BareTTSWorker()
    worker._playback_active = True
    finished: list[str] = []

    class _Level:
        def emit(self, value: float) -> None:
            pass

    class _Finished:
        def emit(self) -> None:
            finished.append("finished")

    worker.playback_level = _Level()
    worker.playback_finished = _Finished()

    TTSWorker._signal_playback_finished(worker)

    assert worker._playback_active is False
    assert finished == ["finished"]
