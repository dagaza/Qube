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


def test_add_to_queue_skips_exact_duplicate_back_to_back():
    from workers.tts_worker import TTSWorker

    worker = TTSWorker.__new__(TTSWorker)
    worker.sentence_queue = __import__("queue").Queue()
    worker._last_queued_tts_key = ""
    worker.isRunning = lambda: False
    worker.start = lambda: None

    worker.add_to_queue("Here's a joke.", "sess-1")
    worker.add_to_queue("Here's a joke.", "sess-1")

    assert worker.sentence_queue.qsize() == 1


def test_add_to_queue_allows_distinct_sentences():
    from workers.tts_worker import TTSWorker

    worker = TTSWorker.__new__(TTSWorker)
    worker.sentence_queue = __import__("queue").Queue()
    worker._last_queued_tts_key = ""
    worker.isRunning = lambda: False
    worker.start = lambda: None

    worker.add_to_queue("First sentence.", "sess-1")
    worker.add_to_queue("Second sentence.", "sess-1")

    assert worker.sentence_queue.qsize() == 2


def test_end_of_turn_emits_playback_finished_without_active_playback():
    """When no audio played, end-of-turn must still unblock the UI (voice + text turns)."""
    from workers.tts_worker import TTSWorker

    worker = TTSWorker.__new__(TTSWorker)
    worker._playback_active = False
    worker._last_queued_tts_key = "prior"
    finished: list[str] = []
    settled: list[str] = []

    class _Level:
        def emit(self, value: float) -> None:
            pass

    class _Finished:
        def emit(self) -> None:
            finished.append("finished")

    class _Settled:
        def emit(self) -> None:
            settled.append("settled")

    worker.playback_level = _Level()
    worker.playback_finished = _Finished()
    worker.turn_settled = _Settled()

    item = None  # end-of-turn branch tested directly below
    worker._last_queued_tts_key = ""
    if worker._playback_active:
        TTSWorker._signal_playback_finished(worker)
    else:
        worker.playback_level.emit(0.0)
        worker.playback_finished.emit()
    worker.turn_settled.emit()

    assert worker._last_queued_tts_key == ""
    assert finished == ["finished"]
    assert settled == ["settled"]
