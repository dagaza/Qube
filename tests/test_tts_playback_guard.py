"""Regression: idle TTS thread must not emit playback_finished on wake-up."""


def test_signal_playback_finished_is_noop_when_idle(monkeypatch):
    from workers.tts_worker import TTSWorker

    worker = TTSWorker.__new__(TTSWorker)
    worker._playback_active = False
    emitted: list[float] = []

    monkeypatch.setattr(worker, "playback_level", type("Sig", (), {"emit": emitted.append})())
    monkeypatch.setattr(
        worker,
        "playback_finished",
        type("Sig", (), {"emit": lambda: emitted.append("finished")})(),
    )

    worker._signal_playback_finished()

    assert emitted == []


def test_signal_playback_finished_clears_active_flag(monkeypatch):
    from workers.tts_worker import TTSWorker

    worker = TTSWorker.__new__(TTSWorker)
    worker._playback_active = True
    finished: list[str] = []

    monkeypatch.setattr(worker, "playback_level", type("Sig", (), {"emit": lambda _v: None})())
    monkeypatch.setattr(
        worker,
        "playback_finished",
        type("Sig", (), {"emit": lambda: finished.append("finished")})(),
    )

    worker._signal_playback_finished()

    assert worker._playback_active is False
    assert finished == ["finished"]
