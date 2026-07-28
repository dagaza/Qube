"""VU meter display behavior and audio-worker level-monitor hooks."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_vu_meter_attack_and_release(qapp):
    from ui.main_window import VUMeter

    meter = VUMeter()
    meter.set_level(0.0)
    meter.set_level(0.8)
    assert meter._level == pytest.approx(0.8)

    meter.set_level(0.0)
    assert 0.0 < meter._level < 0.8
    assert meter._level == pytest.approx(0.8 * 0.82, rel=1e-3)


def test_update_mic_level_applies_display_curve(qapp):
    from ui.main_window import MainWindow, VUMeter

    window = MainWindow.__new__(MainWindow)
    window.vu_meter = VUMeter()
    MainWindow.update_mic_level(window, 0.25)
    assert window.vu_meter._level == pytest.approx(0.25**0.55, rel=1e-3)


def test_set_paused_emits_zero_volume():
    from workers.audio_worker import AudioListenerWorker

    worker = AudioListenerWorker()
    emitted: list[float] = []
    worker.volume_update.connect(emitted.append)

    worker.set_paused(True)
    assert worker.is_paused is True
    assert emitted == [0.0]


def test_request_level_monitor_extends_window(monkeypatch):
    from workers.audio_worker import AudioListenerWorker

    worker = AudioListenerWorker()
    monkeypatch.setattr("workers.audio_worker.time.time", lambda: 100.0)
    worker.request_level_monitor(8.0)
    assert worker._level_monitor_until == pytest.approx(108.0)
    assert worker._level_monitor_active() is True

    monkeypatch.setattr("workers.audio_worker.time.time", lambda: 109.0)
    assert worker._level_monitor_active() is False


def test_set_paused_clears_level_monitor(monkeypatch):
    from workers.audio_worker import AudioListenerWorker

    worker = AudioListenerWorker()
    monkeypatch.setattr("workers.audio_worker.time.time", lambda: 50.0)
    worker.request_level_monitor(10.0)
    worker.set_paused(True)
    assert worker._level_monitor_until == 0.0
