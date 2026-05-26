from core.audio_utils import classify_mic_open_error


def test_classify_mic_open_error_channel_busy() -> None:
    err = OSError("[Errno -9998] Invalid number of channels")
    assert classify_mic_open_error(err) == "Mic Error: device busy or unavailable"


def test_classify_mic_open_error_device_in_use() -> None:
    err = OSError("[Errno -9985] Device unavailable")
    assert classify_mic_open_error(err) == "Mic Error: device in use by another app"
