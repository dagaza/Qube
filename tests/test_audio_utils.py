from core.audio_utils import classify_mic_open_error, get_audio_devices


def test_classify_mic_open_error_channel_busy() -> None:
    err = OSError("[Errno -9998] Invalid number of channels")
    assert classify_mic_open_error(err) == "Mic Error: device busy or unavailable"


def test_classify_mic_open_error_device_in_use() -> None:
    err = OSError("[Errno -9985] Device unavailable")
    assert classify_mic_open_error(err) == "Mic Error: device in use by another app"


def test_get_audio_devices_returns_separate_input_output_lists(monkeypatch) -> None:
    class FakePyAudio:
        def get_device_count(self) -> int:
            return 2

        def get_device_info_by_index(self, index: int) -> dict:
            if index == 0:
                return {"name": "Mic", "maxInputChannels": 1, "maxOutputChannels": 0}
            return {"name": "Speaker", "maxInputChannels": 0, "maxOutputChannels": 2}

        def terminate(self) -> None:
            return None

    monkeypatch.setattr("core.audio_utils.pyaudio.PyAudio", FakePyAudio)
    inputs, outputs = get_audio_devices()
    assert inputs == [(0, "Input 0: Mic")]
    assert outputs == [(1, "Device 1: Speaker")]
