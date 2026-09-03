from core.audio_utils import classify_mic_open_error, get_audio_devices, build_audio_device_menu_rows


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

        def get_host_api_count(self) -> int:
            return 0

        def get_device_info_by_index(self, index: int) -> dict:
            if index == 0:
                return {"name": "Mic", "maxInputChannels": 1, "maxOutputChannels": 0}
            return {"name": "Speaker", "maxInputChannels": 0, "maxOutputChannels": 2}

        def terminate(self) -> None:
            return None

    monkeypatch.setattr("core.audio_utils.pyaudio.PyAudio", FakePyAudio)
    inputs, outputs = get_audio_devices()
    assert inputs == [(0, "Mic")]
    assert outputs == [(1, "Speaker")]


def test_get_audio_devices_prefers_wasapi_on_windows(monkeypatch) -> None:
    import pyaudio

    class FakePyAudio:
        def get_device_count(self) -> int:
            return 4

        def get_host_api_count(self) -> int:
            return 2

        def get_host_api_info_by_index(self, index: int) -> dict:
            if index == 0:
                return {"type": pyaudio.paMME}
            return {"type": pyaudio.paWASAPI}

        def get_device_info_by_index(self, index: int) -> dict:
            devices = {
                0: {
                    "name": "Microsoft Sound Mapper - Input",
                    "hostApi": 0,
                    "maxInputChannels": 2,
                    "maxOutputChannels": 0,
                },
                1: {
                    "name": "Microphone (Realtek High Defini",
                    "hostApi": 0,
                    "maxInputChannels": 2,
                    "maxOutputChannels": 0,
                },
                2: {
                    "name": "Microphone (Realtek High Definition Audio)",
                    "hostApi": 1,
                    "maxInputChannels": 2,
                    "maxOutputChannels": 0,
                },
                3: {
                    "name": "Speakers (Realtek High Definition Audio)",
                    "hostApi": 1,
                    "maxInputChannels": 0,
                    "maxOutputChannels": 2,
                },
            }
            return devices[index]

        def terminate(self) -> None:
            return None

    monkeypatch.setattr("core.audio_utils.sys.platform", "win32")
    monkeypatch.setattr("core.audio_utils.pyaudio.PyAudio", FakePyAudio)
    inputs, outputs = get_audio_devices()
    assert inputs == [(2, "Microphone (Realtek High Definition Audio)")]
    assert outputs == [(3, "Speakers (Realtek High Definition Audio)")]


def test_build_audio_device_menu_rows_marks_active_device() -> None:
    devices = [(0, "Built-in Mic"), (2, "USB Headset")]
    rows = build_audio_device_menu_rows(devices, active_index=2)
    assert rows == [
        (0, "   Built-in Mic"),
        (2, "✓  USB Headset"),
    ]
