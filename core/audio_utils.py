# core/audio_utils.py
from __future__ import annotations

import logging

import pyaudio

logger = logging.getLogger("Qube.Audio")


def _device_input_info(p: pyaudio.PyAudio, device_index: int | None) -> dict | None:
    try:
        if device_index is None:
            return p.get_default_input_device_info()
        return p.get_device_info_by_index(device_index)
    except Exception as exc:
        logger.debug("Input device lookup failed (index=%s): %s", device_index, exc)
        return None


def iter_mic_open_candidates(
    p: pyaudio.PyAudio,
    preferred_index: int | None,
) -> list[tuple[int | None, int, str]]:
    """Return ordered (device_index, channel_count, label) attempts for mono capture."""
    ordered_indices: list[int | None] = []
    if preferred_index is not None:
        ordered_indices.append(preferred_index)
    ordered_indices.append(None)

    candidates: list[tuple[int | None, int, str]] = []
    seen: set[int | None] = set()
    for idx in ordered_indices:
        if idx in seen:
            continue
        seen.add(idx)
        info = _device_input_info(p, idx)
        if not info:
            continue
        max_channels = int(info.get("maxInputChannels", 0) or 0)
        if max_channels <= 0:
            continue
        channels = 1 if max_channels >= 1 else max_channels
        label = str(info.get("name") or f"device {idx}")
        candidates.append((idx, channels, label))
    return candidates


def classify_mic_open_error(exc: Exception | None) -> str:
    """Map PyAudio/ALSA failures to a short user-facing mic status line."""
    if exc is None:
        return "Mic Error: unavailable"
    text = str(exc).lower()
    if "invalid number of channels" in text or "-9998" in text:
        return "Mic Error: device busy or unavailable"
    if "device unavailable" in text or "-9985" in text:
        return "Mic Error: device in use by another app"
    if "unanticipated host error" in text or "alsa" in text or "-9999" in text:
        return "Mic Error: ALSA busy"
    compact = str(exc).strip()
    if len(compact) > 80:
        compact = compact[:77] + "..."
    return f"Mic Error: {compact}"


def get_input_devices() -> list[tuple[int, str]]:
    """Returns a list of tuples: (real_device_index, display_name) for inputs."""
    p = pyaudio.PyAudio()
    devices = []
    
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info.get('maxInputChannels', 0) > 0:
            name = info.get('name', f'Unknown Device {i}')
            devices.append((i, f"Input {i}: {name}"))
            
    p.terminate()
    return devices

def get_output_devices() -> list[tuple[int, str]]:
    """Returns a list of tuples: (real_device_index, display_name) for outputs."""
    p = pyaudio.PyAudio()
    devices = []
    
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info.get('maxOutputChannels', 0) > 0:
            name = info.get('name', f'Unknown Device {i}')
            devices.append((i, f"Device {i}: {name}"))
            
    p.terminate()
    return devices