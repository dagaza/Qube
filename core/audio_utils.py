# core/audio_utils.py
from __future__ import annotations

import logging
import re
import sys

import pyaudio

logger = logging.getLogger("Qube.Audio")

_WINDOWS_EXCLUDED_NAME_FRAGMENTS = (
    "microsoft sound mapper",
    "primary sound capture driver",
    "primary sound driver",
    "loopback",
)


def _host_api_index_for_type(p: pyaudio.PyAudio, api_type: int) -> int | None:
    for index in range(p.get_host_api_count()):
        info = p.get_host_api_info_by_index(index)
        if int(info.get("type", -1)) == api_type:
            return index
    return None


def _preferred_windows_host_api_index(p: pyaudio.PyAudio) -> int | None:
    for api_type in (pyaudio.paWASAPI, pyaudio.paDirectSound, pyaudio.paMME):
        host_api_index = _host_api_index_for_type(p, api_type)
        if host_api_index is not None:
            return host_api_index
    return None


def _normalize_device_name(name: str) -> str:
    normalized = re.sub(r"\s+", " ", name.strip().lower())
    return normalized.rstrip(" .")


def _is_excluded_device_name(name: str) -> bool:
    lowered = name.lower()
    return any(fragment in lowered for fragment in _WINDOWS_EXCLUDED_NAME_FRAGMENTS)


def _iter_user_devices(
    p: pyaudio.PyAudio,
) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """Return deduplicated (index, display_name) lists for inputs and outputs."""
    preferred_host_api: int | None = None
    if sys.platform == "win32":
        preferred_host_api = _preferred_windows_host_api_index(p)

    inputs: list[tuple[int, str]] = []
    outputs: list[tuple[int, str]] = []
    seen_input_names: set[str] = set()
    seen_output_names: set[str] = set()

    for index in range(p.get_device_count()):
        info = p.get_device_info_by_index(index)
        if preferred_host_api is not None and info.get("hostApi") != preferred_host_api:
            continue

        raw_name = str(info.get("name") or f"Unknown Device {index}")
        if _is_excluded_device_name(raw_name):
            continue

        normalized_name = _normalize_device_name(raw_name)
        if int(info.get("maxInputChannels", 0) or 0) > 0:
            if normalized_name in seen_input_names:
                continue
            seen_input_names.add(normalized_name)
            inputs.append((index, raw_name))

        if int(info.get("maxOutputChannels", 0) or 0) > 0:
            if normalized_name in seen_output_names:
                continue
            seen_output_names.add(normalized_name)
            outputs.append((index, raw_name))

    if preferred_host_api is not None and not inputs and not outputs:
        logger.debug(
            "No devices after Windows host API filter (hostApi=%s); falling back to deduped full list",
            preferred_host_api,
        )
        return _iter_user_devices_without_host_api_filter(p)

    return inputs, outputs


def _iter_user_devices_without_host_api_filter(
    p: pyaudio.PyAudio,
) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    inputs: list[tuple[int, str]] = []
    outputs: list[tuple[int, str]] = []
    seen_input_names: set[str] = set()
    seen_output_names: set[str] = set()

    for index in range(p.get_device_count()):
        info = p.get_device_info_by_index(index)
        raw_name = str(info.get("name") or f"Unknown Device {index}")
        if _is_excluded_device_name(raw_name):
            continue

        normalized_name = _normalize_device_name(raw_name)
        if int(info.get("maxInputChannels", 0) or 0) > 0:
            if normalized_name in seen_input_names:
                continue
            seen_input_names.add(normalized_name)
            inputs.append((index, raw_name))

        if int(info.get("maxOutputChannels", 0) or 0) > 0:
            if normalized_name in seen_output_names:
                continue
            seen_output_names.add(normalized_name)
            outputs.append((index, raw_name))

    return inputs, outputs


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


def build_audio_device_menu_rows(
    devices: list[tuple[int, str]],
    active_index: int | None,
) -> list[tuple[int, str]]:
    """Build (device_index, menu_label) rows for audio device pickers."""
    rows: list[tuple[int, str]] = []
    for idx, name in devices:
        prefix = "✓  " if idx == active_index else "   "
        rows.append((idx, f"{prefix}{name}"))
    return rows


def get_audio_devices() -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """Enumerate user-facing input and output devices in a single PyAudio session."""
    p = pyaudio.PyAudio()
    try:
        return _iter_user_devices(p)
    finally:
        p.terminate()


def get_input_devices() -> list[tuple[int, str]]:
    """Returns a list of tuples: (real_device_index, display_name) for inputs."""
    inputs, _outputs = get_audio_devices()
    return inputs


def get_output_devices() -> list[tuple[int, str]]:
    """Returns a list of tuples: (real_device_index, display_name) for outputs."""
    _inputs, outputs = get_audio_devices()
    return outputs