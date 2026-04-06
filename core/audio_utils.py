# core/audio_utils.py
import pyaudio

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