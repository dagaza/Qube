import os
import sys
import threading
import subprocess
import glob

class GPUMonitor:
    def __init__(self):
        self.backend = "none"
        self.current_load = 0.0
        self.pynvml = None
        self.handle = None

        # --- 1. Try NVIDIA (Cross-Platform) ---
        try:
            import pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.pynvml = pynvml
            self.backend = "nvidia"
            print("[TELEMETRY] NVIDIA GPU detected via NVML.")
            return
        except Exception:
            pass

        # --- 2. Try OS-Specific Native Fallbacks ---
        if sys.platform.startswith('linux'):
            # Linux: Ultra-fast kernel file read
            self.amd_sysfs_path = None
            
            # APUs often map to card1 or higher. We use glob to scan all available 
            # DRM cards and lock onto the one that provides active telemetry.
            possible_paths = glob.glob("/sys/class/drm/card*/device/gpu_busy_percent")
            
            for path in possible_paths:
                try:
                    # Test if the file is readable and returns a valid number
                    with open(path, 'r') as f:
                        float(f.read().strip())
                    self.amd_sysfs_path = path
                    self.backend = "linux_sysfs"
                    print(f"[TELEMETRY] AMD GPU/APU detected via Linux sysfs at: {path}")
                    break
                except Exception:
                    continue
                    
            if not self.amd_sysfs_path:
                print("[TELEMETRY] No supported Linux GPU backend found. Defaulting to 0%.")

        elif sys.platform == 'win32':
            # Windows: Background thread using native typeperf utility
            self.backend = "windows_perf"
            print("[TELEMETRY] Windows Non-NVIDIA GPU detected. Starting telemetry thread.")
            self._stop_event = threading.Event()
            self._win_thread = threading.Thread(target=self._poll_windows_gpu, daemon=True)
            self._win_thread.start()

    def _poll_windows_gpu(self):
        """Runs continuously in the background on Windows to prevent UI blocking."""
        cmd = ['typeperf', r'\GPU Engine(*engtype_3D)\Utilization Percentage', '-si', '1']
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            
            while not self._stop_event.is_set():
                line = process.stdout.readline()
                if not line:
                    break
                
                if "," in line and not line.startswith('"(PDH-CSV 4.0)"'):
                    parts = line.strip().split(',')
                    try:
                        vals = [float(x.strip('" ')) for x in parts[1:] if x.strip('" ')]
                        if vals:
                            self.current_load = max(vals)
                    except ValueError:
                        pass
        except Exception as e:
            print(f"[TELEMETRY] Windows GPU polling failed: {e}")

    def get_load(self):
        """Returns the current GPU load as a float (0.0 to 100.0) in microseconds."""
        if self.backend == "nvidia":
            try:
                info = self.pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                return float(info.gpu)
            except Exception:
                return 0.0
                
        elif self.backend == "linux_sysfs":
            try:
                with open(self.amd_sysfs_path, 'r') as f:
                    return float(f.read().strip())
            except Exception:
                return 0.0
                
        elif self.backend == "windows_perf":
            return self.current_load
            
        return 0.0

    def cleanup(self):
        """Safely stops the background thread if the application closes."""
        if self.backend == "windows_perf":
            self._stop_event.set()