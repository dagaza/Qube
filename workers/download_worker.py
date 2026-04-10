from PyQt6.QtCore import QThread, pyqtSignal
import os
import requests
import logging

logger = logging.getLogger("Qube.Downloader")

class DownloadWorker(QThread):
    """
    Background worker for fetching massive AI model weights.
    Streams data in chunks to preserve RAM and allows for safe interruptions.
    """
    # Signals to update the UI
    progress_update = pyqtSignal(int, str)  # Percentage (0-100), Status text
    download_complete = pyqtSignal(str, bool) # Model ID, Success flag
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._cancel_requested = False
        self.model_id = ""
        self.files_to_download = {} 
        
    def start_download(self, model_id: str, files_dict: dict):
        """
        Expects a dictionary mapping local file paths to their download URLs.
        Example: {'models/tts/f5/model.safetensors': 'https://huggingface...'}
        """
        if self.isRunning():
            logger.warning("Downloader is already running. Please wait.")
            return
            
        self.model_id = model_id
        self.files_to_download = files_dict
        self._cancel_requested = False
        self.start()

    def cancel_download(self):
        """Thread-safe kill switch."""
        self._cancel_requested = True
        logger.info(f"User requested download cancellation for {self.model_id}.")

    def run(self):
        total_files = len(self.files_to_download)
        
        for index, (file_path, url) in enumerate(self.files_to_download.items()):
            # 1. Check Kill-Switch
            if self._cancel_requested:
                self.download_complete.emit(self.model_id, False)
                return
                
            # 2. Ensure folder structure exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            file_basename = os.path.basename(file_path)
            
            try:
                # 3. Stream the download
                response = requests.get(url, stream=True, timeout=10)
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                
                downloaded_size = 0
                
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        # 4. Check Kill-Switch mid-download
                        if self._cancel_requested:
                            f.close()
                            os.remove(file_path) # Clean up the corrupted partial file
                            logger.info(f"Cleaned up partial file: {file_basename}")
                            self.download_complete.emit(self.model_id, False)
                            return
                            
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            
                            # 5. Calculate precise global progress across multiple files
                            if total_size > 0:
                                file_progress = (downloaded_size / total_size)
                                overall_progress = int(((index + file_progress) / total_files) * 100)
                                
                                # Only emit every ~1% to avoid flooding the UI thread with signals
                                if downloaded_size % (1024 * 1024) < 8192: 
                                    self.progress_update.emit(overall_progress, f"Downloading {file_basename}...")

            except Exception as e:
                logger.error(f"Failed to download {file_basename}: {e}")
                self.error_occurred.emit(f"Network error while downloading {file_basename}.")
                self.download_complete.emit(self.model_id, False)
                return

        # Success!
        self.progress_update.emit(100, f"{self.model_id} installed successfully!")
        self.download_complete.emit(self.model_id, True)