# workers/internet_worker.py
from PyQt6.QtCore import QThread, pyqtSignal
from mcp.internet_tool import InternetTool
import logging

logger = logging.getLogger("Qube.Workers.InternetWorker")

class InternetWorker(QThread):
    """
    QThread Worker for non-blocking Internet search queries.
    Emits results as they arrive to avoid freezing the UI.
    """
    search_result = pyqtSignal(str)  # Signal to send final results to UI
    search_error = pyqtSignal(str)   # Signal to notify UI of errors

    def __init__(self, query: str, max_results: int = 3, parent=None):
        super().__init__(parent)
        self.query = query
        self.max_results = max_results
        self._stop_flag = False
        self.tool = InternetTool(max_results=self.max_results)

    def run(self):
        """
        Execute the internet search in a separate thread.
        Emits search_result or search_error signals.
        """
        try:
            if self._stop_flag:
                return
            result = self.tool.search(self.query)
            if self._stop_flag:
                return
            self.search_result.emit(result)
        except Exception as e:
            logger.error(f"InternetWorker failed: {e}")
            self.search_error.emit(str(e))

    def stop(self):
        """Set stop flag to safely terminate the thread if needed."""
        self._stop_flag = True