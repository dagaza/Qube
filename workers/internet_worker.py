# workers/internet_worker.py
import logging
import time

from PyQt6.QtCore import QThread, pyqtSignal

from core.web_search_audit import build_standalone_audit_event, record_web_search_audit
from mcp.internet_tool import search_internet

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

    def run(self):
        """
        Execute the internet search in a separate thread.
        Emits search_result or search_error signals.
        """
        try:
            if self._stop_flag:
                return
                
            logger.info(f"Executing manual web search for: '{self.query}'")

            t0 = time.time()
            # 🔑 FIX 1: Call the standalone function directly! 
            # (Note: if your search_internet function accepts max_results, pass it here)
            raw_results = search_internet(self.query)
            latency_ms = (time.time() - t0) * 1000
            
            # Check flag again in case user canceled during the HTTP request
            if self._stop_flag:
                return

            raw_for_audit = (
                [dict(r) for r in raw_results if isinstance(r, dict)]
                if isinstance(raw_results, list)
                else None
            )
            record_web_search_audit(
                build_standalone_audit_event(
                    query=self.query,
                    raw_results=raw_for_audit,
                    latency_ms=latency_ms,
                )
            )

            if raw_results:
                # 🔑 FIX 2: Stringify the list so the LLM can actually read it
                if isinstance(raw_results, list):
                    formatted_results = "\n\n".join([str(item) for item in raw_results])
                else:
                    formatted_results = str(raw_results)
                    
                self.search_result.emit(formatted_results)
            else:
                self.search_error.emit("DuckDuckGo returned no results for this query.")

        except Exception as e:
            logger.error(f"InternetWorker failed: {e}")
            record_web_search_audit(
                build_standalone_audit_event(
                    query=self.query,
                    raw_results=None,
                    error=str(e),
                )
            )
            self.search_error.emit(str(e))

    def stop(self):
        """Set stop flag to safely terminate the thread if needed."""
        self._stop_flag = True