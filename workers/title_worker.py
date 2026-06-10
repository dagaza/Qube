# --- workers/title_worker.py ---
# Deprecated shim — titling lives on SidecarLlmWorker (Qwen3 1.7B CPU).
from workers.sidecar_llm_worker import SidecarLlmWorker as TitleWorker  # noqa: F401
