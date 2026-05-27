# --- workers/title_worker.py ---
# Deprecated shim — titling lives on SidecarLlmWorker (Qwen2-0.5B CPU).
from workers.sidecar_llm_worker import SidecarLlmWorker as TitleWorker  # noqa: F401
