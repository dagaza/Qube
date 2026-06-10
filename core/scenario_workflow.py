"""Guided two-phase scenario comparison workflow (Qube pathway, then external HTTP)."""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import requests

from core.conversation_replay import Scenario
from core.scenario_loader import load_scenario, scenario_id_for, session_file_path

logger = logging.getLogger("Qube.ScenarioWorkflow")

ReadinessFn = Callable[[], tuple[bool, str]]
QubeRunnerFn = Callable[[str], str]
ExternalLauncherFn = Callable[[str, str, str, str], int]
SessionComparerFn = Callable[[str, str], Any]

DEFAULT_EXTERNAL_API_URL = "http://localhost:1234/v1/chat/completions"
DEFAULT_WAIT_FOR_API_SECONDS = 900.0


def models_url_from_chat_completions(api_url: str) -> str:
    raw = str(api_url or DEFAULT_EXTERNAL_API_URL).strip()
    if raw.endswith("/v1/chat/completions"):
        return raw[: -len("/v1/chat/completions")] + "/v1/models"
    parsed = urlparse(raw)
    base = f"{parsed.scheme}://{parsed.netloc}".rstrip("/")
    return f"{base}/v1/models"


def wait_for_external_api(
    api_url: str,
    *,
    timeout_seconds: float = DEFAULT_WAIT_FOR_API_SECONDS,
    poll_interval_seconds: float = 2.0,
) -> bool:
    """Poll LM Studio (or compatible) ``/v1/models`` until it responds or timeout."""
    models_url = models_url_from_chat_completions(api_url)
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    while time.monotonic() < deadline:
        try:
            response = requests.get(models_url, timeout=5)
            if response.status_code == 200:
                logger.info("[ScenarioWorkflow] external API ready at %s", models_url)
                return True
        except requests.RequestException as exc:
            logger.debug("[ScenarioWorkflow] waiting for external API: %s", exc)
        time.sleep(max(0.5, float(poll_interval_seconds)))
    logger.error(
        "[ScenarioWorkflow] timed out waiting for external API at %s",
        models_url,
    )
    return False


def qube_pathway_ready(window: Any) -> tuple[bool, str]:
    """Return whether the Qube pathway test can start (model loaded in app)."""
    from core.app_settings import get_engine_mode

    mode = get_engine_mode()
    if mode == "internal":
        if getattr(window, "_native_model_loading", False) or getattr(
            window, "_native_model_unloading", False
        ):
            return False, "Wait for the model to finish loading or ejecting."
        if not getattr(window, "_native_model_loaded_success", False):
            return (
                False,
                "Load a GGUF model from the toolbar before starting the Qube pathway test.",
            )
        native_engine = getattr(window, "_native_engine", None)
        snap = (
            native_engine.get_model_reasoning_telemetry()
            if native_engine is not None
            else None
        )
        if not bool((snap or {}).get("loaded")):
            return (
                False,
                "Load a GGUF model from the toolbar before starting the Qube pathway test.",
            )
        return True, ""
    return (
        True,
        "Qube is using an external AI engine — ensure the remote endpoint is configured.",
    )


def suggested_external_model_name(window: Any, scenario: Scenario) -> str:
    if str(scenario.model or "").strip():
        return str(scenario.model).strip()
    from core.app_settings import get_engine_mode, get_internal_model_path

    if get_engine_mode() == "internal":
        path = str(get_internal_model_path() or "").strip()
        if path:
            return Path(path).stem
    native_engine = getattr(window, "_native_engine", None)
    snap = (
        native_engine.get_model_reasoning_telemetry() if native_engine is not None else None
    )
    model_name = str((snap or {}).get("model_name") or "").strip()
    if model_name:
        return Path(model_name).stem
    return ""


def resolve_external_api_url(scenario: Scenario) -> str:
    return str(scenario.external_api_url or DEFAULT_EXTERNAL_API_URL).strip()


def build_external_replay_command(
    scenario_path: str,
    *,
    repo_root: Path | str,
    model: str = "",
    api_url: str = "",
    qube_session_path: str = "",
    wait_for_api_seconds: float = DEFAULT_WAIT_FOR_API_SECONDS,
) -> list[str]:
    """Argv for a detached external-phase replay (+ optional offline compare)."""
    root = Path(repo_root)
    cmd = [
        sys.executable,
        "-m",
        "tools.run_scenario_replay",
        "--scenario",
        str(scenario_path),
        "--backend",
        "external",
        "--wait-for-api",
        str(wait_for_api_seconds),
    ]
    if model:
        cmd.extend(["--model", model])
    if api_url:
        cmd.extend(["--api-url", api_url])
    if qube_session_path:
        cmd.extend(["--compare-with", qube_session_path])
    return cmd


def expected_qube_session_path(scenario: Scenario | str | Path) -> Path:
    if not isinstance(scenario, Scenario):
        scenario = load_scenario(scenario)
    return session_file_path(scenario_id_for(scenario), "qube")
