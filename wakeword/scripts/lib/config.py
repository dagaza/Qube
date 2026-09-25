"""Config loading for the wake word pipeline.

Configs are YAML (see ``configs/*.yaml``). PyYAML is part of the pinned training
environment; we import lazily and raise a helpful error if it is missing so the
license gate (which needs no YAML) still runs in a bare checkout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "PyYAML is required to load configs. Install the training environment:\n"
            "  pip install -r environment/requirements-training.txt"
        ) from exc

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Config {config_path} did not parse to a mapping.")
    return data


def require(config: dict[str, Any], *keys: str) -> Any:
    """Fetch a nested key path, raising a clear error if absent."""
    node: Any = config
    trail: list[str] = []
    for key in keys:
        trail.append(key)
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"Missing config key: {'.'.join(trail)}")
        node = node[key]
    return node
