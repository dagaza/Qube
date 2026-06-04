"""Canonical install, bundled resource, and per-user data paths."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def install_root() -> Path:
    """Directory containing the app entry point (repo root in dev, exe dir when frozen)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def resource_path(*parts: str) -> Path:
    """Path to bundled read-only assets shipped with the application."""
    return install_root().joinpath(*parts)


def user_data_root() -> Path:
    """Writable per-user data directory (DB, models, logs — not next to _internal)."""
    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA")
        root = Path(local_app_data) / "Qube" if local_app_data else Path.home() / ".qube"
    else:
        root = Path.home() / ".qube"
    root.mkdir(parents=True, exist_ok=True)
    return root


def models_root() -> Path:
    path = user_data_root() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_db_path() -> Path:
    return user_data_root() / "qube_data.db"


def default_lancedb_dir() -> Path:
    path = user_data_root() / "data" / "lancedb"
    path.mkdir(parents=True, exist_ok=True)
    return path


def logs_dir() -> Path:
    path = user_data_root() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path
