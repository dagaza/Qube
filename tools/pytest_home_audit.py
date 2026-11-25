"""Pytest plugin: record every write under the real Qube data directory, per test (plan S16/S40).

Usage (repo root):

    QUBE_HOME_AUDIT_OUT=<file.jsonl> PYTHONPATH=tools \\
        .venv/bin/python -m pytest -p pytest_home_audit -m "not packaging" tests

The plugin loads before ``tests/conftest.py`` moves ``HOME`` to a throwaway
directory (S40), so it watches the developer's real ``~/.qube`` (Windows:
``%LOCALAPPDATA%\\Qube``). An audit hook records opens for writing, removes,
renames, mkdirs and rmtrees whose resolved path is inside it, one JSON line per
(kind, path, test). With S40 in place the file should stay empty. A Qube app
running at the same time writes its own logs; close it first.
"""

from __future__ import annotations

import json
import os
import sys

if sys.platform == "win32":
    _ROOT = os.path.realpath(
        os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local"), "Qube")
    )
else:
    _ROOT = os.path.realpath(os.path.expanduser("~/.qube"))
_OUT = os.environ.get("QUBE_HOME_AUDIT_OUT", "")
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
_PATH_EVENTS = ("os.remove", "os.rename", "os.replace", "os.mkdir", "os.rmdir", "shutil.rmtree")
_current = {"id": "<collection>"}
_seen: set[tuple[str, str, str]] = set()


def _record(kind: str, path: object) -> None:
    try:
        resolved = os.path.realpath(os.fsdecode(path))  # type: ignore[arg-type]
    except Exception:
        return
    if resolved != _ROOT and not resolved.startswith(_ROOT + os.sep):
        return
    key = (kind, resolved, _current["id"])
    if key in _seen:
        return
    _seen.add(key)
    with open(_OUT, "a", encoding="utf-8") as fh:  # _OUT must be outside the audited root
        fh.write(json.dumps({"kind": kind, "path": resolved, "test": _current["id"]}) + "\n")


def _hook(event: str, args: tuple) -> None:
    try:
        if event == "open":
            path, mode, flags = args
            if isinstance(path, int):
                return
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                mode is None and isinstance(flags, int) and bool(flags & _WRITE_FLAGS)
            )
            if writing:
                _record("write", path)
        elif event in _PATH_EVENTS:
            _record(event, args[0])
    except Exception:
        pass


if _OUT:
    sys.addaudithook(_hook)


def pytest_runtest_setup(item) -> None:
    _current["id"] = item.nodeid


def pytest_runtest_teardown(item) -> None:
    _current["id"] = item.nodeid + " (teardown)"
