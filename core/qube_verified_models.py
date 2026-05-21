"""Load the Model Manager \"Qube Verified\" curated hub catalog from JSON."""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.hf_publisher_branding import TRUSTED_PUBLISHERS, HuggingFaceBrandingResolver

logger = logging.getLogger("Qube.QubeVerifiedModels")

# Last-resort fallback if bundled + user JSON are missing or invalid.
_DEFAULT_CATALOG_RAW: list[dict[str, Any]] = [
    {
        "repo_id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "title": "Llama 3.1 8B Instruct",
    },
]


@dataclass(frozen=True)
class CatalogEntry:
    """Curated Model Manager card: brand metadata + GGUF download repo(s)."""

    catalog_id: str
    title: str
    description: str
    publisher: str
    gguf_repo: str
    gguf_repos: tuple[str, ...] = ()

    @property
    def is_catalog_card(self) -> bool:
        """True when entry uses explicit publisher and/or non-empty description."""
        return bool(self.publisher.strip() or self.description.strip())


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def bundled_verified_models_path() -> Path:
    return _project_root() / "assets" / "qube_verified_models.json"


def default_user_verified_models_path() -> Path:
    return Path.home() / ".qube" / "qube_verified_models.json"


def ensure_user_verified_models_seeded(user_path: Path | None = None) -> Path:
    """Copy bundled JSON to ~/.qube on first run; return the user path."""
    path = Path(user_path) if user_path else default_user_verified_models_path()
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    seed = bundled_verified_models_path()
    if seed.is_file():
        shutil.copyfile(seed, path)
        return path
    path.write_text(json.dumps(_DEFAULT_CATALOG_RAW, indent=2) + "\n", encoding="utf-8")
    return path


def _logo_exists(filename: str) -> bool:
    return (_project_root() / "assets" / "logos" / filename).is_file()


def _slug_catalog_id(title: str, gguf_repo: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", str(title or "").strip().lower()).strip("-")
    if base:
        return base
    return re.sub(r"[^a-z0-9]+", "-", gguf_repo.replace("/", "-").lower()).strip("-") or "model"


def _coerce_gguf_repos(item: dict[str, Any], primary: str) -> tuple[str, ...]:
    seen: list[str] = []
    primary_s = str(primary or "").strip()
    if primary_s:
        seen.append(primary_s)
    raw_list = item.get("gguf_repos")
    if isinstance(raw_list, list):
        for r in raw_list:
            rid = str(r or "").strip()
            if rid and rid not in seen:
                seen.append(rid)
    return tuple(seen)


def _parse_catalog_item(item: dict[str, Any]) -> CatalogEntry | None:
    gguf_repo = str(item.get("gguf_repo") or item.get("repo_id") or "").strip()
    if not gguf_repo:
        logger.warning("Skipping catalog row with no gguf_repo/repo_id: %r", item)
        return None
    title = str(item.get("title", "") or "").strip() or gguf_repo
    catalog_id = str(item.get("id") or item.get("catalog_id") or "").strip() or _slug_catalog_id(title, gguf_repo)
    description = str(item.get("description", "") or "").strip()
    publisher = str(item.get("publisher", "") or "").strip().lower()
    gguf_repos = _coerce_gguf_repos(item, gguf_repo)
    primary = gguf_repos[0] if gguf_repos else gguf_repo
    return CatalogEntry(
        catalog_id=catalog_id,
        title=title,
        description=description,
        publisher=publisher,
        gguf_repo=primary,
        gguf_repos=gguf_repos,
    )


def _unwrap_catalog_root(parsed: Any) -> Any:
    """
    Accept common JSON shapes:
    - ``{"schema_version": 1, "models": [...]}``
    - legacy ``[{repo_id, title}, ...]``
    - mistaken double-wrap ``[{"schema_version": 1, "models": [...]}]``
    """
    if isinstance(parsed, list):
        if len(parsed) == 1 and isinstance(parsed[0], dict):
            sole = parsed[0]
            if isinstance(sole.get("models"), list) or isinstance(sole.get("entries"), list):
                return sole
        return parsed
    return parsed


def _catalog_schema_version(raw: Any) -> int:
    root = _unwrap_catalog_root(raw)
    if isinstance(root, dict):
        try:
            return int(root.get("schema_version") or 0)
        except (TypeError, ValueError):
            return 0
    return 0


def _is_legacy_catalog_file(raw: Any) -> bool:
    """True for pre-catalog stubs (repo_id + title only), not schema v1 objects."""
    root = _unwrap_catalog_root(raw)
    if isinstance(root, dict):
        return not root.get("schema_version")
    if isinstance(root, list):
        if not root:
            return True
        return all(
            isinstance(x, dict)
            and bool(str(x.get("repo_id", "") or "").strip() or str(x.get("title", "") or "").strip())
            and not str(x.get("gguf_repo", "") or "").strip()
            and not str(x.get("publisher", "") or "").strip()
            for x in root
        )
    return True


def _should_replace_user_catalog_from_bundle(
    user_raw: Any,
    user_entries: list[CatalogEntry],
    bundled_raw: Any,
    bundled_entries: list[CatalogEntry],
) -> bool:
    """True when ``~/.qube`` should be overwritten from shipped ``assets/``."""
    if not bundled_entries:
        return False
    if _is_legacy_catalog_file(user_raw):
        return len(user_entries) < len(bundled_entries)
    bundled_version = _catalog_schema_version(bundled_raw)
    user_version = _catalog_schema_version(user_raw)
    if bundled_version > user_version:
        return True
    if len(bundled_entries) <= len(user_entries):
        return False
    # Assets grew (e.g. dev edited assets/); refresh if user catalog is a subset by id.
    user_ids = {e.catalog_id for e in user_entries}
    bundled_ids = {e.catalog_id for e in bundled_entries}
    return user_ids <= bundled_ids


def _maybe_refresh_user_catalog_from_bundle(user_path: Path) -> None:
    """
    Sync ``~/.qube/qube_verified_models.json`` from bundled ``assets/`` when stale.

    Refresh triggers:

    1. **Bundled file newer (mtime)** — editing ``assets/qube_verified_models.json`` in the
       repo (or shipping an app update) bumps its modification time above the copy in
       ``~/.qube``, so the next launch replaces the runtime file. No ``schema_version`` bump
       needed for local dev.

    2. **Schema / subset rules** — legacy stubs, higher ``schema_version`` in assets than
       in ``~/.qube``, or bundled catalog strictly superset by id (existing behavior).

    If you only edit ``~/.qube/qube_verified_models.json``, its mtime stays ahead of
    bundled (unless you edit assets again), so your custom list is preserved.
    """
    bundled = bundled_verified_models_path()
    if not bundled.is_file() or not user_path.is_file():
        return
    try:
        bundled_mtime = bundled.stat().st_mtime
        user_mtime = user_path.stat().st_mtime
        user_raw = json.loads(user_path.read_text(encoding="utf-8"))
        user_entries = _parse_catalog_raw(user_raw)
        bundled_raw = json.loads(bundled.read_text(encoding="utf-8"))
        bundled_entries = _parse_catalog_raw(bundled_raw)
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("Catalog refresh check skipped: %s", exc)
        return
    if not bundled_entries:
        return
    # Prefer shipped assets whenever they are newer than the ~/.qube copy (dev + updates).
    if bundled_mtime > user_mtime:
        try:
            shutil.copyfile(bundled, user_path)
            logger.info(
                "Refreshed user catalog from bundled assets (bundled newer than ~/.qube, %d models)",
                len(bundled_entries),
            )
        except OSError as exc:
            logger.warning("Could not refresh user catalog from bundle: %s", exc)
        return
    if not _should_replace_user_catalog_from_bundle(
        user_raw, user_entries, bundled_raw, bundled_entries
    ):
        return
    try:
        shutil.copyfile(bundled, user_path)
        logger.info(
            "Refreshed user catalog from bundled assets (%d -> %d models)",
            len(user_entries),
            len(bundled_entries),
        )
    except OSError as exc:
        logger.warning("Could not refresh user catalog from bundle: %s", exc)


def _parse_catalog_raw(raw: Any) -> list[CatalogEntry]:
    raw = _unwrap_catalog_root(raw)
    if isinstance(raw, dict):
        raw = raw.get("models", raw.get("entries"))
    if not isinstance(raw, list):
        return []
    out: list[CatalogEntry] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        entry = _parse_catalog_item(item)
        if entry is not None:
            out.append(entry)
    return out


def _read_catalog_file(path: Path) -> list[CatalogEntry]:
    text = path.read_text(encoding="utf-8")
    parsed = json.loads(text)
    return _parse_catalog_raw(parsed)


def branding_for_entry(
    entry: CatalogEntry,
    resolver: HuggingFaceBrandingResolver | None = None,
) -> dict[str, Any] | None:
    """
    Official logo from catalog ``publisher`` when allowlisted; else HF resolver on ``gguf_repo``.
    """
    pub = entry.publisher.strip().lower()
    if pub and pub in TRUSTED_PUBLISHERS:
        meta = TRUSTED_PUBLISHERS[pub]
        logo_name = str(meta.get("logo", "") or "").strip()
        if logo_name and _logo_exists(logo_name):
            return {
                "name": str(meta["name"]),
                "logo": f"/assets/logos/{logo_name}",
                "official": True,
            }
    res = resolver or HuggingFaceBrandingResolver()
    return res.resolve_for_model(entry.gguf_repo)


def load_qube_verified_models(*, user_path: Path | None = None) -> list[CatalogEntry]:
    """
    Return curated hub catalog for Model Manager (empty search / \"Qube Verified\").

    Reads ``~/.qube/qube_verified_models.json`` (seeded from ``assets/`` on first use).
    Syncs from ``assets/`` when the bundled file is newer (mtime) or other refresh rules
    apply — see ``_maybe_refresh_user_catalog_from_bundle``.
    Falls back to the bundled assets file, then a minimal built-in default.
    """
    user = ensure_user_verified_models_seeded(user_path)
    _maybe_refresh_user_catalog_from_bundle(user)
    for candidate in (user, bundled_verified_models_path()):
        if not candidate.is_file():
            continue
        try:
            entries = _read_catalog_file(candidate)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not load verified models from %s: %s", candidate, exc)
            continue
        if entries:
            return entries
        logger.warning("Verified models file has no valid entries: %s", candidate)
    logger.warning("Using built-in default Qube Verified models list")
    return _parse_catalog_raw(_DEFAULT_CATALOG_RAW)


# Back-compat helpers for tests / legacy callers
def _normalize_entries(raw: Any) -> list[dict[str, str]]:
    entries = _parse_catalog_raw(raw if isinstance(raw, list) else [])
    if not entries and isinstance(raw, dict):
        entries = _parse_catalog_raw(raw.get("models", raw.get("entries")))
    return [{"repo_id": e.gguf_repo, "title": e.title} for e in entries]


def _read_verified_models_file(path: Path) -> list[dict[str, str]]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(parsed, dict):
        parsed = parsed.get("models", parsed.get("entries"))
    return _normalize_entries(parsed)
