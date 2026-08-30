from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from core.paths import resource_path


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS model_capabilities (
  model_id TEXT PRIMARY KEY,
  capabilities_json TEXT NOT NULL,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS capability_overrides (
  model_id TEXT PRIMARY KEY,
  overrides_json TEXT NOT NULL,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ground_truth_capabilities (
  model_id TEXT PRIMARY KEY,
  capabilities_json TEXT NOT NULL,
  source TEXT DEFAULT 'runtime_detection',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def default_system_data_dir() -> Path:
    # Deliberately isolated from user DB/LanceDB locations.
    return Path.home() / ".qube" / "system_data"


def _workspace_seed_registry_path() -> Path:
    return resource_path("system_data", "curated_registry.json")


def _workspace_missed_models_path() -> Path:
    return resource_path("system_data", "missed_models.json")


def _workspace_learned_registry_path() -> Path:
    return resource_path("system_data", "learned_capabilities.json")


class SystemCapabilitiesStore:
    def __init__(self, system_data_dir: str | Path | None = None):
        self.system_data_dir = Path(system_data_dir) if system_data_dir else default_system_data_dir()
        self.system_data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.system_data_dir / "capabilities.db"
        self.registry_path = self.system_data_dir / "curated_registry.json"
        self.missed_models_path = self.system_data_dir / "missed_models.json"
        self.learned_registry_path = self.system_data_dir / "learned_capabilities.json"
        self.publisher_guidance_path = self.system_data_dir / "publisher_guidance.json"
        self.model_hf_provenance_path = self.system_data_dir / "model_hf_provenance.json"
        self._ensure_registry_seeded()
        self._ensure_missed_models_seeded()
        self._ensure_learned_registry_seeded()
        self._ensure_publisher_guidance_seeded()
        self._ensure_model_hf_provenance_seeded()
        self.init_db()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()

    def get_cached_capabilities(self, model_id: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT capabilities_json FROM model_capabilities WHERE model_id = ?",
                (model_id,),
            ).fetchone()
        if row is None:
            return None
        try:
            parsed = json.loads(row["capabilities_json"])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

    def upsert_capabilities(self, model_id: str, capabilities: dict[str, Any]) -> None:
        payload = json.dumps(capabilities, ensure_ascii=True)
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO model_capabilities(model_id, capabilities_json, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(model_id)
                DO UPDATE SET capabilities_json = excluded.capabilities_json, updated_at = CURRENT_TIMESTAMP
                """,
                (model_id, payload),
            )
            conn.commit()

    def list_cached_capabilities(self) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT model_id, capabilities_json, updated_at FROM model_capabilities ORDER BY updated_at DESC"
            ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            try:
                parsed = json.loads(r["capabilities_json"])
            except json.JSONDecodeError:
                continue
            out.append(
                {
                    "id": r["model_id"],
                    "capabilities": parsed,
                    "updated_at": r["updated_at"],
                }
            )
        return out

    def set_override(self, model_id: str, overrides: dict[str, bool]) -> None:
        payload = json.dumps(overrides, ensure_ascii=True)
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO capability_overrides(model_id, overrides_json, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(model_id)
                DO UPDATE SET overrides_json = excluded.overrides_json, updated_at = CURRENT_TIMESTAMP
                """,
                (model_id, payload),
            )
            conn.commit()

    def get_override(self, model_id: str) -> dict[str, bool]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT overrides_json FROM capability_overrides WHERE model_id = ?",
                (model_id,),
            ).fetchone()
        if row is None:
            return {}
        try:
            parsed = json.loads(row["overrides_json"])
            if isinstance(parsed, dict):
                return {str(k): bool(v) for k, v in parsed.items()}
        except json.JSONDecodeError:
            pass
        return {}

    def set_ground_truth(self, model_id: str, capabilities: dict[str, bool], source: str = "runtime_detection") -> None:
        payload = json.dumps({str(k): bool(v) for k, v in capabilities.items()}, ensure_ascii=True)
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO ground_truth_capabilities(model_id, capabilities_json, source, created_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(model_id)
                DO UPDATE SET capabilities_json = excluded.capabilities_json, source = excluded.source, created_at = CURRENT_TIMESTAMP
                """,
                (model_id, payload, source),
            )
            conn.commit()

    def get_ground_truth(self, model_id: str) -> dict[str, bool]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT capabilities_json FROM ground_truth_capabilities WHERE model_id = ?",
                (model_id,),
            ).fetchone()
        if row is None:
            return {}
        try:
            parsed = json.loads(row["capabilities_json"])
        except json.JSONDecodeError:
            return {}
        if not isinstance(parsed, dict):
            return {}
        return {str(k): bool(v) for k, v in parsed.items()}

    def load_curated_registry(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {"exact": {}, "patterns": []}
        if not isinstance(raw, dict):
            return {"exact": {}, "patterns": []}

        # Backward compatibility: old flat exact registry.
        if "exact" not in raw and "patterns" not in raw:
            exact: dict[str, dict[str, bool]] = {}
            for model_id, caps in raw.items():
                if not isinstance(caps, dict):
                    continue
                exact[str(model_id).strip().lower()] = {str(k): bool(v) for k, v in caps.items()}
            return {"exact": exact, "patterns": []}

        out_exact: dict[str, dict[str, bool]] = {}
        for model_id, caps in (raw.get("exact") or {}).items():
            if not isinstance(caps, dict):
                continue
            out_exact[str(model_id).strip().lower()] = {str(k): bool(v) for k, v in caps.items()}

        out_patterns: list[dict[str, Any]] = []
        for p in (raw.get("patterns") or []):
            if not isinstance(p, dict):
                continue
            caps = p.get("capabilities") or {}
            if not isinstance(caps, dict):
                continue
            out_patterns.append(
                {
                    "match": str(p.get("match") or "").strip().lower(),
                    "type": str(p.get("type") or "contains").strip().lower(),
                    "capabilities": {str(k): bool(v) for k, v in caps.items()},
                }
            )
        merged: dict[str, Any] = {"exact": out_exact, "patterns": out_patterns}
        pg = raw.get("publisher_guidance")
        if isinstance(pg, dict):
            merged["publisher_guidance"] = pg
        return self._merge_with_seed_registry(merged)

    def _merge_with_seed_registry(self, current: dict[str, Any]) -> dict[str, Any]:
        """Non-destructive merge: keep user entries, add missing seed exact/pattern rules."""
        seed_path = _workspace_seed_registry_path()
        try:
            seed_raw = json.loads(seed_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return current
        if not isinstance(seed_raw, dict):
            return current

        seed_exact = seed_raw.get("exact") or {}
        if not isinstance(seed_exact, dict):
            seed_exact = {}
        seed_patterns = seed_raw.get("patterns") or []
        if not isinstance(seed_patterns, list):
            seed_patterns = []

        merged_exact = dict(current.get("exact") or {})
        for k, v in seed_exact.items():
            kk = str(k).strip().lower()
            if kk not in merged_exact and isinstance(v, dict):
                merged_exact[kk] = {str(x): bool(y) for x, y in v.items()}

        merged_patterns = list(current.get("patterns") or [])
        seen = {
            (
                str(p.get("match") or "").strip().lower(),
                str(p.get("type") or "contains").strip().lower(),
            )
            for p in merged_patterns
            if isinstance(p, dict)
        }
        for p in seed_patterns:
            if not isinstance(p, dict):
                continue
            key = (
                str(p.get("match") or "").strip().lower(),
                str(p.get("type") or "contains").strip().lower(),
            )
            caps = p.get("capabilities") or {}
            if key in seen or not isinstance(caps, dict):
                continue
            merged_patterns.append(
                {
                    "match": key[0],
                    "type": key[1],
                    "capabilities": {str(k): bool(v) for k, v in caps.items()},
                }
            )
            seen.add(key)

        seed_pg = seed_raw.get("publisher_guidance")
        merged_pg: dict[str, Any] = dict(current.get("publisher_guidance") or {})
        if isinstance(seed_pg, dict):
            for section in ("exact", "patterns"):
                seed_section = seed_pg.get(section) or {}
                cur_section = merged_pg.get(section) or {}
                if section == "exact" and isinstance(seed_section, dict):
                    merged_exact_pg = dict(cur_section) if isinstance(cur_section, dict) else {}
                    for k, v in seed_section.items():
                        kk = str(k).strip().lower()
                        if kk not in merged_exact_pg:
                            merged_exact_pg[kk] = v
                    merged_pg["exact"] = merged_exact_pg
                elif section == "patterns" and isinstance(seed_section, list):
                    merged_patterns_pg = list(cur_section) if isinstance(cur_section, list) else []
                    seen_pg = {
                        (
                            str(p.get("match") or "").strip().lower(),
                            str(p.get("type") or "contains").strip().lower(),
                        )
                        for p in merged_patterns_pg
                        if isinstance(p, dict)
                    }
                    for p in seed_section:
                        if not isinstance(p, dict):
                            continue
                        key = (
                            str(p.get("match") or "").strip().lower(),
                            str(p.get("type") or "contains").strip().lower(),
                        )
                        if key not in seen_pg:
                            merged_patterns_pg.append(p)
                            seen_pg.add(key)
                    merged_pg["patterns"] = merged_patterns_pg
            if merged_pg:
                current = dict(current)
                current["publisher_guidance"] = merged_pg

        result: dict[str, Any] = {"exact": merged_exact, "patterns": merged_patterns}
        if current.get("publisher_guidance"):
            result["publisher_guidance"] = current["publisher_guidance"]
        return result

    def append_missed_detection(self, payload: dict[str, Any]) -> None:
        try:
            raw = json.loads(self.missed_models_path.read_text(encoding="utf-8"))
            items = raw if isinstance(raw, list) else []
        except (FileNotFoundError, json.JSONDecodeError):
            items = []
        event = dict(payload or {})
        event["timestamp"] = datetime.now(timezone.utc).isoformat()
        items.append(event)
        self.missed_models_path.write_text(json.dumps(items, ensure_ascii=True, indent=2), encoding="utf-8")

    def load_learned_registry(self) -> dict[str, dict[str, bool]]:
        try:
            raw = json.loads(self.learned_registry_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        if not isinstance(raw, dict):
            return {}
        out: dict[str, dict[str, bool]] = {}
        for model_id, caps in raw.items():
            if isinstance(caps, dict):
                out[str(model_id).strip().lower()] = {str(k): bool(v) for k, v in caps.items()}
        return out

    def upsert_learned_capabilities(self, model_id: str, capabilities: dict[str, bool]) -> None:
        data = self.load_learned_registry()
        key = str(model_id or "").strip().lower()
        if not key:
            return
        existing = data.get(key) or {}
        merged = dict(existing)
        for k, v in capabilities.items():
            merged[str(k)] = bool(v)
        data[key] = merged
        self.learned_registry_path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")

    def export_capabilities_bundle(self) -> dict[str, Any]:
        return {
            "cached_capabilities": self.list_cached_capabilities(),
            "ground_truth": self._list_ground_truth_entries(),
            "learned_registry": self.load_learned_registry(),
        }

    def _list_ground_truth_entries(self) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT model_id, capabilities_json, source, created_at FROM ground_truth_capabilities ORDER BY created_at DESC"
            ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            try:
                caps = json.loads(r["capabilities_json"])
            except json.JSONDecodeError:
                continue
            out.append(
                {
                    "model_id": r["model_id"],
                    "capabilities": caps if isinstance(caps, dict) else {},
                    "source": r["source"],
                    "created_at": r["created_at"],
                }
            )
        return out

    def _ensure_registry_seeded(self) -> None:
        if self.registry_path.exists():
            return
        seed = _workspace_seed_registry_path()
        if seed.exists():
            self.registry_path.write_text(seed.read_text(encoding="utf-8"), encoding="utf-8")
            return
        self.registry_path.write_text("{}", encoding="utf-8")

    def _ensure_missed_models_seeded(self) -> None:
        if self.missed_models_path.exists():
            return
        seed = _workspace_missed_models_path()
        if seed.exists():
            self.missed_models_path.write_text(seed.read_text(encoding="utf-8"), encoding="utf-8")
            return
        self.missed_models_path.write_text("[]", encoding="utf-8")

    def _ensure_learned_registry_seeded(self) -> None:
        if self.learned_registry_path.exists():
            return
        seed = _workspace_learned_registry_path()
        if seed.exists():
            self.learned_registry_path.write_text(seed.read_text(encoding="utf-8"), encoding="utf-8")
            return
        self.learned_registry_path.write_text("{}", encoding="utf-8")

    def _load_publisher_guidance_file(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.publisher_guidance_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {"by_repo_id": {}, "by_model_key": {}}
        if not isinstance(raw, dict):
            return {"by_repo_id": {}, "by_model_key": {}}
        return {
            "by_repo_id": raw.get("by_repo_id") if isinstance(raw.get("by_repo_id"), dict) else {},
            "by_model_key": raw.get("by_model_key") if isinstance(raw.get("by_model_key"), dict) else {},
        }

    def _save_publisher_guidance_file(self, data: dict[str, Any]) -> None:
        self.publisher_guidance_path.write_text(
            json.dumps(data, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def upsert_publisher_guidance(self, repo_id: str, guidance: dict[str, Any]) -> None:
        key = str(repo_id or "").strip()
        if not key:
            return
        data = self._load_publisher_guidance_file()
        by_repo = dict(data.get("by_repo_id") or {})
        by_repo[key] = dict(guidance)
        data["by_repo_id"] = by_repo
        self._save_publisher_guidance_file(data)

    def get_publisher_guidance(self, repo_id: str) -> dict[str, Any] | None:
        key = str(repo_id or "").strip()
        if not key:
            return None
        data = self._load_publisher_guidance_file()
        raw = (data.get("by_repo_id") or {}).get(key)
        return dict(raw) if isinstance(raw, dict) else None

    def _load_provenance_file(self) -> dict[str, str]:
        try:
            raw = json.loads(self.model_hf_provenance_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        if not isinstance(raw, dict):
            return {}
        return {str(k): str(v) for k, v in raw.items() if k and v}

    def _save_provenance_file(self, data: dict[str, str]) -> None:
        self.model_hf_provenance_path.write_text(
            json.dumps(data, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def set_model_hf_provenance(self, local_path: str, repo_id: str) -> None:
        path = str(local_path or "").strip()
        repo = str(repo_id or "").strip()
        if not path or not repo:
            return
        data = self._load_provenance_file()
        data[path] = repo
        self._save_provenance_file(data)

    def get_model_hf_provenance(self, local_path: str) -> str | None:
        path = str(local_path or "").strip()
        if not path:
            return None
        return self._load_provenance_file().get(path)

    def load_model_hf_provenance_map(self) -> dict[str, str]:
        return dict(self._load_provenance_file())

    def remove_model_hf_provenance(self, local_path: str) -> None:
        path = str(local_path or "").strip()
        if not path:
            return
        data = self._load_provenance_file()
        if path in data:
            del data[path]
            self._save_provenance_file(data)
            return
        basename = os.path.basename(path)
        if not basename:
            return
        changed = False
        for stored_path in list(data):
            if os.path.basename(stored_path) == basename:
                del data[stored_path]
                changed = True
        if changed:
            self._save_provenance_file(data)

    def _ensure_publisher_guidance_seeded(self) -> None:
        if self.publisher_guidance_path.exists():
            return
        self.publisher_guidance_path.write_text(
            json.dumps({"by_repo_id": {}, "by_model_key": {}}, indent=2),
            encoding="utf-8",
        )

    def _ensure_model_hf_provenance_seeded(self) -> None:
        if self.model_hf_provenance_path.exists():
            return
        self.model_hf_provenance_path.write_text("{}", encoding="utf-8")
