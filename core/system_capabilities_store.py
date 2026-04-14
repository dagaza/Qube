from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
    return Path(__file__).resolve().parent.parent / "system_data" / "curated_registry.json"


def _workspace_missed_models_path() -> Path:
    return Path(__file__).resolve().parent.parent / "system_data" / "missed_models.json"


def _workspace_learned_registry_path() -> Path:
    return Path(__file__).resolve().parent.parent / "system_data" / "learned_capabilities.json"


class SystemCapabilitiesStore:
    def __init__(self, system_data_dir: str | Path | None = None):
        self.system_data_dir = Path(system_data_dir) if system_data_dir else default_system_data_dir()
        self.system_data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.system_data_dir / "capabilities.db"
        self.registry_path = self.system_data_dir / "curated_registry.json"
        self.missed_models_path = self.system_data_dir / "missed_models.json"
        self.learned_registry_path = self.system_data_dir / "learned_capabilities.json"
        self._ensure_registry_seeded()
        self._ensure_missed_models_seeded()
        self._ensure_learned_registry_seeded()
        self.init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

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
        merged = {"exact": out_exact, "patterns": out_patterns}
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

        return {"exact": merged_exact, "patterns": merged_patterns}

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
