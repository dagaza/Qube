"""Tests for local state backup export and restore."""

from __future__ import annotations

import json
import sqlite3
import zipfile
from pathlib import Path

import pytest

from core.state_backup.export import export_state_backup
from core.state_backup.import_backup import restore_state_backup
from core.state_backup.scheduler import is_auto_backup_due, prune_auto_backups, run_auto_backup_if_due
from core.state_backup.manifest import BACKUP_MANIFEST_NAME, verify_backup_archive


def _init_minimal_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            "INSERT INTO sessions (id, title) VALUES ('s1', 'Test chat')"
        )
        conn.commit()
    finally:
        conn.close()


def _seed_user_data(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _init_minimal_db(root / "qube_data.db")
    (root / "settings.json").write_text(
        json.dumps({"qube.ui.theme.mode": "dark"}),
        encoding="utf-8",
    )
    lance_dir = root / "data" / "lancedb"
    lance_dir.mkdir(parents=True)
    (lance_dir / "marker.txt").write_text("lance-data", encoding="utf-8")
    models_dir = root / "models"
    models_dir.mkdir(parents=True)
    (models_dir / "heavy.gguf").write_bytes(b"\x00" * 1024)


@pytest.fixture
def user_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import core.settings_store as settings_store_module
    from core.settings_store import SettingsStore, reset_settings_store_for_tests

    reset_settings_store_for_tests()
    settings_store_module._store = SettingsStore(user_path=tmp_path / "settings.json")

    root = tmp_path / ".qube"
    _seed_user_data(root)
    monkeypatch.setattr("core.state_backup.paths.user_data_root", lambda: root)
    monkeypatch.setattr("core.paths.user_data_root", lambda: root)
    yield root
    reset_settings_store_for_tests()


def test_export_and_restore_round_trip(user_root: Path, tmp_path: Path) -> None:
    backup_path = tmp_path / "state.qube-backup.zip"
    export_result = export_state_backup(backup_path, user_data_root=user_root)
    assert export_result.ok, export_result.error
    assert export_result.file_count >= 3

    verify = verify_backup_archive(backup_path)
    assert verify.ok, verify.error

    (user_root / "settings.json").write_text('{"qube.ui.theme.mode": "light"}', encoding="utf-8")
    conn = sqlite3.connect(user_root / "qube_data.db")
    try:
        conn.execute("DELETE FROM sessions")
        conn.commit()
    finally:
        conn.close()

    restore_result = restore_state_backup(
        backup_path,
        user_data_root=user_root,
        create_pre_restore_snapshot=False,
    )
    assert restore_result.ok, restore_result.error
    assert restore_result.files_restored >= 3

    restored_settings = json.loads((user_root / "settings.json").read_text(encoding="utf-8"))
    assert restored_settings["qube.ui.theme.mode"] == "dark"

    conn = sqlite3.connect(user_root / "qube_data.db")
    try:
        row = conn.execute("SELECT title FROM sessions WHERE id = 's1'").fetchone()
    finally:
        conn.close()
    assert row is not None
    assert row[0] == "Test chat"


def test_models_directory_is_not_included(user_root: Path, tmp_path: Path) -> None:
    backup_path = tmp_path / "state.qube-backup.zip"
    export_result = export_state_backup(backup_path, user_data_root=user_root)
    assert export_result.ok, export_result.error

    with zipfile.ZipFile(backup_path, mode="r") as archive:
        names = archive.namelist()
    assert not any(name.startswith("models/") for name in names)
    assert "models/heavy.gguf" not in names


def test_corrupt_manifest_is_rejected(user_root: Path, tmp_path: Path) -> None:
    backup_path = tmp_path / "state.qube-backup.zip"
    export_result = export_state_backup(backup_path, user_data_root=user_root)
    assert export_result.ok, export_result.error

    with zipfile.ZipFile(backup_path, mode="r") as archive:
        manifest = json.loads(archive.read(BACKUP_MANIFEST_NAME))
        manifest["files"][0]["sha256"] = "0" * 64
        tampered = tmp_path / "tampered.qube-backup.zip"
        with zipfile.ZipFile(tampered, mode="w", compression=zipfile.ZIP_DEFLATED) as out:
            out.writestr(
                BACKUP_MANIFEST_NAME,
                json.dumps(manifest, indent=2) + "\n",
            )
            for name in archive.namelist():
                if name == BACKUP_MANIFEST_NAME:
                    continue
                out.writestr(name, archive.read(name))

    verify = verify_backup_archive(tampered)
    assert not verify.ok
    assert verify.error is not None
    assert "Checksum mismatch" in verify.error


def test_pre_restore_snapshot_created(user_root: Path, tmp_path: Path) -> None:
    backup_path = tmp_path / "state.qube-backup.zip"
    export_result = export_state_backup(backup_path, user_data_root=user_root)
    assert export_result.ok, export_result.error

    (user_root / "settings.json").write_text('{"changed": true}', encoding="utf-8")
    pre_dir = user_root / "backups"
    restore_result = restore_state_backup(
        backup_path,
        user_data_root=user_root,
        create_pre_restore_snapshot=True,
        pre_restore_dir=pre_dir,
    )
    assert restore_result.ok, restore_result.error
    assert restore_result.pre_restore_backup is not None
    assert restore_result.pre_restore_backup.is_file()
    verify = verify_backup_archive(restore_result.pre_restore_backup)
    assert verify.ok, verify.error


def test_auto_backup_skipped_when_disabled(user_root: Path, tmp_path: Path) -> None:
    from core import app_settings

    app_settings.set_backup_auto_enabled(False)
    result = run_auto_backup_if_due(root=user_root)
    assert result.ok
    assert result.ran is False


def test_auto_backup_runs_and_prunes_old_archives(
    user_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core import app_settings
    from core.state_backup.paths import auto_backups_dir

    app_settings.set_backup_auto_enabled(True)
    app_settings.set_backup_interval_days(30)
    app_settings.set_backup_retention_count(2)
    app_settings.set_backup_last_run_at("")

    first = run_auto_backup_if_due(root=user_root)
    assert first.ok and first.ran, first.error

    auto_dir = auto_backups_dir(user_root)
    for index in range(3):
        path = auto_dir / f"older-{index}.qube-backup.zip"
        path.write_bytes(b"placeholder")

    app_settings.set_backup_last_run_at("2000-01-01T00:00:00+00:00")
    second = run_auto_backup_if_due(root=user_root)
    assert second.ok and second.ran, second.error
    assert second.pruned_count >= 2

    remaining = list(auto_dir.glob("*.qube-backup.zip"))
    assert len(remaining) <= 2


def test_is_auto_backup_due_respects_interval(user_root: Path) -> None:
    from core import app_settings
    from datetime import datetime, timezone

    app_settings.set_backup_auto_enabled(True)
    app_settings.set_backup_interval_days(30)
    app_settings.set_backup_last_run_at(datetime.now(timezone.utc).isoformat())
    assert is_auto_backup_due() is False

    app_settings.set_backup_last_run_at("2000-01-01T00:00:00+00:00")
    assert is_auto_backup_due(now=datetime(2000, 2, 1, tzinfo=timezone.utc)) is True


def test_storage_summary_prefers_latest_archive(user_root: Path, tmp_path: Path) -> None:
    from core.state_backup.storage_summary import (
        estimate_backup_uncompressed_bytes,
        find_latest_backup_archive,
        format_storage_summary_text,
    )

    backup_path = tmp_path / "state.qube-backup.zip"
    export_result = export_state_backup(backup_path, user_data_root=user_root)
    assert export_result.ok, export_result.error

    auto_dir = user_root / "backups" / "auto"
    auto_dir.mkdir(parents=True, exist_ok=True)
    newer_path = auto_dir / "auto-state.qube-backup.zip"
    newer_path.write_bytes(b"\x00" * 2048)

    latest = find_latest_backup_archive(root=user_root)
    assert latest is not None
    assert latest.path == newer_path
    assert latest.size_bytes == 2048

    estimated = estimate_backup_uncompressed_bytes(root=user_root)
    assert estimated >= 1024

    summary = format_storage_summary_text(root=user_root)
    assert "Last backup 2.0 KB" in summary
    assert "models not included" in summary
