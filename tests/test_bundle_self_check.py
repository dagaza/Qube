"""Plan S15: the release self-check for modules and data that fail silently when missing."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from core import bundle_self_check as bsc

_REPO = Path(__file__).resolve().parent.parent


def test_checks_pass_in_dev_environment() -> None:
    result = bsc.run_bundle_self_check()
    assert result["ok"] is True, result
    assert set(result["checks"]) == {"reranker_module", "spelling_en"}
    assert result["checks"]["spelling_en"]["corrected"] == "Intelligence"
    assert result["frozen"] is False


def test_failed_check_is_reported_not_raised() -> None:
    def boom() -> dict:
        raise ModuleNotFoundError("No module named 'fastembed.rerank'")

    result = bsc.run_bundle_self_check([("reranker_module", boom), ("other", lambda: {"ok": True})])
    assert result["ok"] is False
    assert result["checks"]["reranker_module"] == {
        "ok": False,
        "error": "ModuleNotFoundError: No module named 'fastembed.rerank'",
    }
    assert result["checks"]["other"] == {"ok": True}


def test_missing_word_list_fails_the_spelling_check() -> None:
    # What a bundle without resources/en.json.gz looks like to the check (1.3.60).
    with patch("core.knowledge.spelling.spelling_available", return_value=False):
        out = bsc._check_spelling()
    assert out["ok"] is False
    assert "unavailable" in out["error"]


def test_wrong_correction_fails_the_spelling_check() -> None:
    with patch("core.knowledge.spelling.correct_word", return_value=None):
        out = bsc._check_spelling()
    assert out == {"ok": False, "corrected": None}


@pytest.mark.parametrize(
    "argv, requested",
    [
        (["Qube"], False),
        (["Qube", "--self-check-bundle"], True),
        (["Qube", "--self-check-bundle", "/tmp/r.json"], True),
        (["Qube", "--self-check-bundle=/tmp/r.json"], True),
        (["Qube", "--self-check-bundle-extra"], False),
    ],
)
def test_flag_detection(argv: list[str], requested: bool) -> None:
    assert bsc.self_check_requested(argv) is requested


def test_main_writes_result_and_exit_code(tmp_path: Path) -> None:
    ok_path = tmp_path / "ok" / "result.json"
    assert bsc.main(["Qube", "--self-check-bundle", str(ok_path)], [("a", lambda: {"ok": True})]) == 0
    assert json.loads(ok_path.read_text(encoding="utf-8"))["ok"] is True

    bad_path = tmp_path / "bad.json"
    assert bsc.main([f"--self-check-bundle={bad_path}"], [("a", lambda: {"ok": False})]) == 1
    assert json.loads(bad_path.read_text(encoding="utf-8"))["checks"]["a"] == {"ok": False}


def test_main_without_path_or_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    assert bsc.main(["Qube", "--self-check-bundle"], [("a", lambda: {"ok": True})]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    # Windowed frozen builds (console=False) have sys.stdout = None.
    with patch.object(sys, "stdout", None):
        assert bsc.main(["--self-check-bundle"], [("a", lambda: {"ok": True})]) == 0


def test_entry_point_dispatches_before_qt() -> None:
    import qube_entry

    with (
        patch.object(sys, "argv", ["Qube", "--self-check-bundle"]),
        patch("core.bundle_self_check.main", return_value=7) as fake_main,
        patch("core.winget_validation.record_validation_entry") as validation_entry,
    ):
        assert qube_entry.run() == 7
    fake_main.assert_called_once()
    validation_entry.assert_not_called()


def test_spec_bundles_spelling_word_list_and_reranker_module() -> None:
    spec = (_REPO / "qube.spec").read_text(encoding="utf-8")
    assert 'collect_data_files("spellchecker", includes=["resources/en.json.gz"])' in spec
    assert '"fastembed.rerank.cross_encoder"' in spec


@pytest.mark.parametrize(
    "script",
    ["scripts/release/smoke_linux_dist.sh", "scripts/release/smoke_dist.ps1", "scripts/release/smoke_dmg.sh"],
)
def test_release_smoke_scripts_run_the_self_check(script: str) -> None:
    assert bsc.SELF_CHECK_FLAG in (_REPO / script).read_text(encoding="utf-8")
