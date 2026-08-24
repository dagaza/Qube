"""Tests for scripts/render_homebrew_cask.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_render():
    path = Path(__file__).resolve().parents[1] / "scripts" / "render_homebrew_cask.py"
    spec = importlib.util.spec_from_file_location("render_homebrew_cask", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_render_writes_cask(tmp_path, monkeypatch):
    mod = _load_render()
    templates = tmp_path / "homebrew" / "templates"
    templates.mkdir(parents=True)
    (templates / "qube.rb.tmpl").write_text(
        'version "{{VERSION}}"\n'
        'arm "{{SHA256_ARM64}}"\n'
        'intel "{{SHA256_X86_64}}"\n'
        'url ".../v#{version}/Qube-#{version}-arm64.dmg"\n'
        "zap trash: [\n{{ZAP_TRASH_LINES}}\n]\n"
        "caveats <<~EOS\nGatekeeper\nEOS\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "_repo_root", lambda: tmp_path)

    out = mod.render("1.2.3", "AB" * 32, "CD" * 32)
    assert out.is_dir()
    cask = (out / "qube.rb").read_text(encoding="utf-8")

    assert 'version "1.2.3"' in cask
    # Digests are lowercased for Homebrew.
    assert "ab" * 32 in cask
    assert "cd" * 32 in cask
    # Ruby interpolation is left intact for Homebrew to expand at install time.
    assert "Qube-#{version}-arm64.dmg" in cask
    assert '"~/.qube",' in cask
    assert "Gatekeeper" in cask


def test_render_zap_paths_are_sorted_with_trailing_commas(monkeypatch):
    mod = _load_render()
    lines = mod._zap_trash_lines().splitlines()
    assert lines == sorted(lines)
    assert all(line.rstrip().endswith('",') for line in lines)
