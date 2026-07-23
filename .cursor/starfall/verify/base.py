"""Initiative-agnostic verification checks for Starfall.

`BaseVerifier` bundles the checks that apply to *any* Starfall initiative:
tests actually pass, delivered files exist, the work log is well-formed, the
evidence map points at real code, and git is sane. Initiative-specific rules
(e.g. the MCP P6 guardrail) live in sibling plugins that subclass `BaseVerifier`
and extend :meth:`checks` — see ``mcp.py``. The orchestrator
(``.cursor/hooks/starfall_verify.py``) picks the plugin named in
``active-task.md`` (``Verifier:`` field), defaulting to this base.

Every check is deterministic, executable, LLM-independent, and **fails safe**:
missing evidence behind a completion claim returns BLOCKED, never a silent pass.
"""
from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# This file: .cursor/starfall/verify/base.py
VERIFY_DIR = Path(__file__).resolve().parent
CURSOR_DIR = VERIFY_DIR.parents[1]
REPO = CURSOR_DIR.parent
MEM = CURSOR_DIR / "starfall"
HANDOFF = MEM / "handoff.md"
TEST_PLAN = MEM / "test-plan.md"
EVIDENCE = MEM / "evidence-map.md"
LOG = CURSOR_DIR / "starfall-log.md"

PYTEST_TIMEOUT_S = 240

_REPO_DIRS = ("core", "tests", "ui", "docs", "mcp", "workers", "scripts", ".cursor")
_PATH_RE = re.compile(
    r"\b((?:" + "|".join(map(re.escape, _REPO_DIRS)) + r")/[\w./-]+\.\w{1,4})\b"
)
_PATH_SYMBOL_RE = re.compile(
    r"\b((?:" + "|".join(map(re.escape, _REPO_DIRS)) + r")/[\w./-]+\.\w{1,4}):([A-Za-z_]\w*)"
)
_TEST_TOKEN_RE = re.compile(r"\btests/[\w./*-]+\.py(?:::[\w:]+)?")
_COMMIT_RE = re.compile(r"(?i)commit(?:ted)?\s+(?:as\s+)?`?([0-9a-f]{7,40})`?")

# Sections every coordinator work entry must contain (loose, case-insensitive).
_WORKLOG_SECTIONS = ("Phase:", "Gates:", "Architecture Review", "Decisions", "Next step")


@dataclass
class Check:
    name: str
    status: str  # "PASS" | "BLOCKED" | "N/A"
    detail: str


def read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8") if path.exists() else ""
    except Exception as exc:  # pragma: no cover - defensive
        return f"<<read error: {exc}>>"


def status_marker(text: str) -> str:
    for line in text.splitlines():
        m = re.match(r"\s*STATUS:\s*([A-Za-z]+)\s*$", line)
        if m:
            return m.group(1).upper()
    return ""


def git(*args: str, timeout: int = 10) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(REPO), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


class BaseVerifier:
    """Initiative-agnostic checks. Subclass and extend :meth:`checks` for plugins."""

    name = "base"

    def checks(self) -> list:
        return [
            self.check_tests,
            self.check_files,
            self.check_worklog,
            self.check_evidence_map,
            self.check_git,
        ]

    # -- tests ------------------------------------------------------------
    def check_tests(self) -> Check:
        """If Gate 4 is claimed (test-plan COMPLETE), run the referenced tests."""
        text = read(TEST_PLAN)
        if not text:
            return Check("tests", "N/A", "no test-plan.md found")
        if status_marker(text) != "COMPLETE":
            return Check("tests", "N/A", "test-plan.md not marked STATUS: COMPLETE")

        tokens = set(_TEST_TOKEN_RE.findall(text))
        if not tokens:
            return Check("tests", "BLOCKED", "test-plan COMPLETE but no `tests/*.py` references found")

        args: list[str] = []
        missing: list[str] = []
        for tok in sorted(tokens):
            if "*" in tok:
                matches = sorted(str(p.relative_to(REPO)) for p in REPO.glob(tok))
                if not matches:
                    missing.append(tok)
                args.extend(matches)
            else:
                file_part = tok.split("::", 1)[0]
                if not (REPO / file_part).exists():
                    missing.append(tok)
                else:
                    args.append(tok)
        if missing:
            return Check("tests", "BLOCKED", f"referenced test files not found: {', '.join(missing)}")

        whole_files = {a for a in args if "::" not in a}
        targets = sorted(whole_files | {a for a in args if a.split("::", 1)[0] not in whole_files})
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", *targets, "-q", "-p", "no:cacheprovider"],
                capture_output=True,
                text=True,
                cwd=str(REPO),
                timeout=PYTEST_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return Check("tests", "BLOCKED", f"pytest exceeded {PYTEST_TIMEOUT_S}s")
        except Exception as exc:
            return Check("tests", "BLOCKED", f"could not run pytest: {exc}")

        summary = (proc.stdout.strip().splitlines() or ["<no output>"])[-1]
        if proc.returncode != 0:
            tail = (proc.stdout + proc.stderr).strip()[-600:]
            return Check("tests", "BLOCKED", f"pytest failed (rc={proc.returncode}): {tail}")
        return Check("tests", "PASS", f"{len(targets)} target(s): {summary}")

    # -- delivered files --------------------------------------------------
    def check_files(self) -> Check:
        """If handoff is READY, every file it lists as delivered must exist."""
        text = read(HANDOFF)
        if not text:
            return Check("files", "N/A", "no handoff.md found")
        if status_marker(text) != "READY":
            return Check("files", "N/A", "handoff.md not marked STATUS: READY")

        lower = text.lower()
        # Anchor on a Delivered section header, not incidental "delivered" in prose.
        start = lower.find("delivered (")
        if start == -1:
            start = lower.find("delivered:")
        if start == -1:
            start = lower.find("deliver")
        region = text[start:] if start != -1 else text
        for cutoff in ("next slice", "not in this run", "not on disk", "follow-up", "future", "next slices"):
            idx = region.lower().find(cutoff)
            if idx != -1:
                region = region[:idx]

        candidates = sorted(set(_PATH_RE.findall(region)))
        if not candidates:
            return Check("files", "N/A", "handoff READY but lists no parseable delivered paths")
        missing = [c for c in candidates if not (REPO / c).exists()]
        if missing:
            return Check("files", "BLOCKED", f"delivered files missing: {', '.join(missing)}")
        return Check("files", "PASS", f"{len(candidates)} delivered file(s) exist")

    # -- work log structure ----------------------------------------------
    def check_worklog(self) -> Check:
        """The latest coordinator work entry must carry all required sections."""
        text = read(LOG)
        if not text:
            return Check("worklog", "N/A", "no starfall-log.md found")
        # Coordinator entries start "## <expert> - <ts>"; skip "## Hook Turn N".
        entries = re.split(r"(?m)^## (?!Hook Turn)", text)
        coord = [e for e in entries[1:] if e.strip()]
        if not coord:
            return Check("worklog", "N/A", "no coordinator work entries yet")
        last = coord[-1]
        missing = [s for s in _WORKLOG_SECTIONS if s.lower() not in last.lower()]
        if missing:
            header = last.splitlines()[0].strip()
            return Check("worklog", "BLOCKED", f"latest entry '{header}' missing: {', '.join(missing)}")
        return Check("worklog", "PASS", "latest work entry well-formed")

    # -- evidence map -----------------------------------------------------
    def check_evidence_map(self) -> Check:
        """Every file/symbol the evidence map cites must exist (anti-hallucination)."""
        text = read(EVIDENCE)
        if not text:
            return Check("evidence", "N/A", "no evidence-map.md found")

        problems: list[str] = []
        checked = 0
        # path:symbol references — file must exist AND contain the symbol.
        sym_pairs = set(_PATH_SYMBOL_RE.findall(text))
        symbol_paths = {p for p, _ in sym_pairs}
        for path, symbol in sorted(sym_pairs):
            checked += 1
            fp = REPO / path
            if not fp.exists():
                problems.append(f"{path} (missing file)")
                continue
            if not re.search(rf"\b{re.escape(symbol)}\b", read(fp)):
                problems.append(f"{path}:{symbol} (symbol not found)")
        # bare path references — file must exist.
        for path in sorted(set(_PATH_RE.findall(text))):
            if path in symbol_paths:
                continue
            checked += 1
            if not (REPO / path).exists():
                problems.append(f"{path} (missing file)")
        if checked == 0:
            return Check("evidence", "N/A", "no parseable code references in evidence map")
        if problems:
            return Check("evidence", "BLOCKED", f"stale references: {', '.join(problems)}")
        return Check("evidence", "PASS", f"{checked} code reference(s) resolve")

    # -- git --------------------------------------------------------------
    def check_git(self) -> Check:
        """Not on a protected branch; any commit hash cited in the log must resolve."""
        try:
            branch = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        except Exception as exc:
            return Check("git", "BLOCKED", f"could not read branch: {exc}")
        if branch in ("main", "master"):
            return Check("git", "BLOCKED", f"on protected branch '{branch}'")

        log_text = read(LOG)
        hashes = sorted(set(_COMMIT_RE.findall(log_text)))
        unresolved: list[str] = []
        off_branch: list[str] = []
        for h in hashes:
            try:
                if git("rev-parse", "--verify", "--quiet", f"{h}^{{commit}}").returncode != 0:
                    unresolved.append(h)
                    continue
                if branch not in git("branch", "--contains", h).stdout:
                    off_branch.append(h)
            except Exception:
                unresolved.append(h)
        if unresolved:
            return Check("git", "BLOCKED", f"cited commit(s) do not resolve: {', '.join(unresolved)}")
        if off_branch:
            return Check("git", "BLOCKED", f"cited commit(s) not on '{branch}': {', '.join(off_branch)}")
        cited = f"; verified {len(hashes)} cited commit(s)" if hashes else ""
        return Check("git", "PASS", f"branch '{branch}' (not protected){cited}")
