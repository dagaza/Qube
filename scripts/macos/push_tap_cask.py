#!/usr/bin/env python3
"""Push a rendered Homebrew cask to dagaza/homebrew-qube via the Contents API."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

_DEFAULT_REPO = "dagaza/homebrew-qube"
_CASK_PATH = "Casks/qube.rb"


def _api(
    method: str,
    url: str,
    token: str,
    payload: dict | None = None,
) -> dict:
    data = None
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "qube-homebrew-tap-bump",
    }
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body.strip() else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"GitHub API {method} {url} failed ({exc.code}): {detail}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="Release version (for commit message)")
    parser.add_argument(
        "cask_file",
        type=Path,
        help="Path to rendered qube.rb",
    )
    parser.add_argument(
        "--repo",
        default=_DEFAULT_REPO,
        help=f"owner/repo tap (default: {_DEFAULT_REPO})",
    )
    args = parser.parse_args(argv)

    token = os.environ.get("HOMEBREW_TAP_TOKEN", "").strip()
    if not token:
        raise SystemExit("HOMEBREW_TAP_TOKEN is required")

    cask_path = args.cask_file.resolve()
    if not cask_path.is_file():
        raise SystemExit(f"Cask file not found: {cask_path}")

    content_b64 = base64.b64encode(cask_path.read_bytes()).decode("ascii")
    api_url = f"https://api.github.com/repos/{args.repo}/contents/{_CASK_PATH}"

    existing_sha: str | None = None
    try:
        meta = _api("GET", api_url, token)
        existing_sha = str(meta.get("sha") or "") or None
    except SystemExit as exc:
        if "404" not in str(exc):
            raise
        existing_sha = None

    payload: dict = {
        "message": f"qube {args.version}",
        "content": content_b64,
        "committer": {
            "name": "github-actions[bot]",
            "email": "github-actions[bot]@users.noreply.github.com",
        },
    }
    if existing_sha:
        payload["sha"] = existing_sha

    result = _api("PUT", api_url, token, payload)
    commit = result.get("commit", {})
    sha = commit.get("sha", "?")
    print(f"Pushed qube {args.version} to {args.repo} ({sha[:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())