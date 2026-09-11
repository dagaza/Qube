#!/usr/bin/env bash
# Merge microsoft/winget-pkgs master into the PAT owner's fork before wingetcreate submit.
set -euo pipefail

TOKEN="${1:?usage: sync_winget_fork.sh <github_pat> [upstream_branch]}"
UPSTREAM_BRANCH="${2:-master}"

API="https://api.github.com"
AUTH=(-H "Authorization: Bearer ${TOKEN}" -H "Accept: application/vnd.github+json")

USER="$(
  curl -sf "${AUTH[@]}" "${API}/user" |
    python -c "import sys, json; print(json.load(sys.stdin)['login'])"
)"
FORK="${USER}/winget-pkgs"
echo "Syncing ${FORK} with microsoft/winget-pkgs (${UPSTREAM_BRANCH})..."

RESPONSE_FILE="$(mktemp)"
trap 'rm -f "${RESPONSE_FILE}"' EXIT
HTTP="$(
  curl -sS "${AUTH[@]}" -o "${RESPONSE_FILE}" -w "%{http_code}" -X POST \
    "${API}/repos/${FORK}/merge-upstream" \
    -d "{\"branch\":\"${UPSTREAM_BRANCH}\"}"
)"

BODY="$(cat "${RESPONSE_FILE}")"
echo "${BODY}"

if [[ "${HTTP}" == "200" ]]; then
  echo "Fork sync succeeded."
  exit 0
fi

echo "ERROR: could not sync ${FORK} (HTTP ${HTTP})." >&2
if echo "${BODY}" | grep -q '"workflow" scope'; then
  echo "The WINGET_SUBMIT_TOKEN PAT needs the 'workflow' scope to merge upstream" >&2
  echo "workflow file changes, OR sync the fork manually on GitHub:" >&2
  echo "  https://github.com/${FORK}/compare/master...microsoft:winget-pkgs:master" >&2
else
  echo "Open https://github.com/${FORK}/fork and sync with upstream, then retry." >&2
fi
exit 1
