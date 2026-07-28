#!/usr/bin/env bash
# Submit WinGet manifest update PRs for all Qube Windows installer variants.
set -euo pipefail

VERSION="${1:?usage: submit_winget_packages.sh <version> <token> [wingetcreate.exe]}"
VERSION="${VERSION#v}"
TOKEN="${2:?usage: submit_winget_packages.sh <version> <token> [wingetcreate.exe]}"
WINGETCREATE="${3:-./wingetcreate.exe}"
REPO="${WINGET_REPO:-dagaza/Qube}"

declare -A URLS=(
  ["dagaza.Qube"]="https://github.com/${REPO}/releases/download/v${VERSION}/Qube-${VERSION}-Setup.exe"
  ["dagaza.Qube.Vulkan"]="https://github.com/${REPO}/releases/download/v${VERSION}/Qube-${VERSION}-vulkan-Setup.exe"
  ["dagaza.Qube.CUDA"]="https://github.com/${REPO}/releases/download/v${VERSION}/Qube-${VERSION}-cuda-Setup.exe"
)

for package_id in dagaza.Qube dagaza.Qube.Vulkan dagaza.Qube.CUDA; do
  echo "Submitting WinGet update for ${package_id} (${VERSION})..."
  "${WINGETCREATE}" update "${package_id}" \
    --version "${VERSION}" \
    --urls "${URLS[${package_id}]}" \
    --token "${TOKEN}" \
    --submit
done
