#!/usr/bin/env bash
#
# Recompress a .deb data tarball with maximum xz settings.
# Used when fpm's default xz level still leaves CUDA packages near GitHub's 2 GiB cap.
#
# Usage:   scripts/linux/recompress_deb_data.sh <package.deb>
set -euo pipefail

DEB="${1:?Usage: recompress_deb_data.sh <package.deb>}"
if [[ ! -f "$DEB" ]]; then
  echo "Package not found: $DEB" >&2
  exit 1
fi

WORK="$(mktemp -d)"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

cd "$WORK"
ar x "$(readlink -f "$DEB")"

CONTROL_TAR=""
DATA_TAR=""
for candidate in control.tar.xz control.tar.gz control.tar.zst control.tar; do
  if [[ -f "$candidate" ]]; then
    CONTROL_TAR="$candidate"
    break
  fi
done
for candidate in data.tar.xz data.tar.gz data.tar.zst data.tar; do
  if [[ -f "$candidate" ]]; then
    DATA_TAR="$candidate"
    break
  fi
done

if [[ -z "$CONTROL_TAR" || -z "$DATA_TAR" || ! -f debian-binary ]]; then
  echo "Unexpected .deb layout in $DEB" >&2
  ls -la >&2
  exit 1
fi

DATA_ROOT="$WORK/data-root"
mkdir -p "$DATA_ROOT"
tar -xf "$DATA_TAR" -C "$DATA_ROOT"
rm -f "$DATA_TAR"

NEW_DATA="data.tar.xz"
export XZ_OPT="-9e"
tar -cJf "$NEW_DATA" -C "$DATA_ROOT" .
rm -rf "$DATA_ROOT"

OUTPUT="$(mktemp --suffix=.deb)"
ar r "$OUTPUT" debian-binary "$CONTROL_TAR" "$NEW_DATA"
mv -f "$OUTPUT" "$DEB"
echo "Recompressed $DEB with xz -9e"
