#!/usr/bin/env bash
# CLI entry point installed to /usr/bin/qube-uninstall by the .deb package.
set -euo pipefail
exec /opt/qube/uninstall/uninstall.sh "$@"
