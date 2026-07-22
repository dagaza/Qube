#!/bin/bash
# Launcher for Uninstall Qube.app — delegates to Resources/uninstall.sh.
set -euo pipefail

RESOURCES="$(cd "$(dirname "$0")/../Resources" && pwd)"
exec "${RESOURCES}/uninstall.sh" "$@"
