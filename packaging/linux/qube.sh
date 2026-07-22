#!/usr/bin/env bash
# Wrapper installed to /usr/bin/qube by the .deb package.
set -euo pipefail
exec /opt/qube/Qube "$@"
