#!/usr/bin/env bash
# GGUF decode smoke for Linux dist builds (q8 matmul / AVX-VNNI regression).
#
# Usage: scripts/release/smoke_linux_llama_inference.sh [dist/Qube]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DIST="${1:-$REPO_ROOT/dist/Qube}"

exec python3 "$SCRIPT_DIR/smoke_linux_llama_inference.py" --dist-dir "$DIST" --n-gpu-layers 0