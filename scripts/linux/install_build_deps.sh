#!/usr/bin/env bash
#
# Install Ubuntu packages needed to build and smoke-test Linux packages.
#
# Usage:   scripts/linux/install_build_deps.sh [cpu|vulkan|cuda|all]
set -euo pipefail

VARIANT="${1:-all}"

export DEBIAN_FRONTEND=noninteractive
PACKAGES=(
  build-essential
  python3-dev
  ruby
  ruby-dev
  rpm
  xorriso
  zsync
  xvfb
  libportaudio2
  portaudio19-dev
  libegl1
  libgl1
  libglib2.0-0
  libdbus-1-3
  libxcb1
  libxkbcommon0
  libx11-6
  libfontconfig1
  libgomp1
  file
)

if [[ "$VARIANT" == "vulkan" || "$VARIANT" == "all" ]]; then
  PACKAGES+=(
    spirv-headers
    glslang-dev
    glslang-tools
    shaderc
    libvulkan-dev
    libvulkan1
  )
fi

sudo apt-get update
sudo apt-get install -y "${PACKAGES[@]}"

if ! command -v fpm >/dev/null 2>&1; then
  sudo gem install fpm --no-document
fi
