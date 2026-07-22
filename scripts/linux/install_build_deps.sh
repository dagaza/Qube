#!/usr/bin/env bash
#
# Install Ubuntu packages needed to build and smoke-test Linux packages.
#
# Usage:   scripts/linux/install_build_deps.sh
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  python3-dev \
  ruby \
  ruby-dev \
  rpm \
  xorriso \
  zsync \
  xvfb \
  libportaudio2 \
  libgl1 \
  libglib2.0-0 \
  libdbus-1-3 \
  libxcb1 \
  libxkbcommon0 \
  libx11-6 \
  libfontconfig1 \
  libgomp1 \
  file

if ! command -v fpm >/dev/null 2>&1; then
  sudo gem install fpm --no-document
fi
