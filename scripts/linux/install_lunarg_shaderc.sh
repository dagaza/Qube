#!/usr/bin/env bash
#
# Install glslc on Ubuntu 22.04 via LunarG's apt repo.
# Jammy main/universe do not ship a standalone `glslc` package (added in 24.04+).
#
# Usage:   scripts/linux/install_lunarg_shaderc.sh
set -euo pipefail

if command -v glslc >/dev/null 2>&1; then
  echo "glslc already present: $(command -v glslc)"
  glslc --version | head -1
  exit 0
fi

export DEBIAN_FRONTEND=noninteractive
sudo apt-get install -y ca-certificates curl gnupg

curl -fsSL https://packages.lunarg.com/lunarg-signing-key-pub.asc |
  sudo gpg --dearmor -o /usr/share/keyrings/lunarg.gpg

echo "deb [signed-by=/usr/share/keyrings/lunarg.gpg] https://packages.lunarg.com/vulkan/ jammy main" |
  sudo tee /etc/apt/sources.list.d/lunarg-vulkan-jammy.list >/dev/null

sudo apt-get update
sudo apt-get install -y shaderc

command -v glslc
glslc --version | head -1
