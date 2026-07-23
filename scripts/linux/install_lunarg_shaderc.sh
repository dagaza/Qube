#!/usr/bin/env bash
#
# Install Vulkan build toolchain from LunarG on Ubuntu 22.04.
#
# Jammy main repos ship stale Vulkan headers (1.2 era). Current llama.cpp
# ggml-vulkan requires newer headers + glslc, both available from LunarG.
#
# Usage:   scripts/linux/install_lunarg_shaderc.sh
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

if [[ ! -f /etc/apt/sources.list.d/lunarg-vulkan-jammy.list ]]; then
  sudo apt-get install -y ca-certificates curl gnupg
  curl -fsSL https://packages.lunarg.com/lunarg-signing-key-pub.asc |
    sudo gpg --dearmor -o /usr/share/keyrings/lunarg.gpg
  echo "deb [signed-by=/usr/share/keyrings/lunarg.gpg] https://packages.lunarg.com/vulkan/ jammy main" |
    sudo tee /etc/apt/sources.list.d/lunarg-vulkan-jammy.list >/dev/null
fi

sudo apt-get update
sudo apt-get install -y shaderc vulkan-headers libvulkan-dev

command -v glslc
glslc --version | head -1
