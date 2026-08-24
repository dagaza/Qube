#!/usr/bin/env bash
#
# Shared helpers for Linux dist/AppImage smoke tests.
#
# shellcheck shell=bash

smoke_linux_stop_stale_qube() {
  if pgrep -x Qube >/dev/null 2>&1; then
    echo "Stopping stale Qube process(es) from a prior smoke test ..."
    pkill -TERM -x Qube 2>/dev/null || true
    sleep 1
    pkill -KILL -x Qube 2>/dev/null || true
    sleep 0.2
  fi
}

smoke_linux_kill_tree() {
  local pid="${1:?pid required}"
  kill -TERM -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  local _attempt
  for _attempt in $(seq 1 10); do
    if ! kill -0 -"$pid" 2>/dev/null && ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null || true
      smoke_linux_stop_stale_qube
      return 0
    fi
    sleep 0.5
  done
  kill -KILL -"$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  smoke_linux_stop_stale_qube
}

smoke_linux_run_liveness() {
  local seconds="${1:?seconds required}"
  shift
  setsid "$@" &
  local pid=$!
  local _tick
  for _tick in $(seq 1 "$seconds"); do
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      wait "$pid" || true
      smoke_linux_stop_stale_qube
      return 1
    fi
    sleep 1
  done
  smoke_linux_kill_tree "$pid"
  return 0
}
