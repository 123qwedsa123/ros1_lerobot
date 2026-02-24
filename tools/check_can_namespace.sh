#!/usr/bin/env bash
set -euo pipefail

ifaces=("$@")
if [[ ${#ifaces[@]} -eq 0 ]]; then
  ifaces=(slave1 slave2 master1 master2)
fi

echo "[INFO] netns: $(readlink /proc/self/ns/net)"

if ! command -v ip >/dev/null 2>&1; then
  echo "[WARN] 'ip' command not found in this container (install iproute2 if needed)"
fi

missing=0
for name in "${ifaces[@]}"; do
  path="/sys/class/net/$name"
  if [[ ! -e "$path" ]]; then
    echo "[MISS] $name not found in /sys/class/net"
    missing=1
    continue
  fi
  state="unknown"
  if [[ -r "$path/operstate" ]]; then
    state="$(cat "$path/operstate" 2>/dev/null || echo unknown)"
  fi
  echo "[OK]   $name state=$state"
done

if [[ $missing -ne 0 ]]; then
  echo "[HINT] Missing CAN interfaces usually means the container was not started with --network=host." >&2
  exit 1
fi
