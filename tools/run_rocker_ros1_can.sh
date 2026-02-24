#!/usr/bin/env bash
set -euo pipefail

if ! command -v rocker >/dev/null 2>&1; then
  echo "[ERROR] rocker not found on host PATH" >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "[ERROR] docker not found on host PATH" >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "[ERROR] docker daemon is not reachable from this shell" >&2
  echo "[HINT] run this script on host, not inside an existing container." >&2
  exit 1
fi

image="${ROCKER_IMAGE:-ros:noetic-robot}"
if [[ $# -gt 0 ]]; then
  image="$1"
  shift
fi

workspace_dir="${WORKSPACE_DIR:-$HOME/Desktop/piper_master_slave_ws}"
if [[ ! -d "$workspace_dir" ]]; then
  echo "[WARN] workspace directory not found: $workspace_dir" >&2
fi

cmd=("$@")
if [[ ${#cmd[@]} -eq 0 ]]; then
  cmd=(bash)
fi

echo "[INFO] image=$image"
echo "[INFO] workspace=$workspace_dir"
echo "[INFO] enabling host net namespace + privileged mode for SocketCAN"

rocker \
  --nvidia \
  --x11 \
  --user \
  --home \
  --network=host \
  --privileged \
  --volume /dev:/dev \
  --volume /run/udev:/run/udev:ro \
  --volume "$workspace_dir":"$workspace_dir" \
  -- "$image" "${cmd[@]}"
