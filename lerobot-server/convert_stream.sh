#!/usr/bin/env bash
# Streaming converter launcher:
# watches teleop bag episodes and converts incrementally while recording.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${SCRIPT_DIR}/config/bag_convert_config.yaml"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --help|-h)
      cat <<'EOF'
Usage: bash convert_stream.sh [--config <yaml>] [--python <python_bin>] [extra args...]

This runs incremental bag -> LeRobot conversion and watches for new episodes.
All extra args are passed to convert_stream.py.

Examples:
  bash convert_stream.sh
  bash convert_stream.sh --poll-sec 1.0 --settle-sec 2.0
  bash convert_stream.sh --once
EOF
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! -f "${CONFIG}" ]]; then
  echo "[error] config not found: ${CONFIG}"
  exit 1
fi

# Ensure ROS Python packages are available for Phase 1 dependencies.
if ! "${PYTHON_BIN}" -c "import rosbag" >/dev/null 2>&1; then
  ROS_DISTRO="${ROS_DISTRO:-noetic}"
  if [[ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
    # shellcheck disable=SC1090
    source "/opt/ros/${ROS_DISTRO}/setup.bash"
  fi
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/convert_stream.py" \
  --config "${CONFIG}" \
  "${EXTRA_ARGS[@]}"
