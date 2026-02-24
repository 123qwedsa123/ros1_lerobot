#!/usr/bin/env bash
# Export dependency snapshots for:
# 1) conda env (lerobot-mujoco by default)
# 2) ROS Noetic/system python environment

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="${SCRIPT_DIR}"
CONDA_ENV_NAME="${1:-lerobot-mujoco}"

echo "[export] root: ${ROOT_DIR}"
echo "[export] out : ${OUT_DIR}"
echo "[export] conda env: ${CONDA_ENV_NAME}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[error] conda command not found."
  exit 1
fi

conda env export -n "${CONDA_ENV_NAME}" --no-builds > "${OUT_DIR}/conda_${CONDA_ENV_NAME}.yml"
conda list -n "${CONDA_ENV_NAME}" --explicit > "${OUT_DIR}/conda_${CONDA_ENV_NAME}_explicit.txt"
conda run -n "${CONDA_ENV_NAME}" python -m pip freeze > "${OUT_DIR}/conda_${CONDA_ENV_NAME}_pip_freeze.txt"

if [ -f /opt/ros/noetic/setup.bash ]; then
  dpkg-query -W -f='${binary:Package}\n' 'ros-noetic-*' | sort > "${OUT_DIR}/ros_noetic_apt_packages.txt"
fi

if [ -x /usr/bin/python3 ]; then
  /usr/bin/python3 -m pip freeze > "${OUT_DIR}/ros_system_python_pip_freeze.txt" || true
fi

echo "[export] done"
