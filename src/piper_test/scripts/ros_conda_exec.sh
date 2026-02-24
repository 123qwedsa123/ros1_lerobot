#!/usr/bin/env bash
set -euo pipefail

CONDA_SH="${CONDA_SH:-/home/jinhe/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-lerobot-mujoco}"

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[ros_conda_exec] conda init script not found: ${CONDA_SH}" >&2
  exit 1
fi

# Keep ROS Python paths but drop Python 3.8 site-packages to avoid
# pulling system numpy into the conda Python 3.10 runtime.
if [[ -n "${PYTHONPATH:-}" ]]; then
  CLEAN_PYTHONPATH=""
  IFS=':' read -r -a _PY_PATHS <<< "${PYTHONPATH}"
  for _p in "${_PY_PATHS[@]}"; do
    [[ -z "${_p}" ]] && continue
    [[ "${_p}" == *"python3.8/site-packages"* ]] && continue
    if [[ -z "${CLEAN_PYTHONPATH}" ]]; then
      CLEAN_PYTHONPATH="${_p}"
    else
      CLEAN_PYTHONPATH="${CLEAN_PYTHONPATH}:${_p}"
    fi
  done
  export PYTHONPATH="${CLEAN_PYTHONPATH}"
fi

# rospkg/catkin_pkg are installed in /usr/lib on this host; expose them
# without appending all of /usr/lib/python3/dist-packages (which carries old numpy).
ROSPKG_SRC="/usr/lib/python3/dist-packages/rospkg"
CATKIN_PKG_SRC="/usr/lib/python3/dist-packages/catkin_pkg"
if [[ ! -d "${ROSPKG_SRC}" ]]; then
  echo "[ros_conda_exec] rospkg not found: ${ROSPKG_SRC}" >&2
  exit 1
fi
if [[ ! -d "${CATKIN_PKG_SRC}" ]]; then
  echo "[ros_conda_exec] catkin_pkg not found: ${CATKIN_PKG_SRC}" >&2
  exit 1
fi

ROS_COMPAT_PY_DIR="/tmp/ros_conda_py_compat"
mkdir -p "${ROS_COMPAT_PY_DIR}"
ln -sfn "${ROSPKG_SRC}" "${ROS_COMPAT_PY_DIR}/rospkg"
ln -sfn "${CATKIN_PKG_SRC}" "${ROS_COMPAT_PY_DIR}/catkin_pkg"
export PYTHONPATH="${ROS_COMPAT_PY_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

exec "$@"
