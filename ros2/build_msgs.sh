#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate the ROS2 conda environment first."
  echo "Expected example: conda activate quadruped_pympc_ros2_env"
  exit 1
fi

if ! command -v colcon >/dev/null 2>&1; then
  echo "colcon is not available in the current shell."
  echo "Make sure the ROS2 environment is activated before building messages."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${SCRIPT_DIR}/msgs_ws"
PACKAGE_XML="${WORKSPACE_DIR}/src/dls2_interface/package.xml"

if [[ ! -f "${PACKAGE_XML}" ]]; then
  echo "Expected ROS2 interface package at: ${PACKAGE_XML}"
  exit 1
fi

PYTHON_EXECUTABLE="${CONDA_PREFIX}/bin/python"
if [[ ! -x "${PYTHON_EXECUTABLE}" ]]; then
  PYTHON_EXECUTABLE="${CONDA_PREFIX}/bin/python3"
fi

if [[ ! -x "${PYTHON_EXECUTABLE}" ]]; then
  echo "Could not find a Python executable under ${CONDA_PREFIX}/bin."
  exit 1
fi

cd "${WORKSPACE_DIR}"

export Python_EXECUTABLE="${PYTHON_EXECUTABLE}"
export Python_ROOT_DIR="${CONDA_PREFIX}"

colcon build --cmake-args \
  -DPython_EXECUTABLE="${Python_EXECUTABLE}" \
  -DPython_ROOT_DIR="${Python_ROOT_DIR}"

echo
echo "ROS2 messages built successfully."
echo "Next: source ${WORKSPACE_DIR}/install/setup.bash"
