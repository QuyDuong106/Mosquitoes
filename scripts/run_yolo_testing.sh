#!/usr/bin/env bash
#SBATCH --job-name=mosq-yolo-test
#SBATCH --time=02:00:00
#SBATCH --open-mode=truncate
#SBATCH --output=logs/yolo-test-output.log
#SBATCH --error=logs/yolo-test-error.log
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#
# Evaluate YOLO on the test split (see test_yolo_model.py).
# Writes test_yolo_predictions.json (per-image boxes) unless --no-save-predictions.
# Weights default: runs/mosquito/.../best.pt or runs/detect/runs/mosquito/.../best.pt

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
SCRIPTS_DIR="${REPO_ROOT}/scripts"
cd "${REPO_ROOT}"

if [[ -z "${CONDA_BASE:-}" && -d "/data/jjia496/miniconda3" ]]; then
  CONDA_BASE="/data/jjia496/miniconda3"
fi

if [[ -n "${CONDA_BASE:-}" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-Mosquitoes_env}"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! "${PYTHON_BIN}" -c "import ultralytics" 2>/dev/null; then
  echo "ERROR: pip install ultralytics" >&2
  exit 1
fi

exec "${PYTHON_BIN}" "${SCRIPTS_DIR}/test_yolo_model.py" "$@"
