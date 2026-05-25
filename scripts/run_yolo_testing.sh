#!/usr/bin/env bash
#SBATCH --job-name=mosq-yolo-test
#SBATCH --time=02:00:00
#SBATCH --open-mode=truncate
#SBATCH --output=logs/yolo-end-to-end-test-output.log
#SBATCH --error=logs/yolo-end-to-end-test-error.log
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#
# End-to-end (multi-class): evaluate YOLO on the test split (see test_yolo_model.py).
# Logs: logs/yolo-end-to-end-test-{output,error}.log
# Predictions: test_yolo_predictions-end-to-end.json (under SLURM submit dir)
# Weights default: runs/mosquito/yolo_train/weights/best.pt (or runs/detect/...)

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
if ! "${PYTHON_BIN}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "ERROR: ${PYTHON_BIN} is too old. test_yolo_model.py requires Python >= 3.10." >&2
  echo "Set CONDA_BASE/CONDA_ENV (or PYTHON_BIN) so sbatch uses your newer environment." >&2
  "${PYTHON_BIN}" -V >&2 || true
  exit 1
fi

if ! "${PYTHON_BIN}" -c "import ultralytics" 2>/dev/null; then
  echo "ERROR: pip install ultralytics" >&2
  exit 1
fi

SLURM_PRED_ARGS=()
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  SLURM_PRED_ARGS=(--save-predictions "${SLURM_SUBMIT_DIR}/test_yolo_predictions-end-to-end.json")
fi

exec "${PYTHON_BIN}" "${SCRIPTS_DIR}/test_yolo_model.py" "${SLURM_PRED_ARGS[@]}" "$@"
