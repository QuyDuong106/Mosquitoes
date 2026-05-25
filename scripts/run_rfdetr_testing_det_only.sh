#!/usr/bin/env bash
#SBATCH --job-name=mosq-rfdetr-test-det
#SBATCH --time=02:00:00
#SBATCH --open-mode=truncate
#SBATCH --output=logs/rfdetr-detection-only-test-output.log
#SBATCH --error=logs/rfdetr-detection-only-test-error.log
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#
# Detection-only: evaluate RF-DETR (rfdetr_dataset_det/, output_det/).
# Logs: logs/rfdetr-detection-only-test-{output,error}.log
# Predictions: test_predictions-detection-only.json (under SLURM submit dir)

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
  echo "ERROR: ${PYTHON_BIN} is too old. test_rfdetr_model_det.py requires Python >= 3.10." >&2
  echo "Set CONDA_BASE/CONDA_ENV (or PYTHON_BIN) so sbatch uses your newer environment." >&2
  "${PYTHON_BIN}" -V >&2 || true
  exit 1
fi

SLURM_PRED_ARGS=()
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  SLURM_PRED_ARGS=(--save-predictions "${SLURM_SUBMIT_DIR}/test_predictions-detection-only.json")
fi

exec "${PYTHON_BIN}" "${SCRIPTS_DIR}/test_rfdetr_model_det.py" "${SLURM_PRED_ARGS[@]}" "$@"
