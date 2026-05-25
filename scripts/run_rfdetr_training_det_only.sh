#!/usr/bin/env bash
#SBATCH --job-name=mosq-rfdetr-train-det
#SBATCH --time=06:00:00
#SBATCH --open-mode=truncate
#SBATCH --output=logs/rfdetr-detection-only-train-output.log
#SBATCH --error=logs/rfdetr-detection-only-train-error.log
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#
# Detection-only RF-DETR. Checkpoints: output_det/ (not output/).
# Logs: logs/rfdetr-detection-only-train-{output,error}.log
#
#   sbatch --export=ALL scripts/run_rfdetr_export_detection.sh
#   sbatch --export=ALL scripts/run_rfdetr_training_det_only.sh --skip-dataset-export

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
SCRIPTS_DIR="${REPO_ROOT}/scripts"
cd "${REPO_ROOT}"

if [[ -f "${REPO_ROOT}/.env.slurm" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/.env.slurm"
  set +a
fi

if [[ -z "${CONDA_BASE:-}" && -d "/data/jjia496/miniconda3" ]]; then
  CONDA_BASE="/data/jjia496/miniconda3"
fi

if [[ -n "${CONDA_BASE:-}" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-Mosquitoes_env}"
fi

_has_cli_dataset_arg() {
  local a
  for a in "$@"; do
    if [[ "$a" == --dataset || "$a" == --dataset=* ]]; then
      return 0
    fi
  done
  return 1
}

if _has_cli_dataset_arg "$@" || [[ -n "${MOSQUITOES_DATASET:-}" || -n "${KAGGLE_API_TOKEN:-}" ]]; then
  :
else
  cat >&2 <<'EOF'
ERROR: No dataset configuration visible inside this job.
Set MOSQUITOES_DATASET or use --skip-dataset-export after export.
EOF
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! "${PYTHON_BIN}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "ERROR: ${PYTHON_BIN} is too old. train_rfdetr_model_det.py requires Python >= 3.10." >&2
  echo "Set CONDA_BASE/CONDA_ENV (or PYTHON_BIN) so sbatch uses your newer environment." >&2
  "${PYTHON_BIN}" -V >&2 || true
  exit 1
fi

DEFAULT_TRAIN_ARGS=(
  --epochs 50
  --early-stopping
  --early-stopping-patience 10
  --early-stopping-min-delta 0.001
)

exec "${PYTHON_BIN}" "${SCRIPTS_DIR}/train_rfdetr_model_det.py" "${DEFAULT_TRAIN_ARGS[@]}" "$@"
