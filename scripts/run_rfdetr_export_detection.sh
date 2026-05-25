#!/usr/bin/env bash
#SBATCH --job-name=mosq-rfdetr-export-det
#SBATCH --time=00:30:00
#SBATCH --open-mode=truncate
#SBATCH --output=logs/rfdetr-detection-only-export-output.log
#SBATCH --error=logs/rfdetr-detection-only-export-error.log
#
# Detection-only, CPU-only: COCO (*_coco_det.json) + rfdetr_dataset_det/
# Logs: logs/rfdetr-detection-only-export-{output,error}.log
# Does NOT modify train_coco.json, rfdetr_dataset/, or output/.
#
#   cd /path/to/mosquitoes-rf-detr && sbatch --export=ALL scripts/run_rfdetr_export_detection.sh

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
Set MOSQUITOES_DATASET (kagglehub cache root), MOSQUITOES_DATASET_VERSION=3, or --dataset.
EOF
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
exec "${PYTHON_BIN}" "${SCRIPTS_DIR}/export_rfdetr_detection_dataset.py" "$@"
