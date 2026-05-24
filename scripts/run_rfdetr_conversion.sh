#!/usr/bin/env bash
#SBATCH --job-name=mosq-rfdetr-convert
#SBATCH --time=00:30:00
#SBATCH --open-mode=truncate
#SBATCH --output=logs/rfdetr-convert-output.log
#SBATCH --error=logs/rfdetr-convert-error.log
#
# CPU-only: writes train/val/test COCO JSON under <dataset>/labels/.
# (No GPU required; omitting --gres avoids consuming a GPU for this step.)
#
#   cd /path/to/Mosquitoes && sbatch run_rfdetr_conversion.sh
#   sbatch --export=ALL run_rfdetr_conversion.sh
#   sbatch run_rfdetr_conversion.sh --dataset /path/to/dataset_root
#
# Environment: MOSQUITOES_DATASET and/or KAGGLE_API_TOKEN — same rules as run_rfdetr_training.sh

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
See run_rfdetr_training.sh header for MOSQUITOES_DATASET, KAGGLE_API_TOKEN, .env.slurm, or --dataset.
EOF
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
exec "${PYTHON_BIN}" "${SCRIPTS_DIR}/convert_to_coco.py" "$@"
