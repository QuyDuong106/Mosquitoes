#!/usr/bin/env bash
#SBATCH --job-name=mosq-rfdetr
#SBATCH --time=06:00:00
#SBATCH --open-mode=truncate
#SBATCH --output=logs/rfdetr-train-output.log
#SBATCH --error=logs/rfdetr-train-error.log
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#
# Full pipeline: COCO split → rfdetr_dataset/ → train RFDETRSmall (see train_rfdetr_model.py).
#
# Default training behavior (passed explicitly below; extra args from sbatch go last and override):
#   --epochs 50
#   Early stopping on validation mAP: patience 10 epochs, min improvement 0.001 mAP.
#   Disable early stopping: sbatch --export=ALL run_rfdetr_training.sh --no-early-stopping
#
# Submit from your repo clone (so logs and rfdetr_dataset/ land there):
#   cd /path/to/Mosquitoes && sbatch --export=ALL run_rfdetr_training.sh
#   sbatch --chdir=/path/to/Mosquitoes --export=ALL /path/to/Mosquitoes/run_rfdetr_training.sh
#   sbatch --export=ALL run_rfdetr_training.sh --dataset /scratch/you/mosquitoes-compsci760 --epochs 80
#
# Environment (export before sbatch, or use sbatch --export=ALL)
#   MOSQUITOES_DATASET  Dataset ROOT (images) — recommended on HPC
#   MOSQUITOES_LABELS_CSV Optional override path to manual_labels.csv (default: repo copy)
#   KAGGLE_API_TOKEN    Only if you rely on kagglehub instead of MOSQUITOES_DATASET
#   CONDA_BASE          e.g. /path/to/miniconda3 (must contain etc/profile.d/conda.sh)
#   CONDA_ENV           default: Mosquitoes_env
#   PYTHON_BIN          default: python3
#
# If Slurm jobs do not inherit your shell env, use ONE of:
#   sbatch --export=ALL run_rfdetr_training.sh
#   Or create .env.slurm in this repo (not committed) with:
#     export MOSQUITOES_DATASET=/path/to/dataset_root
#     export KAGGLE_API_TOKEN=...   # only if needed
#   Or pass the dataset path on the sbatch line:
#     sbatch run_rfdetr_training.sh --dataset /path/to/dataset_root

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

if _has_cli_dataset_arg "$@"; then
  :
elif [[ -n "${MOSQUITOES_DATASET:-}" || -n "${KAGGLE_API_TOKEN:-}" ]]; then
  :
else
  cat >&2 <<'EOF'
ERROR: No dataset configuration visible inside this job.

  Option A — export before submit (Slurm does not inherit your login shell by default):
    export MOSQUITOES_DATASET=/path/to/dataset_root   # dataset images root
    sbatch --export=ALL run_rfdetr_training.sh

  Option B — put the same export lines in ./.env.slurm next to this script, then:
    sbatch run_rfdetr_training.sh

  Option C — pass the root on the command line:
    sbatch run_rfdetr_training.sh --dataset /path/to/dataset_root

  Option D — Kaggle Hub only (needs token in env or .env.slurm):
    export KAGGLE_API_TOKEN=...
    sbatch --export=ALL run_rfdetr_training.sh
EOF
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! "${PYTHON_BIN}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "ERROR: ${PYTHON_BIN} is too old. train_rfdetr_model.py requires Python >= 3.10." >&2
  echo "Set CONDA_BASE/CONDA_ENV (or PYTHON_BIN) so sbatch uses your newer environment." >&2
  "${PYTHON_BIN}" -V >&2 || true
  exit 1
fi

# Defaults align with train_rfdetr_model.py; append "$@" so sbatch CLI args override last-wins.
DEFAULT_TRAIN_ARGS=(
  --epochs 50
  --early-stopping
  --early-stopping-patience 10
  --early-stopping-min-delta 0.001
)

exec "${PYTHON_BIN}" "${SCRIPTS_DIR}/train_rfdetr_model.py" "${DEFAULT_TRAIN_ARGS[@]}" "$@"
