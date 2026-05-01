#!/usr/bin/env bash
#SBATCH --job-name=mosq-test
#SBATCH --time=02:00:00
#SBATCH --open-mode=truncate
#SBATCH --output=test-output.log
#SBATCH --error=test-error.log
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#
# Evaluate RF-DETR on the test split (see test_mosquito_model.py).
#
# Usage
#   Local (executable):
#     chmod +x run_testing.sh
#     ./run_testing.sh
#     ./run_testing.sh --weights output/checkpoint_best_total.pth --worst-overlap 10 --best-overlap 10
#
#   Slurm:
#     cd /path/to/Mosquitoes && sbatch run_testing.sh
#     # Slurm runs the batch script from a spool dir; SCRIPT_DIR uses SLURM_SUBMIT_DIR
#     # (the cwd where you ran sbatch), not the copied script path.
#     # From elsewhere: sbatch --chdir=/path/to/Mosquitoes /path/to/Mosquitoes/run_testing.sh
#     # Extra CLI args: sbatch --wrap 'cd /path/to/Mosquitoes && ./run_testing.sh --max-images 200'
#
# Under Slurm, predictions JSON is written to ${SLURM_SUBMIT_DIR}/test_predictions.json by
# default (sbatch submit directory). Pass --save-predictions /other/path.json to override
# (your path wins because it is passed last).
#
# Environment (optional)
#   CONDA_BASE  Miniconda/Mambaforge root (must contain etc/profile.d/conda.sh)
#   CONDA_ENV   Environment name (default: Mosquitoes_env)
#   PYTHON_BIN  Python executable override (default: python3)
#
# If Slurm jobs do not inherit your shell env, uncomment and set:
#   export CONDA_BASE=/path/to/miniconda3

set -euo pipefail

# Under sbatch the script often runs from /var/spool/slurmd/...; dirname(BASH_SOURCE) is wrong then.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "${SCRIPT_DIR}"

# Fallback for this cluster account if CONDA_BASE is not exported.
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
  echo "ERROR: ${PYTHON_BIN} is too old. test_mosquito_model.py requires Python >= 3.10." >&2
  echo "Set CONDA_BASE/CONDA_ENV (or PYTHON_BIN) so sbatch uses your newer environment." >&2
  "${PYTHON_BIN}" -V >&2 || true
  exit 1
fi

# Slurm: pin predictions output to submit directory (same as post-cd cwd).
SLURM_PRED_ARGS=()
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  SLURM_PRED_ARGS=(--save-predictions "${SLURM_SUBMIT_DIR}/test_predictions.json")
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/test_mosquito_model.py" "${SLURM_PRED_ARGS[@]}" "$@"
