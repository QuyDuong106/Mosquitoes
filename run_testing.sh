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
# Environment (optional)
#   CONDA_BASE  Miniconda/Mambaforge root (must contain etc/profile.d/conda.sh)
#   CONDA_ENV   Environment name (default: Mosquitoes_env)
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

if [[ -n "${CONDA_BASE:-}" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-Mosquitoes_env}"
fi

exec python3 "${SCRIPT_DIR}/test_mosquito_model.py" "$@"
