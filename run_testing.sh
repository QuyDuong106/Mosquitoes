#!/bin/bash

# ======== SLURM Job Configuration ========
#SBATCH --job-name=mosq-test              # Name of the job in the queue
#SBATCH --time=02:00:00                   # Wall time limit (evaluation is usually shorter than training)
#SBATCH --open-mode=truncate                # Append to output and error logs
#SBATCH --output=test-output.log          # Standard output log
#SBATCH --error=test-error.log            # Standard error log
#SBATCH --gres=gpu:1                      # GPU for RF-DETR inference
#SBATCH --mem=32G                         # Raise if you still see OOM kills (oom_kill in sacct)

# ======== Job Execution Steps ========

cd /data/jjia496/Mosquitoes

source /data/jjia496/miniconda3/bin/activate Mosquitoes_env

# Optional args (also works with sbatch run_testing.sh -- … on many sites):
#   --weights output/checkpoint_best_total.pth
#   --max-side 1280          # shrink large frames before inference (fewer GPU OOMs)
#   --max-images 500         # quick subset
python3 test_mosquito_model.py "$@"
