#!/bin/bash

# ======== SLURM Job Configuration ========
#SBATCH --job-name=mosq-train             # [REQUIRED] Name of the job as it appears in the job queue
#SBATCH --time=04:00:00                   # [REQUIRED] Wall time limit in HH:MM:SS (Increased for training)
#SBATCH --open-mode=append                # Append to output and error logs instead of overwriting
#SBATCH --output=train-output.log         # [RECOMMENDED] File to write standard output
#SBATCH --error=train-error.log           # [RECOMMENDED] File to write standard error
#SBATCH --gres=gpu:1                      # [REQUIRED] Request 1 GPU

# ======== Job Execution Steps ========

# Navigate to the working directory where your code is located
cd /data/jjia496/Mosquitoes

# Activate your specific Conda virtual environment safely within a bash script
source /data/jjia496/miniconda3/bin/activate Mosquitoes_env

# Provide the compute node with your Kaggle token so kagglehub can access the dataset cache
export KAGGLE_API_TOKEN=KGAT_6431cbd1264fa8b86b8bf6e3e7f8fe36

# Run the model training script
python3 train_mosquito_model.py