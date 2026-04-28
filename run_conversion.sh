#!/bin/bash

# ======== SLURM Job Configuration ========
#SBATCH --job-name=mosq-convert           # [REQUIRED] Name of the job as it appears in the job queue
#SBATCH --time=00:30:00                   # [REQUIRED] Wall time limit in HH:MM:SS
#SBATCH --open-mode=append                # Append to output and error logs instead of overwriting
#SBATCH --output=convert-output.log       # [RECOMMENDED] File to write standard output
#SBATCH --error=convert-error.log         # [RECOMMENDED] File to write standard error
#SBATCH --gres=gpu:1                      # [REQUIRED] Request 1 GPU

# ======== Job Execution Steps ========

# Navigate to the working directory where your code is located
cd /data/jjia496/Mosquitoes

# Activate your specific Conda virtual environment safely within a bash script
source /data/jjia496/miniconda3/bin/activate Mosquitoes_env

# Provide the compute node with your Kaggle token so kagglehub can download/verify the dataset
export KAGGLE_API_TOKEN=KGAT_6431cbd1264fa8b86b8bf6e3e7f8fe36

# Run the data format conversion script
python3 convert_to_coco.py