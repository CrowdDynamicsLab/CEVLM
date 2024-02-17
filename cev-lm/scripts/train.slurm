#!/bin/bash
#SBATCH --mem=93g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --partition=
#SBATCH --gpus=1
#SBATCH --account=
#SBATCH --job-name=cevlm_train
#SBATCH --output="scripts/run_logs/train.%j.%N.out"
#SBATCH --no-requeue
#SBATCH --time=48:00:00      # hh:mm:ss for the job

# source ~/anaconda3/etc/profile.d/conda.sh
cd ./cev-lm 
conda activate nedit

export TEXTMORPH_DATA=$(pwd -P)
export PYTHONPATH=.

python textmorph/edit_model/main.py configs/edit_model/edit_baseline.txt


