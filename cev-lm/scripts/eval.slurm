#!/bin/bash
#SBATCH --mem=93g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1    # <- match to OMP_NUM_THREADS
#SBATCH --partition=
#SBATCH --gpus=1
#SBATCH --account=
#SBATCH --job-name=feat_eval
#SBATCH --output="scripts/run_logs/eval.%j.%N.out"
#SBATCH --no-requeue
#SBATCH --time=24:00:00      # hh:mm:ss for the job

# module reset # drop modules and explicitly load the ones needed
# module load default python modtree/gpu openmpi/4.1.2 ucx/1.11.2 gcc/11.2.0 anaconda3_cpu

# module load anaconda3
# conda init bash
eval "$(conda shell.bash hook)"
conda activate sen2

# export MODEL=cev-lm
# export MODEL=gpt3
# export MODEL=mucoco
# export MODEL=ssd-lm
export MODEL=prefix

cd ./cev-lm
python corpus/evaluate_attribute.py $MODEL