#!/bin/bash
#SBATCH --mem=93g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1    # <- match to OMP_NUM_THREADS
#SBATCH --partition=
#SBATCH --gpus=1
#SBATCH --account=
#SBATCH --job-name=compute_yelp_data_pairs
#SBATCH --output="pairs.%j.%N.out"
#SBATCH --no-requeue
#SBATCH --time=24:00:00      # hh:mm:ss for the job

# module reset # drop modules and explicitly load the ones needed
# module load default python modtree/gpu openmpi/4.1.2 ucx/1.11.2 gcc/11.2.0 anaconda3_cpu

# module load anaconda3
# conda init bash
eval "$(conda shell.bash hook)"
conda activate sen

python -m pip install numpy==1.21
cd ./attribute_control
export XLA_FLAGS=--xla_gpu_cuda_data_dir=./envs/diff/

python pairwise_data_gen.py
python -m pip install numpy==1.22.1