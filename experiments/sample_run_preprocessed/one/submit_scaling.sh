#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account='hk-project-pai00071'
#SBATCH --output='scaling_1tflops_oneway.out'
#SBATCH --job-name='1_oneway'
#SBATCH --exclusive

home_directory='/hkfs/work/workspace/scratch/ke4365-jigsaw/Jigsaw'

source $home_directory/.venv/bin/activate
which python

export MASTER_PORT=12342
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export CUDA_LAUNCH_BLOCKING=0

srun python -u $home_directory/scaling_scripts/scaling_timing.py --config_file config.yaml
