#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --account='hk-project-p0021348'
#SBATCH --output='baseline.out'
#SBATCH --job-name='a2a'
#SBATCH --constraint='LSDF'
#SBATCH --exclusive

home_directory='/hkfs/work/workspace/scratch/ke4365-jigsaw'

source $home_directory/.venv/bin/activate
which python
# Change 5-digit MASTER_PORT as you wish, SLURM will raise Error if duplicated with others.
export MASTER_PORT=12342

# Get the first node name as master address.
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export CUDA_LAUNCH_BLOCKING=0

srun python -u $home_directory/experiments/scaling/scaling_scripts/equivalent_twoway.py --config_file config.yaml
