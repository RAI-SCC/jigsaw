#!/bin/bash
#SBATCH --partition=booster
#SBATCH --gres=gpu:4
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --account='hai_1302'
#SBATCH --output='logs/%x_%j.out'
#SBATCH --error='logs/%x_%j.err'

# activate your venv
home_directory="~/."
source $home_directory/.venv/bin/activate
which python
# Change 5-digit MASTER_PORT as you wish, SLURM will raise Error if duplicated with others.
export MASTER_PORT=12342

# Get the first node name as master address.
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

srun --ntasks=4  python -m pytest tests/linear_sync.py


