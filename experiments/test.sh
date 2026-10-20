#!/bin/bash
#SBATCH --partition=booster
#SBATCH --gres=gpu:4
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --account='hai_1302'
#SBATCH --output='logs/test_%x_%j.out'
#SBATCH --error='logs/test_%x_%j.err'

# %x_%j for job numbers

# activate your venv
cd ~/jigsaw

module load Python/3.13.5
module load GCC
module load ParaStationMPI

source .venv/bin/activate
which python

# Force Python output to be completely unbuffered
export PYTHONUNBUFFERED=1

# Change 5-digit MASTER_PORT as you wish, SLURM will raise Error if duplicated with others.
export MASTER_PORT=12342

# Get the first node name as master address.
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

# Added the -u flag for unbuffered execution
srun python -u -m pytest MatMul/tests/Test_SUMMA.py
srun python -u -m pytest MatMul/tests/Test_Cannon.py
srun python -u -m pytest MatMul/tests/Test_MGR.py