#!/bin/bash
#SBATCH --partition=advanced-gpu8
#SBATCH --gres=gpu:8
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --output='baseline.out'
#SBATCH --cpus-per-task=19
#SBATCH --job-name='8_a2a'
#SBATCH --constraint='LSDF'

home_directory='/hkfs/work/workspace/scratch/ke4365-jigsaw'
exp_directory=$PWD
cp $home_directory/src/Linear.py $home_directory/train.py $home_directory/src/TestModel.py $exp_directory/src


module purge
module restore pangu

source $home_directory/.venv/bin/activate
which python
# Change 5-digit MASTER_PORT as you wish, SLURM will raise Error if duplicated with others.
export MASTER_PORT=12342

# Get the first node name as master address.
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export CUDA_LAUNCH_BLOCKING=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1 # https://discuss.pytorch.org/t/torch-distributed-all-reduce-causes-memory-trashing/215024
export TORCH_NCCL_HIGH_PRIORITY=1

srun python -u $home_directory/experiments/scaling/scaling_scripts/equivalent_fourway.py --config_file config.yaml
