#!/bin/bash
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:full:1
#SBATCH --account=scc
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=128000
#SBATCH --job-name=swin_jigsaw
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

mkdir -p logs
cd /hkfs/work/workspace_haic/scratch/tq0693-swinmlp/Jigsaw/
module purge
module load compiler/intel/2025.1_llvm
module load mpi/openmpi/4.1

unset PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.:src
source venv/bin/activate

echo "Start time: $(date)"
echo "Running on node: $HOSTNAME"
echo "Python Pfad: $(which python)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n1)"

export MASTER_ADDR=$(hostname)
export MASTER_PORT=12356
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

perun monitor --app_name "swinmlp" --run_id="1024_swinmlp_fix_seed42" scaling_scripts/train_swinmlp.py --config_file>

