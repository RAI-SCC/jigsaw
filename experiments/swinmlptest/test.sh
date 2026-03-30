#!/bin/bash
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:full:1
#SBATCH --account=scc
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=64000
#SBATCH --job-name=test_swinmlp
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err

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
echo "Python path: $(which python)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n1)"

python -u scaling_scripts/batch_test.py \
    --configs_dir experiments/swinmlptest \
    --seed 42 \
    --n_visual 6 \
    --threshold 0.5