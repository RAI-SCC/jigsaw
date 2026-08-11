#!/bin/bash
# Run the jigsaw distributed test suite under mpirun.
#
# Distributed tests are excluded from the default `pytest tests` run (and from the
# pre-commit hook) by the test_distributed_* filename convention, so this script is
# the only way they execute. Modelled on torch-harmonics' run_tests.sh.

set -euo pipefail

default_grid_row=1
default_grid_col=1

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            bold=$(tput bold 2>/dev/null || echo "")
            normal=$(tput sgr0 2>/dev/null || echo "")
            echo "Runs the jigsaw distributed test suite."
            echo "${bold}Arguments:${normal}"
            echo "  ${bold}-h    | --help:${normal} Prints this text."
            echo "  ${bold}-row  | --grid_row:${normal} Number of ranks in the row (spatial/sequence) direction."
            echo "  ${bold}-col  | --grid_col:${normal} Number of ranks in the column (channel) direction."
            echo ""
            echo "Set JIGSAW_RUN_SLOW_TESTS=1 to include the large-sequence cases."
            exit 0
            ;;
        -row|--grid_row)
            grid_row="$2"
            shift 2
            ;;
        -col|--grid_col)
            grid_col="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

grid_row=${grid_row:-$default_grid_row}
grid_col=${grid_col:-$default_grid_col}
ngpu=$(( grid_row * grid_col ))

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Running distributed tests on grid row x col = ${grid_row} x ${grid_col} (${ngpu} ranks)"

# MASTER_ADDR / MASTER_PORT are set by the dist_env fixture in tests/conftest.py,
# which also derives rank and world size from mpi4py, so only the grid shape is
# passed through the environment here.
mpirun --allow-run-as-root -np "${ngpu}" bash -c "
    export GRID_ROW=${grid_row};
    export GRID_COL=${grid_col};
    export JIGSAW_RUN_SLOW_TESTS=${JIGSAW_RUN_SLOW_TESTS:-0};
    cd ${repo_root} && python3 -m pytest -ra \
        tests/modules/attention/test_distributed_attention.py \
        tests/modules/conv/test_distributed_convolution.py
    "
