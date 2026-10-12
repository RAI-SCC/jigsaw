#!/usr/bin/env bash
#
# Cluster-agnostic launcher: run a command inside the jigsaw container.
#
#   scripts/run.sh python -m my_training_script --foo bar
#   JIGSAW_SITE=horeka scripts/run.sh pytest -m te_required
#
# Everything cluster-specific is read from the environment, normally by sourcing
# a site preset. Resolution order:
#
#   1. $JIGSAW_SITE_FILE   - absolute path to a site env file
#   2. $JIGSAW_SITE        - name of a preset in scripts/sites/<name>.env
#   3. nothing             - the JIGSAW_* vars below must already be exported
#
# Required:
#   JIGSAW_SIF        Path to the .sif image.
#
# Optional:
#   JIGSAW_VENV       Host venv (created --system-site-packages from the image);
#                     its python is used and its site-packages take precedence.
#   JIGSAW_BINDS      Comma-separated bind mounts, e.g. /hkfs,/scratch
#   JIGSAW_PYTHON     Interpreter name when no venv is set (default: python).
#   JIGSAW_APPTAINER  Container runtime (default: apptainer).
#   JIGSAW_MODULES    Space-separated modules to `module load` before launching.
#   JIGSAW_ENV_FILE   Extra --env-file passed straight to apptainer.
#
# The launch contract jigsaw itself relies on (see jigsaw/launch.py): the process
# manager must provide SLURM_PROCID / SLURM_NTASKS / SLURM_LOCALID (srun does),
# and MASTER_ADDR / MASTER_PORT for init_method="env://". This script fills in
# MASTER_ADDR / MASTER_PORT from the SLURM allocation when they are not already
# set, so a bare `srun scripts/run.sh ...` works.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- load the site preset ---------------------------------------------------
if [[ -n "${JIGSAW_SITE_FILE:-}" ]]; then
    site_file="$JIGSAW_SITE_FILE"
elif [[ -n "${JIGSAW_SITE:-}" ]]; then
    site_file="$repo_root/scripts/sites/${JIGSAW_SITE}.env"
else
    site_file=""
fi

if [[ -n "$site_file" ]]; then
    if [[ ! -r "$site_file" ]]; then
        echo "run.sh: site file not found: $site_file" >&2
        echo "        available: $(cd "$repo_root/scripts/sites" 2>/dev/null && ls *.env 2>/dev/null | sed 's/\.env//' | paste -sd' ' -)" >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "$site_file"
fi

# --- validate -------------------------------------------------------------
if [[ -z "${JIGSAW_SIF:-}" ]]; then
    echo "run.sh: JIGSAW_SIF is not set (source a site preset or export it)." >&2
    exit 1
fi
if [[ ! -e "$JIGSAW_SIF" ]]; then
    echo "run.sh: image not found: $JIGSAW_SIF" >&2
    exit 1
fi
if [[ -n "${JIGSAW_VENV:-}" && ! -x "$JIGSAW_VENV/bin/python" ]]; then
    echo "run.sh: JIGSAW_VENV set but $JIGSAW_VENV/bin/python is missing." >&2
    echo "        Build it (see containers/README.md) or unset JIGSAW_VENV to" >&2
    echo "        use the image interpreter + PYTHONPATH=<repo>." >&2
    exit 1
fi
if [[ $# -eq 0 ]]; then
    echo "run.sh: no command given." >&2
    echo "usage: scripts/run.sh <command> [args...]" >&2
    exit 1
fi

# --- optional module loads ----------------------------------------------------
if [[ -n "${JIGSAW_MODULES:-}" ]]; then
    # shellcheck disable=SC1091
    if command -v module >/dev/null 2>&1; then
        # word-splitting is intentional here
        # shellcheck disable=SC2086
        module load $JIGSAW_MODULES
    fi
fi

# --- distributed rendezvous defaults ----------------------------------------
if [[ -z "${MASTER_ADDR:-}" ]]; then
    if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
        MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
    else
        MASTER_ADDR="localhost"
    fi
fi
export MASTER_ADDR
export MASTER_PORT="${MASTER_PORT:-29500}"

# --- TE / NCCL knobs jigsaw wants (only if the caller has not chosen) --------
export NVTE_FRAMEWORK="${NVTE_FRAMEWORK:-pytorch}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

# --- Triton / Inductor runtime-build toolchain ------------------------------
# Triton (and Inductor's C++ wrapper) compile a small shim at runtime the first
# time a kernel is used -- jigsaw hits this in the ring attention fused merge
# and in `xw_twoway`. Two things in the NGC image + HoreKa combination break
# that build; fix both unless the caller opted out with JIGSAW_FIX_TOOLCHAIN=0:
#   * The host shell leaks CC/CXX pointing at an Intel oneAPI `icx`/`icpx` that
#     is not in the image. Drop them so the build falls back to the image gcc.
#   * Triton links against `libcuda.so.1`, but the image's
#     /usr/local/cuda/compat copies are unresolvable in the container mount;
#     apptainer's `--nv` injects a working driver into /.singularity.d/libs;
#     point TRITON_LIBCUDA_PATH there. It is an *in-container* path so it
#     cannot be probed from here -- jigsaw's merge kernel degrades to its eager
#     path if the value is ever wrong, and a caller on a non-apptainer runtime
#     can export the right dir (or JIGSAW_FIX_TOOLCHAIN=0) themselves.
if [[ "${JIGSAW_FIX_TOOLCHAIN:-1}" == "1" ]]; then
    case "${CC:-}" in */icx | */icpx | *oneapi* | *Intel_OneAPI*) unset CC CXX ;; esac
    export TRITON_LIBCUDA_PATH="${TRITON_LIBCUDA_PATH:-/.singularity.d/libs}"
fi

# --- assemble the apptainer invocation ------------------------------------
runtime="${JIGSAW_APPTAINER:-apptainer}"

bind_args=()
if [[ -n "${JIGSAW_BINDS:-}" ]]; then
    IFS=',' read -ra _binds <<< "$JIGSAW_BINDS"
    for b in "${_binds[@]}"; do
        [[ -n "$b" ]] && bind_args+=(--bind "$b")
    done
fi

env_file_args=()
[[ -n "${JIGSAW_ENV_FILE:-}" ]] && env_file_args=(--env-file "$JIGSAW_ENV_FILE")

# NGC images bake some of these into %environment, which beats plain host-env
# passthrough - so force them with --env, which wins over the baked values.
env_args=(
    --env "MASTER_ADDR=$MASTER_ADDR"
    --env "MASTER_PORT=$MASTER_PORT"
    --env "NVTE_FRAMEWORK=$NVTE_FRAMEWORK"
    --env "CUDA_DEVICE_MAX_CONNECTIONS=$CUDA_DEVICE_MAX_CONNECTIONS"
)
for v in NCCL_SOCKET_IFNAME NCCL_DEBUG NVTE_FUSED_ATTN NVTE_FLASH_ATTN \
         WORLD_SIZE RANK LOCAL_RANK TRITON_LIBCUDA_PATH; do
    [[ -n "${!v:-}" ]] && env_args+=(--env "$v=${!v}")
done

# Make the checked-out tree importable even without an editable install.
in_container_pythonpath="$repo_root"
[[ -n "${PYTHONPATH:-}" ]] && in_container_pythonpath="$repo_root:$PYTHONPATH"

# Route bare commands (python, pytest, ...) through the venv when one is set;
# otherwise honour JIGSAW_PYTHON for a bare "python"/"python3".
if [[ -n "${JIGSAW_VENV:-}" && "$1" != /* && -x "$JIGSAW_VENV/bin/$1" ]]; then
    cmd="$1"; shift
    set -- "$JIGSAW_VENV/bin/$cmd" "$@"
elif [[ "$1" == "python" || "$1" == "python3" ]]; then
    shift
    set -- "${JIGSAW_PYTHON:-python}" "$@"
fi

exec "$runtime" exec --nv \
    "${bind_args[@]}" \
    "${env_file_args[@]}" \
    "${env_args[@]}" \
    --env "PYTHONPATH=$in_container_pythonpath" \
    "$JIGSAW_SIF" \
    "$@"
