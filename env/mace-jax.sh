#!/bin/bash
# Cluster launcher: run a command inside the group's JAX Singularity environment with this checkout
# of mace-jax (and les_jax) first on PYTHONPATH.
#
#   env/mace-jax.sh mace_jax_run_train --config configs/water.yaml
#   env/mace-jax.sh python examples/01_train_from_python.py
#   JAX_PLATFORMS=cpu env/mace-jax.sh pytest tests
#
# The Python stack (jax + CUDA libs, flax, e3nn-jax, optax, jraph, h5py, torch-cpu + e3nn 0.4.4 for
# the Wigner-3j coefficients, cuequivariance, ...) comes from an ext3 overlay image; point
# MACE_JAX_OVERLAY at a copy you can read. MACE_JAX_CUEQ=1 runs under the CUDA-12.9 image with the
# cuEquivariance kernels (MACE_JAX_CUEQ_OPS must hold cuequivariance_ops_jax). Compile caches go to
# MACE_JAX_CACHE_DIR (default $SCRATCH/.cache/mace-jax; the home filesystem has tight storage and
# inode quotas on this cluster, so caches must not go to $HOME). Off the cluster, use ./install.sh.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OVERLAY=${MACE_JAX_OVERLAY:-/scratch/slv5250/mace-jax-env/mace-jax-env.ext3}
CACHE=${MACE_JAX_CACHE_DIR:-${SCRATCH:-$HOME}/.cache/mace-jax}
CUEQ_OPS=${MACE_JAX_CUEQ_OPS:-/projects/rps/jjd8110/depablolab/svarner/jay-nemora/mace-jax-port/.cueq_ops}
# optional: a torch MACE checkout, only needed to convert torch checkpoints (mace-jax-from-torch)
TORCH_MACE=${MACE_JAX_TORCH_MACE:-/projects/rps/jjd8110/depablolab/svarner/jay-nemora/code/mace/mace_repo}

if [[ ! -r "$OVERLAY" ]]; then
  echo "cannot read overlay $OVERLAY: set MACE_JAX_OVERLAY to a readable copy, or use ./install.sh" >&2
  exit 2
fi
PYPATH="$HERE"
[[ -d "$TORCH_MACE" ]] && PYPATH="$PYPATH:$TORCH_MACE"
if [[ "${MACE_JAX_CUEQ:-0}" == "1" ]]; then
  [[ -d "$CUEQ_OPS/cuequivariance_ops_jax" ]] || { echo "MACE_JAX_CUEQ=1 but $CUEQ_OPS has no cuequivariance_ops_jax" >&2; exit 2; }
  IMAGE=/share/apps/images/cuda-12.9.1-cudnn-9.10.2-ubuntu-24.04.2.sif
  PYPATH="$CUEQ_OPS:$PYPATH"
else
  IMAGE=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
fi
mkdir -p "$CACHE"

exec singularity exec --nv --overlay "$OVERLAY:ro" "$IMAGE" bash -c '
source /ext3/env.sh
export LD_LIBRARY_PATH=/ext3/miniforge3/lib:$LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH='"$PYPATH"':${PYTHONPATH:-}
export PATH='"$HERE"'/bin:$PATH
export MACE_JAX_CG_TRANSFORM_CACHE='"$CACHE"'/cg
export JAX_COMPILATION_CACHE_DIR='"$CACHE"'/xla
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
exec "$@"' mace-jax "$@"
