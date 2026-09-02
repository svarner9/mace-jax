#!/usr/bin/env bash
# Install mace-jax (this branch, with les_jax) and its dependencies into a conda environment.
#
#   ./install.sh                  # GPU (CUDA 12) build of JAX, env named "mace-jax"
#   ./install.sh --cpu            # CPU-only JAX (laptops, CI)
#   ./install.sh --cueq           # also install the cuEquivariance CUDA kernels (Hopper+ GPUs)
#   ./install.sh --prefix PATH    # put the env at PATH instead of a named env
#   ./install.sh --no-env         # install into the currently active Python instead
#
# What it does: (1) creates a conda env with python 3.11 (mamba if available, else conda),
# (2) installs JAX, a CPU torch (needed only for the Wigner-3j coefficients, so imported checkpoints
# match torch e3nn's sign convention) and e3nn 0.4.4, (3) installs this package in editable mode
# with the test extras, (4) runs an import check and prints the training CLI's usage line.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

JAX_TARGET="jax[cuda12]"
CUEQ=0
ENV_NAME="mace-jax"
PREFIX=""
NO_ENV=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu) JAX_TARGET="jax"; shift ;;
    --cueq) CUEQ=1; shift ;;
    --name) ENV_NAME="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --no-env) NO_ENV=1; shift ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

if [[ $NO_ENV -eq 0 ]]; then
  CONDA_BIN=$(command -v mamba || command -v conda || true)
  if [[ -z "$CONDA_BIN" ]]; then
    echo "need mamba or conda on PATH (or use --no-env inside an existing Python env)" >&2; exit 2
  fi
  if [[ -n "$PREFIX" ]]; then
    "$CONDA_BIN" create -y -q -p "$PREFIX" python=3.11 pip
    PY="$PREFIX/bin/python"
  else
    "$CONDA_BIN" create -y -q -n "$ENV_NAME" python=3.11 pip
    PY="$("$CONDA_BIN" run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)')"
  fi
else
  PY=$(command -v python)
fi
echo "installing into $PY"
PIP="$PY -m pip"

$PIP install -q --upgrade pip
$PIP install -q "$JAX_TARGET>=0.10.0"
# torch is used on the CPU only, to evaluate Wigner-3j coefficients with the torch e3nn convention.
$PIP install -q torch --index-url https://download.pytorch.org/whl/cpu
$PIP install -q "e3nn==0.4.4"
$PIP install -q -e ".[test]"
if [[ $CUEQ -eq 1 ]]; then
  $PIP install -q "cuequivariance-ops-jax-cu12>=0.10,<0.11"
fi

$PY - <<'PYEOF'
import jax, mace_jax, les_jax
from mace_jax.training import TrainingConfig
print(f"mace_jax {mace_jax.__version__} | jax {jax.__version__} | devices {jax.devices()}")
print("les_jax at", les_jax.__file__)
PYEOF
echo
echo "done. Activate with:  conda activate ${PREFIX:-$ENV_NAME}"
echo "train with:           mace_jax_run_train --help"
echo "run the tests with:   pytest tests"
