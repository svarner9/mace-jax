# Installing

Three routes, from most to least self-contained.

## A. `install.sh` — a fresh conda environment (recommended)

```bash
git clone git@github.com:svarner9/mace-jax.git && cd mace-jax && git checkout training
./install.sh                 # CUDA-12 JAX; or  ./install.sh --cpu  on a machine without a GPU
conda activate mace-jax
pytest tests                 # ~5 min on CPU
```

Options: `--cpu` (CPU JAX), `--cueq` (cuEquivariance CUDA kernels, Hopper or newer), `--name NAME` /
`--prefix PATH` (where the env goes), `--no-env` (install into the active Python). Needs `mamba` or
`conda` on `PATH` unless `--no-env` is used, and access to PyPI.

What gets installed: `jax` (with CUDA plugins unless `--cpu`), a **CPU** build of `torch` and
`e3nn==0.4.4` (used only to evaluate Wigner-3j coefficients with the same sign convention as torch
MACE — no GPU torch, and no torch at all at training time), this package in editable mode with its
dependencies (flax, optax, e3nn-jax, jraph, h5py, gin-config, cuequivariance, ...), and the test
extras (pytest, ase). `les_jax` is part of the same package tree.

If your CUDA driver does not support CUDA 12 wheels, change the `JAX_TARGET` line in `install.sh`
to `jax[cuda13]`; nothing else depends on the CUDA version.

## B. By hand, into an existing Python ≥ 3.10

```bash
pip install "jax[cuda12]"                                  # or plain `jax` for CPU
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "e3nn==0.4.4"
pip install -e ".[test]"
```

Check: `python -c "import jax, mace_jax, les_jax; print(jax.devices())"` and
`mace_jax_run_train --help`.

A note on `cuequivariance`: the package metadata pins `cuequivariance>=0.10,<0.11` and
`cuequivariance-jax`. They are pure-Python fallbacks unless the CUDA kernels
(`cuequivariance-ops-jax-cu12`, `--cueq`) are installed; `--enable_cueq` needs the kernels.

## C. The group cluster launcher (no install)

`env/mace-jax.sh <command>` runs `<command>` inside the shared Singularity image with this checkout
(and `les_jax`) first on `PYTHONPATH`:

```bash
env/mace-jax.sh mace_jax_run_train --config configs/water.yaml
JAX_PLATFORMS=cpu env/mace-jax.sh pytest tests
MACE_JAX_CUEQ=1 env/mace-jax.sh mace_jax_run_train ... --enable_cueq
```

It needs read access to the overlay image (`MACE_JAX_OVERLAY`, default the owner's copy under
`/scratch`) — ask for a copy or use route A. Compile caches go to `MACE_JAX_CACHE_DIR` (default
`$SCRATCH/.cache/mace-jax`); never to `$HOME`, whose quota is small. GPU work goes through SLURM;
`runs/slurm/*.sbatch` in the owner's checkout are templates (account, partition, `stdbuf -oL` so the
log survives an external kill, `--time` under the two-hour mark the scheduler enforces on this
account).

## Weights

No model weights are in the repository. Trained models live wherever you put `--results_dir`;
converted torch models wherever `mace-jax-from-torch --output` points. mlmd looks them up by name
under `MLMD_MODEL_DIR`.
