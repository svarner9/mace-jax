# mace-jax

MACE and MACE-LES interatomic potentials in JAX: training from XYZ files with the torch-MACE
workflow, evaluation, export to bundles that [mlmd](https://github.com/svarner9/mlmd) runs, and
import of torch-trained MACE checkpoints.

```bash
mace_jax_run_train --name water --train_file train.xyz --valid_file valid.xyz \
    --model ScaleShiftMACE --r_max 5.0 --num_interactions 2 --hidden_irreps "128x0e + 128x1o" \
    --energy_weight 1 --forces_weight 100 --max_num_epochs 500 --swa --start_swa 375
```

```python
from mace_jax.training import TrainingConfig, train

result = train(TrainingConfig(
    name="water", train_file="train.xyz", valid_file="valid.xyz",
    model="ScaleShiftMACE", r_max=5.0, num_interactions=2, hidden_irreps="128x0e + 128x1o",
    energy_weight=1.0, forces_weight=100.0, max_num_epochs=500, swa=True, start_swa=375,
))
print(result.summary())      # validation errors and the path of the exported model
```

Either way the run directory holds the log, per-epoch metrics, checkpoints and the trained model as
a bundle (`checkpoints/<tag>.npz` + `.json`; a directory for MACE-LES). Evaluate it, use it from
ASE, or hand it to mlmd for MD:

```bash
mace_jax_eval_configs --configs test.xyz --model results/checkpoints/<tag>.npz --output pred.xyz
```

```python
from mace_jax.training import ase_calculator
atoms.calc = ase_calculator("results/checkpoints/<tag>.npz")      # eV, eV/Å
```

Options are named as in torch `mace_run_train` (`--r_max`, `--hidden_irreps`, `--E0s`, `--lr`,
`--swa`, ...); a torch command line usually needs only the executable name changed. The
differences that matter are listed in [`docs/migration_from_torch.md`](docs/migration_from_torch.md).

## This branch (`training`)

This is the group's fork of [ACEsuit/mace-jax](https://github.com/ACEsuit/mace-jax) at commit
`fe19806` plus:

- the torch-consistent Clebsch–Gordan convention (the PyPI `mace_jax` gives wrong results with
  torch-trained checkpoints), exact full/reduced CG handling on import, padded-edge masking,
  torch-exact AMSGrad, jit-safe ZBL/Agnesi blocks, and the other fixes the group accumulated while
  reproducing torch MACE numbers — see [`UPSTREAM.md`](UPSTREAM.md);
- `les_jax`: MACE-LES (latent charges + Latent Ewald Summation) as a `ScaleShiftMACE` subclass, with
  parity to torch `MACELES`;
- `mace_jax.training`: the training API and CLI above, MACE-LES training, model export, and the
  end-to-end tests in `tests/test_training_api.py`.

## Install

```bash
git clone git@github.com:svarner9/mace-jax.git && cd mace-jax && git checkout training
./install.sh              # conda env "mace-jax": CUDA-12 JAX, torch-cpu + e3nn 0.4.4, this package
conda activate mace-jax
pytest tests              # ~5 min on CPU
```

`./install.sh --cpu` for a machine without a GPU. On the group cluster `env/mace-jax.sh <command>`
runs inside the shared Singularity environment with this checkout on `PYTHONPATH` (no install).
Details, and the pip-by-hand route, in [`docs/install.md`](docs/install.md).

## What is in the box

| | |
|---|---|
| **Models** | `MACE`, `ScaleShiftMACE` (torch default; energy scale/shift inside the model), `MACELES` (MACE-LES, long-range electrostatics from latent charges) |
| **Data** | XYZ/extxyz in; converted once to HDF5 shards + `statistics.json` (E0s, neighbor count, scale/shift) and cached; fixed-shape batches packed by an edge budget so JAX compiles once |
| **Training** | energy/forces/stress losses (weighted, Huber, universal, ...), AMSGrad/Adam/AdamW, ReduceLROnPlateau / exponential schedules, EMA, stage-two (SWA) loss switch and parameter averaging, gradient clipping, early stopping, W&B, checkpoints and resume, multi-GPU |
| **Output** | `.log`, JSON-lines `.metrics`, the operative gin config, `.ckpt` checkpoints (parameters + optimizer + architecture), the exported model bundle |
| **Precision** | float32 or float64; TF32 is turned off (`matmul_precision='highest'`) because it costs 1.6e-2 eV/Å RMS in MACE forces |
| **Torch import** | `mace-jax-from-torch` converts a torch `.model` file to a bundle (MACE-OFF, MACE-MP, your own); parity to ~1e-7 eV/atom in float64 |

Units everywhere: eV, eV/Å, Å, eV/Å³.

## Documentation

- [`docs/training.md`](docs/training.md) — the workflow: data, model, loss, optimizer, stage two, checkpoints, resume, multi-GPU
- [`docs/configuration.md`](docs/configuration.md) — every option with its default; YAML files; the gin escape hatch
- [`docs/maceles.md`](docs/maceles.md) — training MACE-LES: what the model is, what the data needs, what to check
- [`docs/bundles.md`](docs/bundles.md) — model files: formats, what `config.json` records and why (`use_reduced_cg`, activation constants, scale/shift), loading in mace-jax, ASE and mlmd
- [`docs/migration_from_torch.md`](docs/migration_from_torch.md) — for torch MACE users: what is the same, what differs
- [`docs/install.md`](docs/install.md) — install routes and the cluster launcher
- [`examples/`](examples) — runnable scripts: Python training, YAML + CLI, evaluation and ASE use, MACE-LES, resume and export
- [`BUILD_LOG.md`](BUILD_LOG.md) — dated record of the decisions behind this branch and the measurements that back them

The upstream README (the lower-level `mace-jax-train` + gin workflow, `mace-jax-predict`, LAMMPS
export, distributed training) is preserved as [`docs/upstream_README.md`](docs/upstream_README.md);
everything in it still works.

## Layout

```
mace_jax/
  training/        TrainingConfig, train(), data preparation, MACE-LES batching, export, ASE calculator
  cli/             mace_jax_run_train, mace_jax_eval_configs, mace_jax_export (+ the upstream mace-jax-* tools)
  modules/         the models (MACE, ScaleShiftMACE), blocks, losses, radial functions
  tools/           gin-configured trainer (gin_model, gin_datasets, gin_functions, train), bundles, optimizers
  data/            XYZ/HDF5 readers, graph construction, the streaming loader
  adapters/        e3nn / cuEquivariance / nnx adapters (torch-parity numerics live here)
les_jax/           MACE-LES: LESMACE model, Ewald sum, torch import
configs/           example YAML configs (and the upstream gin examples)
examples/          runnable scripts
tests/             pytest suite; test_training_api.py is the end-to-end training test
env/               cluster launcher; bin/ console entry points for PYTHONPATH use
```

## License

MIT (`MIT.md`), as upstream.
