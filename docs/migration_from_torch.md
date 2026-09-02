# For torch MACE users

The option names are torch `mace_run_train`'s, so a torch command line usually needs only the
executable name changed:

```bash
mace_run_train      --name water --train_file train.xyz --valid_file val.xyz --model ScaleShiftMACE \
                    --r_max 5.0 --num_interactions 2 --hidden_irreps "128x0e + 128x1o" --E0s average \
                    --energy_weight 1 --forces_weight 100 --lr 0.01 --amsgrad --ema --ema_decay 0.99 \
                    --max_num_epochs 400 --swa --start_swa 300 --batch_size 16 --default_dtype float64
mace_jax_run_train  --name water --train_file train.xyz --valid_file val.xyz --model ScaleShiftMACE \
                    --r_max 5.0 --num_interactions 2 --hidden_irreps "128x0e + 128x1o" --E0s average \
                    --energy_weight 1 --forces_weight 100 --lr 0.01 --amsgrad --ema --ema_decay 0.99 \
                    --max_num_epochs 400 --swa --start_swa 300 --batch_size 16 --default_dtype float64
```

Same for `mace_eval_configs --configs test.xyz --model ... --output pred.xyz` →
`mace_jax_eval_configs`.

## What is the same

Model classes (`MACE`, `ScaleShiftMACE`, `MACELES`) and their meaning; the architecture options and
their defaults (`r_max`, `num_interactions`, `max_ell`, `correlation`, `hidden_irreps` or
`num_channels`/`max_L`, `MLP_irreps`, radial basis, interaction blocks, `pair_repulsion`,
`distance_transform`); `E0s` handling (`average`, dict, JSON, isolated-atom frames); `scaling`;
loss names and weights; AMSGrad, EMA, ReduceLROnPlateau, stage two / SWA with its weights and
learning rate; `error_table`; W&B; the output units.

## What differs

| | torch `mace_run_train` | `mace_jax_run_train` |
|---|---|---|
| batches | `--batch_size` structures | packed by an edge budget into fixed shapes (JAX compiles once). `--batch_size` is converted to the budget from the data statistics; `--batch_max_edges` sets it directly. `--valid_batch_size` is accepted and ignored. |
| data loading | XYZ read at every start | XYZ converted once to HDF5 shards + `statistics.json`, cached under `results/data/` |
| first epochs | start immediately | the first step compiles (tens of seconds to a few minutes for a large model); a persistent compile cache (`JAX_COMPILATION_CACHE_DIR`) makes restarts fast |
| float32 | TF32 matmuls on Ampere+ GPUs | TF32 off by default (`--matmul_precision highest`), because it costs 1.6e-2 eV/Å RMS in MACE forces |
| checkpoints | `.pt` state dicts + `.model` pickles | `.ckpt` pickles (parameters + optimizer + architecture) and the exported bundle `.npz` + `.json` (or a directory for MACE-LES). Only bundles carry the architecture; there is no state-dict-only format by design. |
| `--restart_latest` | resumes | resumes from the newest periodic checkpoint in `results/checkpoints/` (`--checkpoint_every` controls how often one is written) |
| `--model MACE` | scale/shift inside the model | applies the energy scale/shift *outside* the module (mace-jax's historical behaviour) and records it in the bundle; prefer `ScaleShiftMACE`, which is torch's form and the default |
| `--scheduler` | `ReduceLROnPlateau`, `ExponentialLR` | the same names (mapped to the JAX schedules); `constant` too |
| loss reductions, stage two, weight decay set | per-atom energy MSE over structures, force MSE over all atoms; stage two evaluates the EMA parameters; decay on interaction `linear`/`skip_tp_full` and product weights | the same (checked against torch 0.3.15 in `BUILD_LOG.md`); `--swa_average_params` is an extra that torch does not have |
| `--stress_weight` for MACELES | supported (torch stress is missing the Ewald cell term) | train on energies and forces; mlmd computes stress correctly in MD |
| multi-GPU | `--distributed` | `--launcher local` (one node), `--distributed` (multi-node) |
| fine-tuning a foundation model | `--foundation_model` | not through this CLI; the gin workflow's `--torch-checkpoint` (see `upstream_README.md`) imports a torch model as the starting point |
| not available | | `--lora`, multi-head fine-tuning options, dipole/polarizability heads through this CLI (the gin workflow has multi-head support) |

## Converting torch models

```bash
mace-jax-from-torch --torch-model MACE-OFF23_small.model --output MACE-OFF23_small.npz
```

writes a bundle that evaluates to ~1e-7 eV/atom of the torch model in float64. Only pickled
`.model` files are accepted (they carry the architecture); a `.pt` state dict is refused, see
[`bundles.md`](bundles.md). Needs the torch `mace` package and a CPU torch.

## Using a model from torch-side code

There is no torch export. The bundle is evaluated by JAX: `ase_calculator` for ASE-driven
workflows, mlmd for MD, `mace-jax-predict` / `mace_jax_eval_configs` for batch evaluation.
