# Training a model

This page walks through one training run end to end. Every option mentioned is documented with its
default in [`configuration.md`](configuration.md); the torch-MACE user's summary of differences is
[`migration_from_torch.md`](migration_from_torch.md).

## 1. Data

Give the trainer XYZ / extended-XYZ files with the reference energy in the frame header
(`energy=...`, eV) and per-atom forces as a column (`forces`, eV/Å); stress (`stress`, eV/Å³, as
written by ASE) is optional. Names other than `energy`/`forces`/`stress` are selected with
`--energy_key`, `--forces_key`, `--stress_key`.

```
--train_file train.xyz            required; several files may be given
--valid_file valid.xyz            or --valid_fraction 0.1 to split it off the training set
--test_file  test.xyz             optional; evaluated at the end (and every epoch with --eval_test)
```

Isolated-atom reference energies (E0s) come from `--E0s`: `average` (default) fits them by least
squares to the training energies as a function of composition — sufficient when compositions vary
enough; otherwise give them explicitly, `--E0s '{1: -13.6, 8: -2041.4}'` or a JSON file. Frames with
`config_type=IsolatedAtom` in the training file take precedence, as in torch MACE.

The first run converts the XYZ files to HDF5 shards and writes `statistics.json` (E0s, average
neighbor count within `r_max`, energy shift and scale) under `<results_dir>/data/<key>/`. `<key>`
hashes the files and every option that affects the conversion, so a second run with the same input
skips the step and a changed `r_max` or property key does not reuse stale shards. Point `--data_dir`
somewhere else to share the cache between runs, or hand the trainer a directory of shards made by
`mace-jax-preprocess` directly.

**Batching.** JAX compiles the training step for fixed array shapes, so mace-jax packs structures
into batches by an *edge budget* and pads to a fixed size, rather than a fixed number of structures.
`--batch_size N` (torch's meaning) is converted to that budget from the dataset statistics
(N × mean atoms × mean neighbors, 10 % headroom); `--batch_max_edges` sets the budget directly.
The log prints both. Larger budgets use the GPU better; the padding fraction is reported.

## 2. Model

```
--model ScaleShiftMACE            MACE | ScaleShiftMACE | MACELES
--r_max 5.0                       cutoff, Å
--num_interactions 2              message-passing layers
--hidden_irreps "128x0e + 128x1o" or --num_channels 128 --max_L 1
--max_ell 3 --correlation 3
--num_radial_basis 8 --num_cutoff_basis 5
--interaction RealAgnosticResidualInteractionBlock --interaction_first ...
--scaling rms_forces_scaling      energy scale/shift from the data
```

`ScaleShiftMACE` (the default, as in torch) puts the energy scale and shift *inside* the model:
`E = Σ E0 + scale · Σ_i ε_i + shift · N`, with `scale` the RMS force and `shift` the mean interaction
energy per atom of the training set (`rms_forces_scaling`). The saved model therefore returns eV
with no further bookkeeping. `MACELES` adds latent charges and long-range electrostatics
([`maceles.md`](maceles.md)). `MACE` is the earlier mace-jax behaviour: the affine transform is
applied outside the module and recorded in the model config as `output_mean`/`output_std`;
`ase_calculator`, `mace_jax_eval_configs` and `mace-jax-predict` apply it, but a raw
`graphdef.apply` does not. Note that the external transform also scales Σ E0, so the two conventions
are not the same function of the parameters; pick one per project (`ScaleShiftMACE` unless you are
continuing an older `MACE` run).

The full receptive field is `num_interactions × r_max`; size the training structures accordingly.

## 3. Loss and optimizer

```
--loss weighted --energy_weight 1 --forces_weight 100 --stress_weight 0
--optimizer adam --amsgrad --lr 0.01 --weight_decay 5e-7 --clip_grad 10
--scheduler ReduceLROnPlateau --lr_factor 0.8 --scheduler_patience 50
--ema --ema_decay 0.99
--max_num_epochs 500 --patience 100 --eval_interval 1
```

Losses: `weighted` (per-atom energy + forces + stress MSE, the torch default), `ef`, `forces_only`,
`huber`, `universal` (Huber with the MACE-MP weighting), `virials`, `dipole`, `l1l2`. Reductions follow
torch: the energy term is a mean over structures of the per-atom error squared, the force term a
mean over every atom in the batch (so large structures weigh more, as in torch). Only the weights a
loss uses are passed on (a stress weight on `ef` is ignored, not an error). The reported errors follow
`--error_table` (default `PerAtomRMSE`): energies in meV/atom, forces in meV/Å, in physical units
whatever the model's internal scaling.

`--amsgrad` (on by default, like torch) uses the torch-exact AMSGrad variant; the exponential moving
average of the parameters (`--ema`) is what gets evaluated and saved when enabled.

## 4. Stage two (SWA)

```
--swa --start_swa 375 --swa_lr 1e-3 --swa_energy_weight 1000 --swa_forces_weight 100
```

From epoch `start_swa` (default 75 % of `max_num_epochs`) the loss weights switch to the `swa_*`
values and the learning rate to `swa_lr`; evaluation and the saved model keep using the EMA
parameters — exactly torch MACE's stage two (torch updates an `AveragedModel` but never evaluates
or saves it). `--swa_average_params` additionally evaluates and exports the running average of the
parameters collected since the switch; that is a different model from torch's and is off by default. Set `start_swa` for the run you are doing;
copying it from a run with a different `max_num_epochs` puts the switch in the wrong place.

## 5. Outputs, checkpoints, resume

Everything goes to `--results_dir` (default `results/`), prefixed by the run tag
`<YYYYmmdd_HHMM>_<name>`:

```
results/
  <name>_config.yaml              the resolved TrainingConfig
  <tag>.log                       what was printed
  <tag>.metrics                   one JSON record per evaluation (mode=train/valid/eval_test)
  <tag>.gin                       the operative gin configuration (reproducibility)
  data/<key>/                     HDF5 shards, statistics.json, dataset_info.json
  checkpoints/<tag>_best.ckpt     lowest validation loss so far
  checkpoints/<tag>_epoch<N>.ckpt every --checkpoint_every epochs (--keep_checkpoints kept)
  checkpoints/<tag>.npz + .json   the exported model (MACE-LES: checkpoints/<tag>_les/)
```

`.ckpt` files are pickles with parameters, optimizer state, EMA/SWA parameters and the architecture
(`model_config`). `--restart_latest` resumes from the newest periodic checkpoint in the run's
`checkpoints/`; `--resume_from PATH` from a specific one. The exported bundle is what you evaluate
and hand to MD ([`bundles.md`](bundles.md)); `mace_jax_export CKPT OUT` writes one from any
checkpoint (for instance the best one, when the final evaluation parameters are not what you want).

Plot the metrics with `mace-jax-train-plot --path results --keys rmse_e_per_atom,rmse_f`.

## 6. Precision and hardware

`--default_dtype float32` (default) or `float64`. On NVIDIA GPUs JAX evaluates float32 matrix
products in TF32 unless told otherwise; this branch sets `jax_default_matmul_precision='highest'`
for the whole process because the TF32 error in MACE forces is large (1.6e-2 eV/Å RMS, max 0.14,
measured on 128 waters against float64; 9e-6 at full float32 precision). `--matmul_precision None`
restores JAX's default if you want the speed and accept the error.

`--enable_cueq` switches the tensor products to the cuEquivariance kernels (needs
`cuequivariance-ops-jax-cu12`; on the cluster `MACE_JAX_CUEQ=1 env/mace-jax.sh ...`). Numerically
this is a different implementation; check a smoke run against the default before a long training.

Multi-GPU on one node: `--launcher local` starts one process per visible GPU (data-parallel).
Multi-node: `--distributed` with the JAX coordinator variables; see the upstream README.

## 7. Two typical runs

A gas-phase molecule dataset, 128-channel model, stage two at 75 %:

```bash
mace_jax_run_train --name mol --train_file train.xyz --valid_file valid.xyz --test_file test.xyz \
  --model ScaleShiftMACE --r_max 5.0 --num_interactions 2 --hidden_irreps "128x0e + 128x1o" \
  --energy_weight 1 --forces_weight 100 --lr 0.01 --ema \
  --max_num_epochs 400 --swa --start_swa 300 --swa_energy_weight 1000 --swa_forces_weight 100 \
  --default_dtype float64 --batch_size 16
```

Bulk water with long-range electrostatics (MACE-LES), reproducing the group's `water_les_d3` setup:

```bash
mace_jax_run_train --name water_les --train_file train.xyz --valid_file val.xyz \
  --model MACELES --r_max 4.0 --num_interactions 2 --max_ell 2 --hidden_irreps "128x0e + 128x1o" \
  --energy_weight 10 --forces_weight 100 --lr 0.01 --clip_grad 10 --ema \
  --max_num_epochs 400 --swa --start_swa 200 --swa_energy_weight 1000 --swa_forces_weight 1000 \
  --batch_size 6 --default_dtype float32
```

Both are in `configs/` as YAML (`mace_jax_run_train --config configs/water_les.yaml`), and the same
runs from Python are `examples/01_train_from_python.py` and `examples/04_train_maceles.py`.
