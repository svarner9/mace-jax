# Configuration reference

A run is defined by one `TrainingConfig` (`mace_jax.training.config`). It is built in any of three
equivalent ways and can be written back out as YAML:

```python
config = TrainingConfig(train_file="train.xyz", r_max=5.0, ...)      # Python
config = TrainingConfig.from_yaml("water.yaml")                        # YAML
```

```bash
mace_jax_run_train --train_file train.xyz --r_max 5.0 ...              # command line
mace_jax_run_train --config water.yaml --seed 7                        # YAML + overrides (CLI wins)
mace_jax_run_train --config water.yaml --print-config                  # show the resolved config and stop
```

YAML keys are the option names below, one per line (`hidden_irreps: "128x0e + 128x1o"`,
`radial_MLP: [64, 64, 64]`, `E0s: {1: -13.6, 8: -2041.4}`). On the command line every option takes
`--name value`; underscores or dashes both work (`--max_num_epochs`, `--max-num-epochs`); booleans
are switches (`--swa`) with a `--no-swa` form; lists are comma separated (`--radial_MLP 64,64,64`).
The resolved configuration is written to `<results_dir>/<name>_config.yaml` at the start of every
run, so a run can always be repeated from that file.

## Options

The table is generated from the dataclass (`python -m mace_jax.training.config` prints it), so the
defaults are the code's.

### run

| option | default | meaning |
|---|---|---|
| `name` | `'mace'` | Run name; prefixes every output file. |
| `results_dir` | `'results'` | Directory for logs, metrics, checkpoints and the exported model. |
| `seed` | `123` | Random seed (parameters and data shuffling). |
| `default_dtype` | `'float32'` | Numeric precision of the model and training. Choices: `float32`, `float64`. |
| `device` | `None` | JAX platform: 'cpu', 'cuda', 'tpu' or None for JAX's default. |
| `matmul_precision` | `'highest'` | jax_default_matmul_precision. 'highest' turns TF32 off on NVIDIA GPUs (with TF32 on, MACE float32 forces were off by 1.6e-2 eV/A RMS). None keeps JAX's default. |
| `log_level` | `'INFO'` | Python logging level. Choices: `DEBUG`, `INFO`, `WARNING`. |
| `debug` | `False` | Enable JAX NaN/Inf checks (jax_debug_nans/infs); slow, for diagnosing a diverging run. |

### data

| option | default | meaning |
|---|---|---|
| `train_file` | `None` | Training set: .xyz/.extxyz file(s), or a directory of mace-jax HDF5 shards. |
| `valid_file` | `None` | Validation set (same formats). Omit to split valid_fraction off the training set. |
| `test_file` | `None` | Test set (same formats); evaluated at the end and, with eval_test, every eval_interval. |
| `valid_fraction` | `0.1` | Fraction of the training set held out for validation when valid_file is not given. |
| `energy_key` | `'energy'` | Name of the energy in the XYZ info fields. |
| `forces_key` | `'forces'` | Name of the forces in the XYZ arrays. |
| `stress_key` | `'stress'` | Name of the stress in the XYZ info fields (eV/A^3). |
| `virials_key` | `'virials'` | Name of the virials in the XYZ info fields. |
| `dipole_key` | `'dipole'` | Name of the dipole in the XYZ info fields. |
| `config_type_weights` | `{'Default': 1.0}` | Per config_type loss weights. |
| `E0s` | `'average'` | Isolated-atom reference energies: 'average' (least-squares fit over the training compositions), a {Z: E} dict, or a JSON file path. Isolated-atom frames in the training set (config_type=IsolatedAtom) take precedence. |
| `atomic_numbers` | `None` | Elements the model knows; default: the elements present in train + valid. |
| `statistics_file` | `None` | Reuse an existing statistics.json (E0s, avg_num_neighbors, mean, std) instead of computing one. |
| `data_dir` | `None` | Where the HDF5 conversion of the XYZ input is cached; default <results_dir>/data. |
| `batch_size` | `None` | Target structures per batch; converted to an edge budget from the dataset statistics (see batch_max_edges). |
| `batch_max_edges` | `None` | Edge budget per padded batch (mace-jax batching unit). Overrides batch_size. Default: batch_size 16 worth of edges, or 8192. |
| `num_workers` | `0` | Data-loading worker processes per training process. |
| `shuffle` | `True` | Shuffle the training set every epoch. |

### model

| option | default | meaning |
|---|---|---|
| `model` | `'ScaleShiftMACE'` | Model class. 'ScaleShiftMACE' (torch default) and 'MACELES' put the energy scale/shift inside the model; 'MACE' applies it outside (recorded as output_mean/output_std). Choices: `MACE`, `ScaleShiftMACE`, `MACELES`. |
| `r_max` | `5.0` | Cutoff radius (A). |
| `num_interactions` | `2` | Number of message-passing layers. |
| `max_ell` | `3` | Highest spherical-harmonic degree of the edge features. |
| `correlation` | `3` | Body order of the equivariant product basis minus one (3 = 4-body). |
| `hidden_irreps` | `'128x0e + 128x1o'` | Hidden features, e.g. '128x0e + 128x1o + 128x2e'. Overrides num_channels/max_L. |
| `num_channels` | `None` | Torch shorthand: channels per irrep; with max_L builds hidden_irreps. |
| `max_L` | `None` | Torch shorthand: highest l of the hidden features (with num_channels). |
| `MLP_irreps` | `'16x0e'` | Hidden width of the nonlinear readout. |
| `num_radial_basis` | `8` | Number of Bessel radial basis functions. |
| `num_cutoff_basis` | `5` | Polynomial order of the radial cutoff envelope. |
| `radial_type` | `'bessel'` | Radial basis type. Choices: `bessel`, `gaussian`, `chebyshev`. |
| `radial_MLP` | `[64, 64, 64]` | Hidden layers of the radial MLP. |
| `interaction` | `'RealAgnosticResidualInteractionBlock'` | Interaction block for layers 2..N. Choices: `RealAgnosticResidualInteractionBlock`, `RealAgnosticAttResidualInteractionBlock`, `RealAgnosticInteractionBlock`, `RealAgnosticDensityInteractionBlock`, `RealAgnosticDensityResidualInteractionBlock`, `RealAgnosticResidualNonLinearInteractionBlock`. |
| `interaction_first` | `'RealAgnosticResidualInteractionBlock'` | Interaction block for the first layer. Choices: `RealAgnosticResidualInteractionBlock`, `RealAgnosticAttResidualInteractionBlock`, `RealAgnosticInteractionBlock`, `RealAgnosticDensityInteractionBlock`, `RealAgnosticDensityResidualInteractionBlock`, `RealAgnosticResidualNonLinearInteractionBlock`. |
| `gate` | `'silu'` | Nonlinearity of the readout MLP. Choices: `silu`, `tanh`, `abs`, `None`. |
| `pair_repulsion` | `False` | Add the ZBL pair-repulsion term. |
| `distance_transform` | `'None'` | Radial distance transform. Choices: `None`, `Agnesi`, `Soft`. |
| `use_reduced_cg` | `True` | Reduced Clebsch-Gordan basis in the symmetric contraction (torch default True). Part of the model definition: never change it for a saved model. |
| `scaling` | `'rms_forces_scaling'` | Energy scale/shift: 'rms_forces_scaling' (scale = RMS force, shift = mean interaction energy per atom), 'std_scaling', 'no_scaling'. Choices: `rms_forces_scaling`, `std_scaling`, `no_scaling`. |
| `avg_num_neighbors` | `None` | Override the average neighbor count used to normalize messages (default: from the data). |
| `les_sigma` | `None` | MACELES: Gaussian width of the latent charges (A); default 1.0. |
| `les_dl` | `None` | MACELES: reciprocal-space resolution (A); default 2.0. |
| `les_norm_factor` | `None` | MACELES: LES energy prefactor; default 90.4756 (Coulomb constant in eV*A/e^2). |

### loss

| option | default | meaning |
|---|---|---|
| `loss` | `'weighted'` | Loss function. Choices: `weighted`, `ef`, `forces_only`, `stress`, `huber`, `virials`, `dipole`, `energy_forces_dipole`, `l1l2`, `l1l2energyforces`, `universal`. |
| `energy_weight` | `1.0` | Weight of the energy term. |
| `forces_weight` | `100.0` | Weight of the forces term. |
| `stress_weight` | `0.0` | Weight of the stress term (needs stress labels). |
| `virials_weight` | `0.0` | Weight of the virials term. |
| `dipole_weight` | `0.0` | Weight of the dipole term. |
| `huber_delta` | `0.01` | Delta of the Huber losses. |

### optimizer

| option | default | meaning |
|---|---|---|
| `optimizer` | `'adam'` | Optimizer. Choices: `adam`, `adamw`, `amsgrad`, `sgd`, `schedulefree`. |
| `amsgrad` | `True` | Use AMSGrad (torch default). Equivalent to optimizer=amsgrad. |
| `lr` | `0.01` | Learning rate. |
| `weight_decay` | `5e-07` | Weight decay on the interaction weights. |
| `beta` | `0.9` | Adam beta1. |
| `clip_grad` | `10.0` | Global gradient-norm clip; None disables. |
| `scheduler` | `'ReduceLROnPlateau'` | Learning-rate schedule. Choices: `ReduceLROnPlateau`, `ExponentialLR`, `constant`, `plateau`, `exponential`. |
| `lr_factor` | `0.8` | ReduceLROnPlateau: multiply the learning rate by this on a plateau. |
| `scheduler_patience` | `50` | ReduceLROnPlateau: epochs without validation improvement before reducing. |
| `lr_scheduler_gamma` | `0.9993` | ExponentialLR: per-epoch decay. |
| `ema` | `False` | Evaluate and save an exponential moving average of the parameters. |
| `ema_decay` | `0.99` | EMA decay. |
| `max_num_epochs` | `2048` | Maximum number of epochs. |
| `patience` | `2048` | Early stopping: epochs without validation improvement. |
| `eval_interval` | `1` | Validate every N epochs. |
| `eval_train` | `True` | Also report training-set metrics every eval_interval. |
| `eval_test` | `False` | Also report test-set metrics every eval_interval (always reported at the end). |

### swa

| option | default | meaning |
|---|---|---|
| `swa` | `False` | Stage two: at start_swa switch to the swa_* loss weights and swa_lr, and average parameters. |
| `start_swa` | `None` | Epoch at which stage two starts; default 75% of max_num_epochs. |
| `swa_lr` | `0.001` | Learning rate in stage two. |
| `swa_energy_weight` | `1000.0` | Energy weight in stage two. |
| `swa_forces_weight` | `100.0` | Forces weight in stage two. |
| `swa_stress_weight` | `10.0` | Stress weight in stage two. |
| `swa_virials_weight` | `10.0` | Virials weight in stage two. |
| `swa_dipole_weight` | `1.0` | Dipole weight in stage two. |

### checkpoints

| option | default | meaning |
|---|---|---|
| `checkpoint_every` | `10` | Write a resumable checkpoint every N epochs (0 disables; the best model is always saved). |
| `keep_checkpoints` | `2` | How many periodic checkpoints to keep (0 = all). |
| `save_all_checkpoints` | `False` | Keep every periodic checkpoint (same as keep_checkpoints=0). |
| `restart_latest` | `False` | Resume from the newest checkpoint in <results_dir>/checkpoints if there is one. |
| `resume_from` | `None` | Resume from this checkpoint file. |
| `export_model` | `True` | Export the final model as a mace-jax bundle next to the checkpoints. |

### logging

| option | default | meaning |
|---|---|---|
| `error_table` | `'PerAtomRMSE'` | Which error summary to print each evaluation. Choices: `PerAtomRMSE`, `rel_PerAtomRMSE`, `TotalRMSE`, `PerAtomMAE`, `rel_PerAtomMAE`, `TotalMAE`, `PerAtomRMSEstressvirials`, `PerAtomMAEstressvirials`, `DipoleRMSE`, `DipolePolarRMSE`, `EnergyDipoleRMSE`. |
| `wandb` | `False` | Log to Weights & Biases. |
| `wandb_project` | `None` | W&B project. |
| `wandb_entity` | `None` | W&B entity. |
| `wandb_name` | `None` | W&B run name (default: the run tag). |
| `wandb_dir` | `None` | W&B local directory. |

### runtime

| option | default | meaning |
|---|---|---|
| `enable_cueq` | `False` | Use the cuEquivariance kernels (needs cuequivariance-ops-jax). |
| `launcher` | `'none'` | Multi-GPU launcher: 'none' (one device), 'local' (one process per visible GPU), 'auto'. Choices: `none`, `local`, `auto`. |
| `distributed` | `False` | Initialize jax.distributed (multi-host); see docs/training.md. |

### advanced

| option | default | meaning |
|---|---|---|
| `gin_configs` | `[]` | gin config files applied before the fields above (advanced). |
| `gin_bindings` | `[]` | gin bindings applied last, e.g. "mace_jax.tools.gin_functions.train.log_errors='PerAtomMAE'" (advanced). |

## How the options reach the trainer

The trainer underneath is mace-jax's gin-configured stack (`mace_jax.tools.gin_model`,
`gin_datasets`, `gin_functions`). `TrainingConfig.to_cli_args()` turns the config into the flags of
`mace-jax-train` and `to_gin_bindings()` into gin bindings for what those flags do not cover (the
architecture). `--print-config` shows both. You never need gin, but it is there when you do:

- `gin_configs: [my.gin]` applies gin files before the options above;
- `gin_bindings: ["mace_jax.tools.gin_functions.train.log_errors='PerAtomMAE'"]` applies bindings
  after everything else, so any gin-configurable parameter of the stack can be set.

The upstream gin workflow (`mace-jax-train configs/aspirin.gin ...`) is documented in
[`upstream_README.md`](upstream_README.md) and still works unchanged; models trained either way are
the same objects and export to the same bundles.

## Environment variables

| variable | effect |
|---|---|
| `JAX_PLATFORMS=cpu` | force CPU (tests, debugging) |
| `MACE_JAX_CG_TRANSFORM_CACHE` | directory for the cached Clebsch–Gordan transforms (default: inside the package tree; the cluster launcher points it at `$SCRATCH`) |
| `JAX_COMPILATION_CACHE_DIR` | persistent XLA compile cache; saves minutes on every restart |
| `XLA_PYTHON_CLIENT_PREALLOCATE=false` | do not reserve the whole GPU at import (set by the cluster launcher) |
| `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` | needed by e3nn 0.4.4's constants file under torch ≥ 2.6 (set by the launcher) |
| `WANDB_*` | the usual Weights & Biases settings when `--wandb` is on |
