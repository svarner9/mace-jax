# Relation to upstream

Base: `ACEsuit/mace-jax` commit `fe19806ad8077a90975eeb83fdd777a9513e9f0c` (2026-04-25, "Add
explicit CUDA 13 cuEquivariance extras"). The `training` branch is that commit plus the commits
listed by `git log fe19806..training`; this file summarizes what they change and why, so a later
rebase onto a newer upstream knows what to keep.

## Provenance

The group's fork lived in `/projects/rps/jjd8110/depablolab/svarner/jay-nemora/mace-jax-port`
(July 2026), where most changes were never committed: they were applied as a working-tree overlay
placed first on `PYTHONPATH` by that project's `env/maj.sh`. The audited copy vendored into
`mlmd/third_party/mace-jax` (2026-08-27/31) is the source of every fork change here; two further
edits (`_apply_output_normalization`, `output_mean`/`output_std` in checkpoints, a calculator
re-entrancy fix) come from the sibling checkout `mace-jax-port-learn-the-lr`. `les_jax` was untracked
in the port and is vendored in mlmd (`third_party/les-jax`); it is now a package in this repository.

## Deviations from upstream `fe19806`

**Numerics / torch parity**

- `tools/cg.py`: unchanged from upstream — the torch `e3nn.o3.wigner_3j` Clebsch–Gordan
  convention. (An earlier local patch in the group's checkout used `e3nn_jax`'s CG, which differs by
  a sign for some paths; it was reverted because torch-checkpoint parity is mandatory. The PyPI
  `mace_jax` release carries that sign difference.)
- `adapters/cuequivariance/symmetric_contraction.py`: a context (`native_symmetric_contraction_import_mode`)
  tells the torch-weight conversion whether the source used the reduced CG basis; full and reduced
  bases can have equal dimension, so shape inference alone silently picked the wrong one. The
  compiled contraction no longer contains a `lax.cond` + host callback index validator (broke under
  vmap, cost a host round trip); the on-disk CG transform cache is keyed by x32/x64.
- `adapters/e3nn/o3/_spherical_harmonics.py`: double-`where` normalization so exactly-zero vectors
  do not produce NaN gradients.
- `adapters/nnx/torch.py`, `tools/import_from_torch.py`: enter the CG import mode from the target
  module's `use_reduced_cg`.
- `modules/radial.py`: `ZBLBasis`, `AgnesiTransform`, `SoftTransform` index arrays with
  `jnp.asarray(...)`; the numpy indexing failed inside `jax.jit` (upstream fixed the same issue
  independently in dd3f885 / 36cd9ed after `fe19806`).

**Model**

- `modules/models.py`: `MACE._forward_core` holds the shared forward body; `MACE.__call__` and
  `ScaleShiftMACE.__call__` assemble energies from it. Per-layer messages are returned
  (`message_list`); two no-op hook methods exist for subclasses. Forward pass bit-identical.
- `modules/utils.py`: padded edges get a beyond-cutoff sentinel vector and an `edge_mask`
  (zero-length padded edges made second derivatives NaN); the total energy is a graph-mask-weighted
  sum. Consequence: `_graph_to_data` expects *padded* graphs (jraph convention: the last graph is
  padding); every training/prediction/calculator path pads.

**Training stack**

- `tools/amsgrad.py`: `torch_amsgrad` (torch's exact AMSGrad ordering), `zero_nonfinite`,
  `add_decayed_weights_where_grad`; `optax.tree_utils.tree_cast` instead of a private optax helper.
- `tools/train.py`: `nan_to_num(..., posinf=0, neginf=0)` in gradient sanitization.
- `tools/gin_model.py`: `edge_mask`/`padded_edge_length`/`node_mask`/`pbc`/`les_nvec` in the data
  dict; `_apply_output_normalization`; `output_mean`/`output_std` recorded; `model_type`
  (`MACE` | `ScaleShiftMACE` | `MACELES`) with the scale/shift inside the module for the latter two;
  the exported model config records `model_class`, scale/shift, LES parameters and
  `normalize2mom_consts`.
- `tools/model_builder.py`: rebuilds `ScaleShiftMACE` and `LESMACE` from a config.
- `tools/gin_functions.py`: `flags(matmul_precision='highest')` (TF32 off); MAE error tables
  require energy and forces.
- `tools/bundle.py`: trainer checkpoints (`{params, config}` wrapper) load as bundles.
- `data/utils.py`: `GraphGlobals.pbc`, `GraphGlobals.les_nvec`.
- `cli/mace_jax_train.py`: `run_training()` returns the run's outputs; MACE-LES loader wrapping.
- `cli/mace_jax_preprocess.py`: `--scaling` honoured; `run()` returns a summary.
- `cli/mace_jax_predict.py`: applies the recorded output normalization.
- `cli/mace_jax_from_torch.py`, `cli/mace_jax_create_lammps_model.py`: serializable state on disk;
  torch as a soft import.
- `calculators/mace.py`: evaluates a calculator-free copy of the Atoms.

**New**

- `les_jax/`: MACE-LES model, Ewald sum, torch import.
- `mace_jax/training/`: `TrainingConfig`, `train`, data preparation, MACE-LES batching, bundle
  export/load, ASE calculator.
- `cli/mace_jax_run_train.py`, `cli/mace_jax_eval_configs.py`, `cli/mace_jax_export.py`; `bin/`;
  `env/mace-jax.sh`; `install.sh`; `configs/*.yaml`; `examples/`; `docs/`.
- `tests/test_training_api.py`, `tests/test_amsgrad.py`, `tests/test_padded_edges.py`.

## Upstream since `fe19806` (not merged)

14 commits as of 2026-09-02: the `radial.py` jnp.asarray fix (dd3f885, 36cd9ed — already here),
OpenEquivariance integration and a unified equivariance-backend configuration (f54a434, 3080a22),
MACE OEQ config compatibility (594563b), conversion-CLI fixes. A rebase would conflict mainly in
`symmetric_contraction.py`, `gin_model.py` and `mace_jax_from_torch.py`.

## Known limitations carried from the fork

- The e3nn activation-normalization constant is held in a process-global registry
  (`adapters/e3nn/math/_normalize_activation.py`). Building a model imported from torch (exact
  constant) and a fresh JAX model (estimated constant) in the same process makes the second
  registration retroactively update the first model's wrappers. Every bundle records the constant
  it was built with, so files on disk are unambiguous; in one process, build one kind of model. The
  test suite shows the effect as an order-dependent tolerance failure in
  `tests/test_blocks.py::TestNonLinearReadoutBlock` when run after the torch-import tests.
- MACE-LES stress through the generic path misses the Ewald cell derivative (as in torch `les`).
