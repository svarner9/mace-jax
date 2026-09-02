# mace-jax `training` branch — build log

Running record of decisions, issues and measurements. Newest entries at the bottom.
Handoff that started this: `../HANDOFF_TRAINING.md` (in the owner's `07_md_engine/`).

## 2026-09-02 — survey: what existed, what was authoritative

**Three copies of the fork, one authoritative.** `mlmd/third_party/mace-jax` (vendored, audited,
pushed) = the jay-nemora `mace-jax-port` HEAD `913ba90` + its 12 never-committed working-tree edits
+ the audit's `radial.py` jit patch. The sibling checkout `mace-jax-port-learn-the-lr` carries the
same overlay plus three later edits (`_apply_output_normalization` with `output_mean/std` recorded in
checkpoints, physical-unit `mace-jax-predict`, a calculator re-entrancy fix). Git history for the
fork is four commits; everything else was uncommitted. Decision: construct the branch on upstream
`fe19806` as six logical commits (adapters, modules, tools, cli, tests, les_jax) with the change
rationale in the messages, rather than graft a history that does not exist. `UPSTREAM.md` lists the
deviations.

**Two training front-ends existed.** (a) `mace_jax.cli.mace_jax_train`: argparse over gin, streaming
HDF5, EMA/SWA/plateau/checkpoints/W&B/multi-GPU — used for every POLYMER model
(`04_train/majrun`, `sweep/`, `06_cofix`). (b) `probes/train_paper.py` (6–7k lines, NEMORA-entangled,
architecture in an in-script spec table) — used for the paper's MACE-JAX/MACE-LES-JAX models and the
`water_les_d3` bundle. The gin stack is the base of this branch; from (b) the torch-exact optimizer
pieces (`torch_amsgrad`, `zero_nonfinite`, gated weight decay — already in the fork's `amsgrad.py`)
and the dataset-wide fixed-`num_k` LES batching idea are what carried over.

**Upstream moved** (14 commits past `fe19806`): same `radial.py` fix, OpenEquivariance. Stayed on
`fe19806` per the handoff; rebase later if wanted.

**Baseline test suite of the assembled fork** (CPU, mlmd env): 186 passed, 2 failed, 36 skipped.
Failures: `test_pair_repulsion` — fails on the vendored copy too and *passes* on upstream
`fe19806`: the fork's `_graph_to_data` emits `edge_mask` from jraph, whose convention is "last graph
is padding", so the test's unpadded single graph was entirely masked (energy independent of
distance). Real callers all pad; fixed the test to pad, documented the contract in `docs/bundles.md`.
`test_blocks::TestNonLinearReadoutBlock` — passes alone, fails after the torch-import tests: the
process-global `normalize2mom` registry (see UPSTREAM.md known limitations). Not fixed here
(consequential redesign); recorded.

## 2026-09-02 — decisions for the training layer

1. **Keep gin underneath, present torch's surface.** A `TrainingConfig` dataclass with the
   `mace_run_train` option names generates the argparse parser, reads/writes YAML, and translates
   itself into `mace-jax-train` flags + gin bindings for the architecture. Rewriting the 5k-line gin
   stack was out of scope and would have thrown away the tested trainer; hiding it costs ~600 lines.
   gin files/bindings stay available (`gin_configs`, `gin_bindings`).
2. **XYZ in.** The trainer converts XYZ to HDF5 shards + `statistics.json` itself, cached under
   `results/data/<sha1 of files+options>`. torch users never see the preprocess step.
3. **Scale/shift inside the model.** `model_type='ScaleShiftMACE'` (default) and `'MACELES'` build
   `ScaleShiftBlock(scale=std, shift=mean)` from the statistics and make the external transform the
   identity. Mathematically the same loss for the interaction part; semantically torch's form
   (`E = ΣE0 + std·E_int + mean·N`) instead of the gin stack's `E = std·(ΣE0 + E_int) + mean·N`. This is
   what makes a saved model return eV with no statistics file, and what makes JAX-trained bundles
   drop into mlmd `MACE(...)`. `'MACE'` keeps the old convention for continuity with existing runs
   (POLYMER v0/swC/cc). Observation supporting the change: the POLYMER `cc_swC` run's validation
   energy RMSE stayed at ~420 meV/atom for 50 epochs while forces improved to 94 meV/Å — consistent
   with the external transform scaling ΣE0 (≈ −113 eV/atom × 0.158) and the interaction energy having
   to absorb it; the GPU validation run below tests the new default on the same data.
4. **Bundles are the deliverable.** Training ends by exporting `<tag>.npz + .json` (MACE) or
   `<tag>_les/config.json + params.msgpack` (MACE-LES) — the two formats mlmd already loads.
   `mace_jax_export` does the same from any checkpoint. The config records `model_class`,
   `use_reduced_cg`, `normalize2mom_consts`, scale/shift, LES parameters, provenance.
5. **TF32 off** by default (`flags.matmul_precision='highest'`), process-wide, with the measured
   reason in the docstring (1.6e-2 eV/Å RMS force error on water, mlmd audit).
6. **MACE-LES batching.** Dataset-wide `num_k` from a one-pass scan of the HDF5 cells; per-batch
   host-side `build_kgrid` zero-padded to `num_k`, attached as a new `GraphGlobals.les_nvec` field
   (NamedTuple field with default `None`, so every other path is unchanged). `GraphGlobals.pbc` added
   for the same reason (LESMACE routes periodic graphs to the reciprocal branch; the neighbor list
   never needed it). One static shape ⇒ one compiled program; mixed cell sizes pay the largest grid.
7. **`batch_size` → edge budget** from the dataset statistics (mean atoms × avg neighbors × 1.1),
   so torch's knob works; `batch_max_edges` overrides.
8. **Preprocess `--scaling` honoured** (was silently RMS-forces for every choice) and `run()` returns
   a summary the training API needs.

## 2026-09-02 — verification

**Unit/e2e tests** (`tests/test_training_api.py`, CPU, synthetic EMT Cu/Al clusters and periodic Cu
cells, tiny models, 1–4 epochs): config round trip and torch-name CLI parsing; ScaleShiftMACE run
writes log/metrics/config/checkpoints/bundle, validation loss decreases over 4 epochs, bundle config
records the model definition (`normalize2mom` silu = 1.6742560863), bundle loads standalone and its
forces match central finite differences to < 2e-4 eV/Å (float64), export from a `.ckpt` round-trips,
rerun reuses the cached data; `model='MACE'` records `output_mean/std` equal to the statistics file;
MACE-LES trains on periodic cells, exports the directory bundle, loads through the mlmd-style path
and passes finite differences to < 5e-4 eV/Å; k-grid padding rows are inert and an undersized grid
raises. Full suite after all changes: **202 passed, 36 skipped, 0 failed** (3:52 on a login node).

**GPU validation** (submitted; results appended below when done):
- `runs/slurm/cc_scaleshift.sbatch` — POLYMER C–C dataset (1383 frames, 65-atom octamer), swC
  architecture (128x0e+128x1o+128x2e, r_max 6, float64), `ScaleShiftMACE`, 25 epochs, then
  `mace_jax_eval_configs` on the test set. Comparison point: the 2026-08-07 `cc_swC` run with the
  external-normalization model reached valid F-RMSE 141.6 meV/Å at epoch 20 (E-RMSE 423 meV/atom).
- `runs/slurm/water_les.sbatch` — 1000-frame subset of the revPBE-D3 water set (384 atoms,
  periodic), `water_les_d3` architecture, MACELES, float32, 6 epochs, then evaluation on 200
  validation frames. Checks the LES k-grid plumbing and throughput on a real periodic system.

Cluster notes: runs go through `env/mace-jax.sh` (this checkout first on PYTHONPATH inside the
group's Singularity image; caches under `$SCRATCH/.cache/mace-jax`); `stdbuf -oL` so the log survives
the ~2 h external kill; `--time 1:50`.

## 2026-09-02 — GPU validation, MACE-LES (job 16780201, L40S)

`runs/slurm/water_les.sbatch`: 1000 training / 200 validation frames of revPBE-D3 water (384 atoms,
periodic; 20 validation frames have 768), MACELES 128x0e+128x1o, max_ell 2, r_max 4, float32,
batch_size 6 → `batch_max_edges=67158` (168 batches/epoch), E0s fitted (H −187.86, O −93.93 eV
from the full set in the earlier run; here fitted on the subset), scale = RMS force 0.7916 eV/Å.

- Preprocessing 1000+200 frames: 6 s. k-grid scan: 1200 cells, `num_k = 6859` (19³), all periodic.
- Compile + first epoch 129 s; steady state **14.3 s/epoch** (168 batches, ~12 batches/s).
- Validation: F-RMSE 924 → 1136 → 166 → 110 → 88 → 76 → **68.4 meV/Å** after 6 epochs;
  E-RMSE 77 meV/atom with MAE ≈ RMSE (a per-atom offset the 10:100 energy:force weighting has not
  removed after six epochs on 1000 frames; the full 400-epoch run with stage two is what reaches
  the 0.2 meV/atom class). Loss monotone from epoch 2. Bundle exported
  (`20260902_0243_water_les_les/config.json + params.msgpack`).
- Found while evaluating the bundle: `ase_calculator` did not attach the k-grid for MACE-LES, so
  `mace_jax_eval_configs` would have evaluated the real-space branch only (training and mlmd attach
  their own grids and were correct). Fixed (`MACEJAXCalculator(graph_transform=...)`), the LES e2e
  test now asserts the reciprocal term changes the energy, the job's evaluation step was cancelled
  and resubmitted as `water_les_eval.sbatch` (job 16780533).

## 2026-09-02 — GPU validation, C–C dataset (job 16780024, H200) and a shared-stack E0 bug

`runs/slurm/cc_scaleshift.sbatch`, swC architecture, ScaleShiftMACE, float64, 25 epochs: 52
batches/epoch at `batch_max_edges=32768`, **8.3 s/epoch** (5.1 train + 3.3 eval incl. test set),
789 315 parameters, compile ≈ 7 min (float64, r_max 6, 128x2e). Validation F-RMSE
1154 → 291 (ep 5) → 144 (10) → 96 (20) → 96 meV/Å (25); held-out test 135 meV/Å at epoch 25
(the owner's 2026-08-07 run: 141.6 at epoch 20). `mace_jax_eval_configs` on the exported bundle
reproduces the trainer's numbers (valid 91 vs 96 meV/Å with the final EMA parameters; test 147 vs
135 — same model, the trainer's last test metric is at epoch 25, the bundle is the final EMA state).

**Energy RMSE stayed at 464–470 meV/atom with MAE = RMSE**, from epoch 5 to 25, on train,
validation and test alike; the MACE-LES water run showed the same pattern at 76.6 meV/atom (train
frames −76.8, validation −76.3, 768-atom frames −76.3: a per-atom constant). The owner's 500-epoch
run had it too (410–430 meV/atom, unchanged through stage two at 10× energy weight). The gradient
of the energy loss reaches every readout weight (checked directly), so it was not the loss path.

Cause: `gin_model.model()` builds the model's E0 array as `[E0_dict.get(z, 0.0) for z in
range(num_species)]` — species *indices* used as keys into a dict keyed by *atomic number*. With
Z = [1, 6, 8] only hydrogen's E0 is found (in the wrong slot), carbon and oxygen get 0.0 silently;
the exported cc bundle indeed carried `atomic_energies = [0.0, −15.59, 0.0]` against statistics
{1: −15.59, 6: −299.84, 8: −149.92}. The network then has to produce ~−100 eV/atom per element
through its readouts; it gets within ~0.5 eV/atom and the rest stays as a per-element offset. Every
model trained with the gin stack and a statistics/`--E0s` input (all POLYMER v0/sw*/cc/v1 models)
has this: relative energies and forces are fine, absolute energies and the dissociation asymptote
are not, which is why the POLYMER parity scripts fit a per-element energy offset before comparing.
Upstream `fe19806` has the same lines.

Fix: `_atomic_energies_by_species` maps the dict through `atomic_numbers` (species index i ↔
`atomic_numbers[i]`) and raises on a missing element instead of substituting zero. The e2e test now
asserts the bundle's `atomic_energies` equal the statistics E0s in element order. GPU runs
resubmitted to confirm the energies now converge.

Also found in this pass: `export_bundle` for MACE-LES wrote the initial parameters (flax
`replace_by_pure_dict` fills the State, not the module it came from); `ase_calculator` did not
attach the Ewald k-grid for MACE-LES. Both fixed; the LES test compares bundle vs checkpoint
predictions and asserts the reciprocal-space term is active.
