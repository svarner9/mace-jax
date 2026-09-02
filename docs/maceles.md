# Training MACE-LES

MACE-LES (Cheng et al.'s Latent Ewald Summation on a MACE backbone) adds long-range electrostatics
to MACE without charge labels: a second readout stack predicts one latent charge `q_i` per atom from
the same node features, and a parameter-free Ewald sum of those charges is added to the energy,

```
E = Σ_i E0(Z_i) + scale · Σ_i ε_i + shift · N + E_LES(q, r, cell)
```

`E_LES` is added *after* the scale/shift block and is not scaled (this is torch `MACELES`'s
convention, and it matters for parity). In JAX the model is `les_jax.model.LESMACE`, a
`ScaleShiftMACE` subclass; the Ewald sum is `les_jax.ewald`, which reproduces torch `les` term by term
(`sigma = 1.0` Å, `dl = 2.0` Å, self-interaction removed, `norm_factor = 90.4756` eV·Å/e²).

## Running it

```bash
mace_jax_run_train --name water_les --train_file train.xyz --valid_file val.xyz \
  --model MACELES --r_max 4.0 --num_interactions 2 --max_ell 2 --hidden_irreps "128x0e + 128x1o" \
  --energy_weight 10 --forces_weight 100 --lr 0.01 --clip_grad 10 --ema \
  --max_num_epochs 400 --swa --start_swa 200 --swa_energy_weight 1000 --swa_forces_weight 1000 \
  --batch_size 6 --default_dtype float32
```

Everything else about the run (data, loss, optimizer, outputs) is as in [`training.md`](training.md).
`--les_sigma`, `--les_dl`, `--les_norm_factor` change the Ewald parameters; the defaults are the
ones every published MACE-LES model uses.

## What happens to the data

The reciprocal-space sum runs over an integer frequency grid whose size depends on the cell
(`N_k = floor(|a| / dl)` per lattice vector). A compiled JAX program needs one static shape, so the
trainer scans the cells of all shards once, takes the largest grid any structure needs, and attaches
a zero-padded grid of that size to every batch (`mace_jax.training.les`). Padded rows have `k = 0`
and contribute nothing, so the energy is exactly torch's for every structure. The cost is that a
dataset mixing very different cell sizes pays the largest grid on every batch — slower, never wrong.

Non-periodic structures (`pbc = F F F`) take the real-space branch (pairwise `erf` Coulomb) instead.
A dataset may mix periodic and non-periodic frames; the periodicity flag travels with each graph.

## What to check

- **Forces against finite differences** on one structure from the exported bundle
  (`tests/test_training_api.py::test_maceles_trains_on_periodic_data_and_exports_directory_bundle`
  does this on a toy system; do it on yours). The LES forces come from differentiating the same
  energy, so they are consistent by construction; this catches data-plumbing mistakes (a wrong cell,
  a missing `pbc`).
- **Stress is not supported for MACE-LES training.** The generic stress path differentiates through
  the atomic positions only and misses the Ewald sum's explicit cell dependence (torch `les` has the
  same gap, ~550 bar on 128 waters). Train on energies and forces; mlmd computes the correct stress
  in MD by a strain derivative.
- **Cell changes in MD.** The k-grid is fixed at construction. mlmd re-checks it when the cell changes
  (NPT) and the LES energy is discontinuous when a k-shell crosses the cutoff; see mlmd's docs.

## Reference point

The group's `water_les_d3` model (revPBE-D3 water, 15 344 training frames of 128 molecules, this
architecture, 400 epochs with stage two at 200) reached 0.21 meV/atom and 6.3 meV/Å on validation,
against 0.26 meV/atom and 7.0 meV/Å for the torch twin trained with the same hyperparameters.
