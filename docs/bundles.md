# Model files (bundles)

A trained model is saved as a *bundle*: the parameters together with the architecture that
produced them. Rebuilding a MACE model from a bare parameter tree is not supported anywhere in this
code, on purpose: the parameter names and shapes are identical for several different models (see
"What the config records" below), so a wrong guess loads cleanly and gives wrong forces.

## Formats

| model class | files | written by | read by |
|---|---|---|---|
| `MACE`, `ScaleShiftMACE` | `<name>.npz` (flax msgpack of the full module state) + `<name>.json` (architecture) | training (`export_model`, default on), `mace_jax_export`, `mace-jax-from-torch` | `mace_jax.training.load_bundle` / `ase_calculator`, `mace-jax-predict`, `mace_jax_eval_configs`, mlmd `MACE(path)` |
| `MACELES` | directory: `config.json` + `params.msgpack` (flax msgpack of the `nnx.Param` state) | training, `mace_jax_export` | the same, and mlmd `MACELES(path)` |
| training checkpoint | `<tag>_best.ckpt`, `<tag>_epoch<N>.ckpt` (pickle: parameters, optimizer state, EMA/SWA parameters, `model_config`) | training | `--resume_from`, `mace_jax_export`, `load_bundle` |

The MACE-LES directory layout is what `mlmd/tools/convert_maceles_torch.py` produces for torch
MACE-LES models, so JAX-trained and torch-converted MACE-LES models are interchangeable in mlmd.

## Loading

```python
from mace_jax.training import ase_calculator, load_bundle

atoms.calc = ase_calculator("results/checkpoints/<tag>.npz", dtype="float64")   # ASE, eV / eV/Å
bundle = load_bundle("results/checkpoints/<tag>.npz", dtype="float32")         # config, params, graphdef
```

`ase_calculator` is the safe route: it applies `output_mean`/`output_std` for `model='MACE'`
bundles, so every model class returns physical units. If you call `bundle.graphdef.apply(params)`
yourself, feed it the data dictionary from `mace_jax.tools.gin_model._graph_to_data` built from a
*padded* graph (jraph treats the last graph of a batch as padding; an unpadded single graph is
masked out entirely), and apply the affine transform for `MACE` bundles yourself.

In mlmd, put the bundle in `MLMD_MODEL_DIR` and refer to it by name: `MACE("water")`,
`MACELES("water_les")`.

Note that `load_bundle`/`load_model_bundle` set `jax_enable_x64` according to the `dtype` argument
(a process-wide flag). mlmd's calculators restore the previous value; call your own code with that
in mind.

## What the config records, and why

`<name>.json` / `config.json` hold everything `mace_jax.tools.model_builder._build_jax_model` needs
to rebuild the module, plus provenance. The entries that have bitten people:

- **`use_reduced_cg`** — which Clebsch–Gordan basis the symmetric contraction uses. Full and reduced
  bases can have the same dimension, so every parameter still loads under the wrong setting and the
  forces are silently wrong. Never edit it; it is part of the model.
- **`normalize2mom_consts`** — e3nn wraps scalar activations in a second-moment normalization whose
  constant is estimated by Monte Carlo. torch e3nn's value for silu is `1.6791767924`; mace-jax
  estimates `1.6742560863` (0.3 % apart). A model trained in JAX is trained *with* the JAX estimate
  and must be loaded with it; a model converted from torch carries torch's exact value. Either way
  the config stores the number the model was built with and the loader uses it verbatim. (The
  estimate is deterministic: numpy seed 0, 10⁶ float32 samples.)
- **`atomic_inter_scale` / `atomic_inter_shift`** (`ScaleShiftMACE`, `MACELES`) — the energy scale
  and per-atom shift inside the model, from `statistics.json` (`std`, `mean`).
- **`output_mean` / `output_std`** (`MACE`) — the same affine transform applied *outside* the module;
  `1.0`/`0.0` for the other classes.
- **`atomic_numbers`, `atomic_energies`** — element table and E0s (eV); `avg_num_neighbors` — the
  message normalization constant from the training set.
- **`les`** (`MACELES`) — `sigma`, `dl`, `remove_self_interaction`, `norm_factor` of the Ewald sum.
- **`model_class`** (`MACE`, `ScaleShiftMACE`, `LESMACE`) and `torch_model_class` (the name torch
  uses; `LESMACE` maps to `ScaleShiftMACE` for the backbone).
- **`source`** — the training config, statistics file and best checkpoint the bundle came from;
  `dtype` — the training precision.

## Energies: two conventions

With `ScaleShiftMACE`/`MACELES` (default) the model computes
`E = Σ_i E0(Z_i) + scale · Σ_i ε_i + shift · N` — the torch MACE form. With `model='MACE'` the
trainer applies `E = std · (Σ_i E0(Z_i) + Σ_i ε_i) + mean · N` outside the module, i.e. it scales the
E0 sum too. Both fit the data; they are different functions of the parameters, and a model must be
used with the convention it was trained under. The bundle records which one applies, and
`ase_calculator` does the right thing for both.

## Absolute energies in float32

MACE energies are dominated by Σ E0 (often > 99.9 % of the total). In float32 that sum alone carries
round-off of order 1e-3 eV for tens of atoms and grows with system size, and it changes with the
padding of the batch. Compare *interaction* energies (subtract Σ E0 in float64 on the host) or
evaluate in float64 when absolute energies matter; forces are unaffected. mlmd reports energies
relative to Σ E0 for this reason.
