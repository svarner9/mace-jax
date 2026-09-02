#!/usr/bin/env bash
# Train from the command line with a YAML config, overriding a few options on the command line.
#
# The YAML holds the settings that define the model and the run; the command line changes what
# varies between runs (seed, output directory). Command-line values win over the file.
#
#   examples/02_train_from_yaml.sh                # uses configs/molecule.yaml
#   mace_jax_run_train --config configs/molecule.yaml --print-config   # inspect without training
set -euo pipefail
cd "$(dirname "$0")/.."

mace_jax_run_train --config configs/molecule.yaml \
    --train_file data/train.xyz --valid_file data/valid.xyz \
    --results_dir results/molecule_seed1 --seed 1

# Evaluate the exported model on a test set: energies (meV/atom) and forces (meV/A) errors, plus an
# extxyz file with MACE_energy / MACE_forces next to the reference values.
BUNDLE=$(ls -t results/molecule_seed1/checkpoints/*.npz | head -1)
mace_jax_eval_configs --configs data/test.xyz --model "$BUNDLE" --output results/molecule_seed1/pred_test.xyz
