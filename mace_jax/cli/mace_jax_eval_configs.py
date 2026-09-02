"""``mace_jax_eval_configs``: evaluate a trained model on an XYZ file (torch ``mace_eval_configs``).

    mace_jax_eval_configs --configs test.xyz --model results/checkpoints/<tag>.npz --output pred.xyz

Writes an extended-XYZ file with the model's energy (``MACE_energy``), forces (``MACE_forces``)
and, for periodic structures, stress (``MACE_stress``) alongside the reference values, and prints
the RMSE / MAE against the reference ``--energy_key`` / ``--forces_key`` if they are present.
Values are in eV, eV/A and eV/A^3 for every model class.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np


def _errors(reference, prediction):
    reference = np.asarray(reference, dtype=float).reshape(-1)
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    delta = prediction - reference
    return float(np.sqrt(np.mean(delta**2))), float(np.mean(np.abs(delta)))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog='mace_jax_eval_configs',
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--configs', required=True, help='XYZ/extxyz file to evaluate')
    parser.add_argument('--model', required=True, help='bundle (.npz/.json, MACE-LES directory) or .ckpt')
    parser.add_argument('--output', required=True, help='extxyz file to write')
    parser.add_argument('--default_dtype', '--default-dtype', choices=['float32', 'float64'], default='float64')
    parser.add_argument('--energy_key', '--energy-key', default='energy')
    parser.add_argument('--forces_key', '--forces-key', default='forces')
    parser.add_argument('--info_prefix', '--info-prefix', default='MACE_')
    parser.add_argument('--compute_stress', '--compute-stress', action='store_true')
    args = parser.parse_args(argv)

    import ase.io  # noqa: PLC0415

    from mace_jax.training.export import ase_calculator  # noqa: PLC0415

    calc = ase_calculator(args.model, dtype=args.default_dtype)
    frames = ase.io.read(args.configs, index=':')
    e_ref, e_pred, f_ref, f_pred, n_atoms = [], [], [], [], []
    for atoms in frames:
        atoms.calc = calc
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces())
        atoms.info[f'{args.info_prefix}energy'] = energy
        atoms.arrays[f'{args.info_prefix}forces'] = forces
        if args.compute_stress and atoms.pbc.any():
            atoms.info[f'{args.info_prefix}stress'] = np.asarray(atoms.get_stress(voigt=True))
        atoms.calc = None
        if args.energy_key in atoms.info:
            e_ref.append(float(atoms.info[args.energy_key]))
            e_pred.append(energy)
            n_atoms.append(len(atoms))
        if args.forces_key in atoms.arrays:
            f_ref.append(np.asarray(atoms.arrays[args.forces_key]))
            f_pred.append(forces)
    ase.io.write(args.output, frames, format='extxyz')
    print(f'{len(frames)} structures -> {args.output}')
    if e_ref:
        n = np.asarray(n_atoms, dtype=float)
        rmse, mae = _errors(np.asarray(e_ref) / n, np.asarray(e_pred) / n)
        print(f'energy per atom: RMSE {1000 * rmse:.3f} meV, MAE {1000 * mae:.3f} meV')
    if f_ref:
        rmse, mae = _errors(np.concatenate(f_ref), np.concatenate(f_pred))
        print(f'forces:          RMSE {1000 * rmse:.3f} meV/A, MAE {1000 * mae:.3f} meV/A')


if __name__ == '__main__':
    main(sys.argv[1:])
