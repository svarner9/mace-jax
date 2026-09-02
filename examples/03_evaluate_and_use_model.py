"""Use a trained model: ASE calculator, a parity check against reference data, a finite-difference
force check.

BUNDLE is what a training run exported (`results/.../checkpoints/<tag>.npz`, or the MACE-LES
directory `<tag>_les/`); any `.ckpt` checkpoint works too. Energies come back in eV, forces in
eV/A, whatever the model class (`docs/bundles.md`).
"""
import numpy as np
from ase.io import read

from mace_jax.training import ase_calculator, load_bundle
from mace_jax.training.export import describe_bundle

BUNDLE = "results/my_model/checkpoints/20260902_1200_my_model.npz"
TEST = "data/test.xyz"

print(describe_bundle(BUNDLE))
bundle = load_bundle(BUNDLE, dtype="float64")
print("elements:", bundle.config["atomic_numbers"], "r_max:", bundle.config["r_max"], "A")

# 1) parity on the test set
calc = ase_calculator(BUNDLE, dtype="float64")
frames = read(TEST, index=":")
de, df = [], []
for atoms in frames:
    e_ref, f_ref = atoms.get_potential_energy(), atoms.get_forces()   # reference values in the file
    atoms.calc = calc
    de.append((atoms.get_potential_energy() - e_ref) / len(atoms))
    df.append(atoms.get_forces() - f_ref)
de, df = np.array(de), np.concatenate(df)
print(f"{len(frames)} test structures: energy RMSE {1e3 * np.sqrt(np.mean(de**2)):.2f} meV/atom, "
      f"force RMSE {1e3 * np.sqrt(np.mean(df**2)):.2f} meV/A")

# 2) forces are the exact derivative of the energy (finite differences on one atom)
atoms = frames[0]
atoms.calc = calc
i, h = 0, 1e-4
analytic = atoms.get_forces()[i]
numeric = np.zeros(3)
for k in range(3):
    plus, minus = atoms.copy(), atoms.copy()
    plus.positions[i, k] += h
    minus.positions[i, k] -= h
    plus.calc = minus.calc = calc
    numeric[k] = -(plus.get_potential_energy() - minus.get_potential_energy()) / (2 * h)
print("atom 0 force, analytic vs finite difference:", analytic, numeric)

# 3) the same model in mlmd for MD: copy the bundle into MLMD_MODEL_DIR and use MACE("my_model")
#    (MACELES("my_model_les") for a MACE-LES directory bundle).
