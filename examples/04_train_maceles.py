"""Train MACE-LES (MACE + latent-charge Ewald electrostatics) on periodic data.

The group's revPBE-D3 water settings (docs/maceles.md): 128-channel two-layer backbone with
max_ell 2 and r_max 4 A, energy:force weights 10:100 switching to 1000:1000 in stage two, EMA,
gradient clipping at 10, batches of ~6 structures, float32. Only `model="MACELES"` differs from a
plain MACE run; the k-grid for the Ewald sum is attached to every batch automatically.
"""
from mace_jax.training import TrainingConfig, train

config = TrainingConfig(
    name="water_les",
    results_dir="results/water_les",
    train_file="data/water_train.xyz",     # periodic frames: Lattice= and pbc="T T T" in the header
    valid_file="data/water_val.xyz",
    model="MACELES",
    r_max=4.0,
    num_interactions=2,
    max_ell=2,
    correlation=3,
    hidden_irreps="128x0e + 128x1o",
    num_radial_basis=8,
    num_cutoff_basis=5,
    energy_weight=10.0,
    forces_weight=100.0,
    lr=0.01,
    clip_grad=10.0,
    ema=True,
    max_num_epochs=400,
    swa=True,
    start_swa=200,
    swa_energy_weight=1000.0,
    swa_forces_weight=1000.0,
    batch_size=6,
    default_dtype="float32",
    seed=42,
)

result = train(config)
print(result.summary())
print("bundle directory:", result.bundle)   # config.json + params.msgpack -> mlmd MACELES(...)
