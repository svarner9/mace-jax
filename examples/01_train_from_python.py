"""Train a MACE model from Python: XYZ files in, a bundle out.

Edit the four paths/values in capitals and run the script. It writes everything to RESULTS
(log, per-epoch metrics, checkpoints, the exported model) and prints the validation errors and
the bundle path at the end. The settings are a sensible default for a few-thousand-frame molecular
dataset; `docs/training.md` explains each one.
"""
from mace_jax.training import TrainingConfig, train

TRAIN = "data/train.xyz"          # frames with energy= (eV) in the header and a forces column (eV/A)
VALID = "data/valid.xyz"          # or set valid_file=None and valid_fraction=0.1
TEST = "data/test.xyz"            # optional
RESULTS = "results/my_model"

config = TrainingConfig(
    name="my_model",
    results_dir=RESULTS,
    train_file=TRAIN,
    valid_file=VALID,
    test_file=TEST,
    # model (torch defaults; ScaleShiftMACE puts the energy scale/shift inside the model)
    model="ScaleShiftMACE",
    r_max=5.0,
    num_interactions=2,
    max_ell=3,
    correlation=3,
    hidden_irreps="128x0e + 128x1o",
    E0s="average",                # least-squares fit of the isolated-atom energies; or {Z: E}
    # loss and optimizer
    energy_weight=1.0,
    forces_weight=100.0,
    lr=0.01,
    ema=True,
    # length, stage two, batching, precision
    max_num_epochs=400,
    swa=True,
    start_swa=300,
    swa_energy_weight=1000.0,
    swa_forces_weight=100.0,
    batch_size=16,                # converted to mace-jax's edge budget from the dataset statistics
    default_dtype="float32",      # TF32 is off; use float64 for the last word on accuracy
    seed=1,
)

result = train(config)
print(result.summary())
print("bundle:", result.bundle)   # results/my_model/checkpoints/<tag>.npz (+ .json)
