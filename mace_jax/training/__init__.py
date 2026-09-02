"""Train MACE and MACE-LES models in JAX with the torch-MACE workflow.

The three things most users need:

    from mace_jax.training import TrainingConfig, train

    config = TrainingConfig(
        name="water",
        train_file="train.xyz", valid_file="valid.xyz",
        model="ScaleShiftMACE", r_max=5.0, hidden_irreps="128x0e + 128x1o",
        energy_weight=1.0, forces_weight=100.0, max_num_epochs=200,
    )
    result = train(config)
    print(result.bundle)          # the saved model, ready for mace_jax / mlmd

`TrainingConfig` uses the torch ``mace_run_train`` argument names wherever the concept exists in
both codes (``--r_max``, ``--hidden_irreps``, ``--num_interactions``, ``--E0s``, ``--lr``, ``--swa``,
...). The same object is what the ``mace_jax_run_train`` command line builds, and it can be written
to / read from YAML (``TrainingConfig.from_yaml`` / ``to_yaml``).

Behind the API the existing mace-jax stack does the work: XYZ input is converted to the HDF5
streaming format once (cached next to the results), the model, loss and optimizer are configured
through gin, and `mace_jax.tools.train` runs the epochs. Advanced users can still hand gin files or
bindings through ``gin_configs`` / ``gin_bindings``.
"""

from .config import TrainingConfig
from .export import export_bundle, load_bundle
from .run import TrainingResult, train

__all__ = [
    'TrainingConfig',
    'TrainingResult',
    'export_bundle',
    'load_bundle',
    'train',
]
