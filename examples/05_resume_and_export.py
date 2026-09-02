"""Resume an interrupted run, then export a model from a chosen checkpoint.

Jobs on shared clusters get killed; `restart_latest=True` picks up the newest periodic checkpoint in
`<results_dir>/checkpoints/` (written every `checkpoint_every` epochs) and continues with the same
optimizer state, EMA/SWA parameters and epoch counter. Re-running the same script is the whole
restart procedure. The data conversion is cached, so the restart skips it.
"""
from pathlib import Path

from mace_jax.training import TrainingConfig, export_bundle, train
from mace_jax.training.export import describe_bundle

RESULTS = "results/my_model"

config = TrainingConfig.from_yaml(Path(RESULTS) / "my_model_config.yaml")   # written by the first run
config.restart_latest = True
result = train(config)
print(result.summary())

# The run exports its final evaluation parameters. To ship the best-validation checkpoint instead
# (or any epoch checkpoint), export it explicitly:
best = result.best_checkpoint
bundle = export_bundle(best, Path(RESULTS) / "best_model")      # -> best_model.npz + best_model.json
print("exported", bundle, "->", describe_bundle(bundle))
# Same from the shell:  mace_jax_export results/my_model/checkpoints/<tag>_best.ckpt results/my_model/best_model
