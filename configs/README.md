# configs/

- `molecule.yaml`, `water_les.yaml` — `TrainingConfig` files for `mace_jax_run_train --config ...`
  (torch option names; every option in `docs/configuration.md`).
- `aspirin.gin`, `aspirin_small.gin`, `mace_mp_0b3.gin` — the upstream gin examples for the
  lower-level `mace-jax-train` entry point (`docs/upstream_README.md`). Still supported; the YAML
  route generates bindings of the same kind.
