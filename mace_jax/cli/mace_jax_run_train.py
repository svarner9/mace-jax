"""``mace_jax_run_train``: train a MACE or MACE-LES model from the command line.

Options are those of torch ``mace_run_train`` wherever both codes have the concept; run with
``--help`` for the full list, or put them in a YAML file and pass ``--config``:

    mace_jax_run_train --name water --train_file train.xyz --valid_file valid.xyz \\
        --model ScaleShiftMACE --r_max 5.0 --hidden_irreps "128x0e + 128x1o" \\
        --energy_weight 1 --forces_weight 100 --max_num_epochs 200 --swa --start_swa 150

    mace_jax_run_train --config configs/water.yaml --seed 7

Outputs land in ``--results_dir`` (default ``results/``): ``<tag>.log``, ``<tag>.metrics``
(JSON lines), the operative gin config, ``checkpoints/<tag>_best.ckpt`` plus periodic checkpoints,
and the exported model bundle ``checkpoints/<tag>.npz`` + ``.json`` (``checkpoints/<tag>_les/`` for
MACE-LES).
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from mace_jax.training.config import config_from_args, describe
from mace_jax.training.run import train


def main(argv: Sequence[str] | None = None) -> None:
    config, ns = config_from_args(list(argv) if argv is not None else None)
    if ns.print_config:
        print(describe(config))
        return
    result = train(config)
    print(result.summary())


if __name__ == '__main__':
    main(sys.argv[1:])
